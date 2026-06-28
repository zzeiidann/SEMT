"""
MLT JointModel Training Script - IMDB & Yelp
HAN + Neural Topic Model (VAE/NGTM) trained jointly.
Loop over 2 datasets x 3 K-topics configurations.
Results saved to mlt_results.csv and mlt_results.png.
"""

import re
import math
import random
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Config

NUM_TOPICS_LIST = [30, 50, 80]
EPOCHS          = 50
BATCH_SIZE      = 32
TOPN            = 10

DATASETS = [
    {
        "name":      "IMDB",
        "url":       "https://raw.githubusercontent.com/zzeiidann/Data/main/IMDB%20Dataset.csv",
        "text_col":  "review",
        "label_col": "sentiment",
        "label_map": {"positive": 1, "negative": 0},
    },
    {
        "name":      "Yelp",
        "url":       "https://raw.githubusercontent.com/zzeiidann/Data/refs/heads/main/yelp_sentiment_30k_balanced.csv",
        "text_col":  None,   # auto-detected below
        "label_col": None,
        "label_map": None,
    },
]


# Preprocessing

STOPWORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>",      " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"&\w+;",          " ", text)
    text = re.sub(r"\.{2,}",         " ", text)
    text = re.sub(r"[^a-z\s]",       " ", text)
    text = re.sub(r"\s+",            " ", text).strip()
    return text

def tokenize(text: str):
    text = clean_text(text)
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 2]

def split_into_sentences(text: str, max_sent_len: int = 50, max_sents: int = 10):
    raw_sents = re.split(r"(?<=[.!?])\s+", text)
    sents = []
    for s in raw_sents:
        tokens = tokenize(s)
        if tokens:
            sents.append(tokens[:max_sent_len])
        if len(sents) >= max_sents:
            break
    if not sents:
        tokens = tokenize(text)
        sents = [tokens[:max_sent_len]] if tokens else [["unknown"]]
    return sents


# Load datasets

def load_dataset(cfg: dict) -> pd.DataFrame:
    print(f"Loading {cfg['name']} ...")
    df = pd.read_csv(cfg["url"])
    df.columns = [c.strip().lower() for c in df.columns]

    if cfg["text_col"] is None:
        text_col = [c for c in df.columns if "review" in c or "text" in c][0]
    else:
        text_col = cfg["text_col"].lower()

    if cfg["label_col"] is None:
        label_col = [c for c in df.columns if "sentiment" in c or "label" in c or "stars" in c][0]
    else:
        label_col = cfg["label_col"].lower()

    df["review"] = df[text_col].apply(clean_text)

    if cfg["label_map"]:
        df["label"] = df[label_col].map(cfg["label_map"])
    else:
        le = LabelEncoder()
        df["label"] = le.fit_transform(df[label_col].astype(str))

    print(f"{cfg['name']} ready - {len(df):,} rows | labels: {df['label'].value_counts().to_dict()}")
    return df[["review", "label"]]


# Vocabulary

class Vocabulary:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, max_size: int = 15000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2id  = {}
        self.id2word  = {}

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        special   = [self.PAD, self.UNK]
        words     = [w for w, c in counter.most_common(self.max_size) if c >= self.min_freq]
        all_words = special + words
        self.word2id = {w: i for i, w in enumerate(all_words)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        print(f"Vocab size: {len(self.word2id)}")
        return self

    def encode(self, word):
        return self.word2id.get(word, self.word2id[self.UNK])

    def __len__(self):
        return len(self.word2id)


# Dataset & DataLoader

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocabulary,
                 topic_vocab_size: int,
                 max_sents: int = 10, max_sent_len: int = 50):
        self.vocab            = vocab
        self.topic_vocab_size = topic_vocab_size
        self.samples = []
        for text, label in zip(texts, labels):
            sents     = split_into_sentences(text, max_sent_len, max_sents)
            enc_sents = [[vocab.encode(w) for w in s] for s in sents]
            bow       = self._make_bow(sents)
            self.samples.append((enc_sents, bow, int(label)))

    def _make_bow(self, sents):
        bow = np.zeros(self.topic_vocab_size, dtype=np.float32)
        for sent in sents:
            for w in sent:
                idx = self.vocab.encode(w)
                if idx < self.topic_vocab_size:
                    bow[idx] += 1
        s = bow.sum()
        if s > 0:
            bow /= s
        return bow

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    enc_sents_list, bows, labels = zip(*batch)
    batch_size   = len(batch)
    max_sents    = max(len(d) for d in enc_sents_list)
    max_sent_len = max((len(s) for d in enc_sents_list for s in d), default=1)

    doc_lengths  = torch.LongTensor([len(d) for d in enc_sents_list])
    padded_docs  = torch.zeros(max_sents, batch_size, max_sent_len, dtype=torch.long)
    sent_lengths = torch.ones(max_sents, batch_size, dtype=torch.long)

    for b, doc in enumerate(enc_sents_list):
        for s_i, sent in enumerate(doc):
            l = len(sent)
            padded_docs[s_i, b, :l] = torch.LongTensor(sent)
            sent_lengths[s_i, b]    = max(l, 1)

    return (
        padded_docs,
        sent_lengths,
        doc_lengths,
        torch.FloatTensor(np.stack(bows)),
        torch.LongTensor(labels),
    )


# Model Architecture

class TopicVAE(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_topics, dropout=0.2):
        super().__init__()
        self.num_topics = num_topics
        self.en1   = nn.Linear(vocab_size, hidden_dim)
        self.en2   = nn.Linear(hidden_dim, hidden_dim)
        self.drop  = nn.Dropout(dropout)
        self.mu_fc = nn.Linear(hidden_dim, num_topics)
        self.lv_fc = nn.Linear(hidden_dim, num_topics)
        self.g1    = nn.Linear(num_topics, num_topics)
        self.g2    = nn.Linear(num_topics, num_topics)
        self.g3    = nn.Linear(num_topics, num_topics)
        self.g4    = nn.Linear(num_topics, num_topics)
        self.g_drop= nn.Dropout(dropout)
        self.de    = nn.Linear(num_topics, vocab_size)

    def encode(self, x):
        h  = F.relu(self.en1(x))
        h  = F.relu(self.en2(h))
        h  = self.drop(h)
        return self.mu_fc(h), self.lv_fc(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def generate(self, z):
        h = torch.tanh(self.g1(z))
        h = torch.tanh(self.g2(h))
        h = torch.tanh(self.g3(h))
        return self.g_drop(self.g4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z   = self.reparameterize(mu, logvar)
        h   = self.generate(z)
        return mu, logvar, F.softmax(self.de(h), dim=-1)

    def get_topic_words(self, topn=10):
        W = self.de.weight.detach().cpu().numpy()
        return [np.argsort(W[:, k])[::-1][:topn].tolist() for k in range(self.num_topics)]


class HAN(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_topics,
                 word_rnn_size, sent_rnn_size, num_classes, dropout=0.3):
        super().__init__()
        self.num_topics = num_topics
        self.embedding  = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.word_gru   = nn.GRU(emb_dim, word_rnn_size, num_layers=2,
                                  batch_first=False, bidirectional=True, dropout=dropout)
        word_out        = word_rnn_size * 2
        self.word_att1  = nn.Linear(word_out, num_topics, bias=False)
        self.word_att2  = nn.Linear(num_topics, 1, bias=False)
        self.sent_gru   = nn.GRU(word_out, sent_rnn_size, num_layers=1,
                                  batch_first=False, bidirectional=True, dropout=dropout)
        sent_out        = sent_rnn_size * 2
        self.sent_att   = nn.Linear(sent_out, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(sent_out, 100),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, num_classes),
            nn.Tanh(),
        )

    def forward(self, padded_docs, sent_lengths, doc_lengths):
        max_sents, batch_size, _ = padded_docs.shape
        word_att_dict = {}
        sent_repr_list = []

        for s_i in range(max_sents):
            words     = padded_docs[s_i]
            emb       = self.embedding(words).permute(1, 0, 2)
            out, _    = self.word_gru(emb)
            att_raw   = self.word_att1(out)

            word_ids  = words.permute(1, 0)
            for w_i in range(word_ids.shape[0]):
                for b_i in range(batch_size):
                    wid = int(word_ids[w_i, b_i].item())
                    if wid > 0:
                        if wid not in word_att_dict:
                            word_att_dict[wid] = att_raw[w_i, b_i].detach()
                        else:
                            word_att_dict[wid] = (word_att_dict[wid] + att_raw[w_i, b_i].detach()) / 2

            att_score = F.softmax(F.relu(self.word_att2(att_raw)), dim=0)
            sent_repr_list.append((out * att_score).sum(dim=0))

        sent_stack  = torch.stack(sent_repr_list, dim=0)
        sent_out, _ = self.sent_gru(sent_stack)
        sent_att    = F.softmax(F.relu(self.sent_att(sent_out)), dim=0)
        doc_repr    = (sent_out * sent_att).sum(dim=0)
        return self.classifier(doc_repr), word_att_dict


class JointModel(nn.Module):
    def __init__(self, vocab_size, topic_vocab_size, emb_dim, hidden_dim,
                 num_topics, word_rnn_size, sent_rnn_size, num_classes, dropout=0.3):
        super().__init__()
        self.topic_model = TopicVAE(topic_vocab_size, hidden_dim, num_topics, dropout)
        self.han         = HAN(vocab_size, emb_dim, num_topics,
                                word_rnn_size, sent_rnn_size, num_classes, dropout)

    def forward(self, padded_docs, sent_lengths, doc_lengths, bows):
        mu, logvar, recon    = self.topic_model(bows)
        logit, word_att_dict = self.han(padded_docs, sent_lengths, doc_lengths)
        return mu, logvar, recon, logit, word_att_dict


# Loss functions

def topic_loss(bow, mu, logvar, recon):
    nll = -torch.sum(bow * torch.log(recon + 1e-10), dim=1)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return (nll + kld).mean(), nll.mean(), kld.mean()

def attention_sharing_loss(word_att_dict, topic_model, kl_loss_fn):
    if not word_att_dict:
        return torch.tensor(0.0, requires_grad=True)
    de_weight = topic_model.de.weight
    total, count = torch.tensor(0.0).to(de_weight.device), 0
    for wid, han_att in word_att_dict.items():
        if wid >= de_weight.shape[0]:
            continue
        tm_att  = de_weight[wid]
        han_att = han_att.to(de_weight.device)
        kl1 = kl_loss_fn(F.log_softmax(tm_att,  dim=0), F.softmax(han_att.detach(), dim=0))
        kl2 = kl_loss_fn(F.log_softmax(han_att, dim=0), F.softmax(tm_att.detach(),  dim=0))
        total  = total - 2.0 / (2.0 + kl1 + kl2)
        count += 1
    return total / max(count, 1)


# Metrics

def classification_metrics(y_true, y_pred):
    return {
        "Accuracy":        accuracy_score(y_true, y_pred),
        "F1_Macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision_Macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_Macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

def compute_tc_npmi(topic_word_ids, corpus_token_ids, topn=10, window=10):
    N = len(corpus_token_ids)
    if N == 0:
        return 0.0
    word_freq, pair_freq = Counter(), Counter()
    for doc in corpus_token_ids:
        for w in set(doc):
            word_freq[w] += 1
        for i in range(len(doc)):
            for j in range(i + 1, min(i + window + 1, len(doc))):
                wi, wj = doc[i], doc[j]
                if wi != wj:
                    pair_freq[(min(wi, wj), max(wi, wj))] += 1
    scores = []
    for topic in topic_word_ids:
        t_scores = []
        for wi, wj in combinations(topic[:topn], 2):
            p_wi = word_freq[wi] / N
            p_wj = word_freq[wj] / N
            p_co = pair_freq[(min(wi, wj), max(wi, wj))] / N
            if p_wi > 0 and p_wj > 0 and p_co > 0:
                pmi  = math.log(p_co / (p_wi * p_wj))
                npmi = pmi / (-math.log(p_co))
                t_scores.append(npmi)
        if t_scores:
            scores.append(np.mean(t_scores))
    return float(np.mean(scores)) if scores else 0.0

def compute_topic_diversity(topic_word_ids, topn=10):
    all_words = [w for topic in topic_word_ids for w in topic[:topn]]
    return len(set(all_words)) / len(all_words) if all_words else 0.0


# Training & evaluation

def train_epoch(model, loader, opt, opt_att, clf_loss_fn, kl_loss_fn,
                clf_w, tm_w, att_interval, device):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        padded_docs, sent_lengths, doc_lengths, bows, labels = [x.to(device) for x in batch]
        mu, logvar, recon, logit, word_att_dict = model(padded_docs, sent_lengths, doc_lengths, bows)
        loss_clf       = clf_loss_fn(logit, labels)
        loss_tm, _, _  = topic_loss(bows, mu, logvar, recon)
        loss           = clf_w * loss_clf + tm_w * loss_tm
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        total_loss += loss.item()

        if (step + 1) % att_interval == 0:
            mu, logvar, recon, logit, word_att_dict = model(padded_docs, sent_lengths, doc_lengths, bows)
            att_loss = attention_sharing_loss(word_att_dict, model.topic_model, kl_loss_fn)
            opt_att.zero_grad()
            att_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt_att.step()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        padded_docs, sent_lengths, doc_lengths, bows, labels = [x.to(device) for x in batch]
        _, _, _, logit, _ = model(padded_docs, sent_lengths, doc_lengths, bows)
        all_preds.extend(logit.argmax(dim=1).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    return classification_metrics(all_labels, all_preds)


def run_experiment(df, dataset_name, num_topics_list,
                   vocab_max=12000, topic_vocab=5000,
                   emb_dim=200, hidden_dim=500,
                   word_rnn=150, sent_rnn=150,
                   clf_w=9.0, tm_w=1.0,
                   epochs=EPOCHS, batch_size=BATCH_SIZE,
                   att_interval=50, topn=TOPN,
                   test_size=0.2, device=DEVICE):
    print(f"\n[DATASET: {dataset_name}] rows: {len(df)}")

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_test   = int(len(df) * test_size)
    test_df  = df.iloc[:n_test]
    train_df = df.iloc[n_test:]

    print("Building vocabulary ...")
    vocab = Vocabulary(max_size=vocab_max, min_freq=2)
    vocab.build(train_df["review"].tolist())
    topic_vocab_size = min(topic_vocab, len(vocab))

    print("Building corpus index for TC-NPMI ...")
    corpus_token_ids = [
        [vocab.encode(w) for w in tokenize(t) if vocab.encode(w) < topic_vocab_size]
        for t in tqdm(train_df["review"].tolist(), desc="corpus")
    ]
    corpus_token_ids = [d for d in corpus_token_ids if d]

    num_classes = df["label"].nunique()
    results = []

    for K in num_topics_list:
        print(f"\n[{dataset_name}] K = {K} topics")

        train_ds = ReviewDataset(train_df["review"].tolist(), train_df["label"].tolist(),
                                  vocab, topic_vocab_size)
        test_ds  = ReviewDataset(test_df["review"].tolist(), test_df["label"].tolist(),
                                  vocab, topic_vocab_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   collate_fn=collate_fn, num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                   collate_fn=collate_fn, num_workers=2, pin_memory=True)

        model = JointModel(
            vocab_size       = len(vocab),
            topic_vocab_size = topic_vocab_size,
            emb_dim          = emb_dim,
            hidden_dim       = hidden_dim,
            num_topics       = K,
            word_rnn_size    = word_rnn,
            sent_rnn_size    = sent_rnn,
            num_classes      = num_classes,
        ).to(device)

        clf_loss_fn = nn.CrossEntropyLoss()
        kl_loss_fn  = nn.KLDivLoss(reduction="batchmean")
        opt         = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
        opt_att     = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)

        best_f1, best_metrics, best_state = 0.0, {}, {}
        for ep in range(1, epochs + 1):
            tr_loss = train_epoch(model, train_loader, opt, opt_att,
                                   clf_loss_fn, kl_loss_fn, clf_w, tm_w, att_interval, device)
            ev = evaluate_model(model, test_loader, device)
            print(f"Epoch {ep}/{epochs} | loss={tr_loss:.4f} | acc={ev['Accuracy']:.4f} | f1={ev['F1_Macro']:.4f}")
            if ev["F1_Macro"] > best_f1:
                best_f1      = ev["F1_Macro"]
                best_metrics = ev
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

        topic_word_ids = model.topic_model.get_topic_words(topn=topn)
        tc_npmi = compute_tc_npmi(topic_word_ids, corpus_token_ids, topn=topn)
        td      = compute_topic_diversity(topic_word_ids, topn=topn)

        row = {
            "Dataset":          dataset_name,
            "K":                K,
            "Accuracy":         round(best_metrics.get("Accuracy", 0), 4),
            "F1_Macro":         round(best_metrics.get("F1_Macro", 0), 4),
            "Precision_Macro":  round(best_metrics.get("Precision_Macro", 0), 4),
            "Recall_Macro":     round(best_metrics.get("Recall_Macro", 0), 4),
            "TC_NPMI":          round(tc_npmi, 4),
            "Topic_Diversity":  round(td, 4),
        }
        results.append(row)
        print(f"\n[{dataset_name}] K={K} best results:")
        for k, v in row.items():
            if k not in ("Dataset", "K"):
                print(f"  {k:20s}: {v}")

    return results


# Main

all_results = []

for ds_cfg in DATASETS:
    df = load_dataset(ds_cfg)
    results = run_experiment(df=df, dataset_name=ds_cfg["name"], num_topics_list=NUM_TOPICS_LIST)
    all_results.extend(results)


# Results table

results_df = pd.DataFrame(all_results)
print("\nRESULTS SUMMARY")
print(results_df.to_string(index=False))

results_df.to_csv("mlt_results.csv", index=False)
print("\nSaved -> mlt_results.csv")


# Plot

metrics_to_plot = ["Accuracy", "F1_Macro", "Precision_Macro", "Recall_Macro", "TC_NPMI", "Topic_Diversity"]
colors  = {"IMDB": "#2196F3", "Yelp": "#FF5722"}
markers = {"IMDB": "o",       "Yelp": "s"}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    for ds in results_df["Dataset"].unique():
        sub = results_df[results_df["Dataset"] == ds]
        ax.plot(sub["K"], sub[metric],
                marker=markers.get(ds, "o"), color=colors.get(ds, "gray"),
                linewidth=2, markersize=8, label=ds)
        for _, row in sub.iterrows():
            ax.annotate(f"{row[metric]:.3f}", (row["K"], row[metric]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
    ax.set_title(metric.replace("_", " "), fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Topics (K)", fontsize=10)
    ax.set_xticks(NUM_TOPICS_LIST)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("MLT JointModel - IMDB vs Yelp x K Topics", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("mlt_results.png", dpi=150, bbox_inches="tight")
print("Plot saved -> mlt_results.png")