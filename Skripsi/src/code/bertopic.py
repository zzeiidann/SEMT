"""
BERTopic Training Script - IMDB & Yelp
Loop over 2 datasets x 3 n_topics configurations.
Results printed as a table and saved to bertopic_results.csv.
"""

import re
import random
import os
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from collections import defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import umap
import hdbscan

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Config

BERT_MODEL     = "bert-base-uncased"
N_TOPICS_LIST  = [30, 50, 80]
SAMPLE_FRAC    = 1.0  # pakai semua data, set < 1.0 kalau mau subsample

UMAP_PARAMS = dict(
    n_neighbors  = 15,
    n_components = 5,
    min_dist     = 0.0,
    metric       = "cosine",
    random_state = SEED,
)

HDBSCAN_PARAMS = dict(
    min_cluster_size        = 10,
    min_samples             = 5,
    metric                  = "euclidean",
    cluster_selection_method= "eom",
    prediction_data         = True,
)

VECTORIZER_PARAMS = dict(
    stop_words  = "english",
    ngram_range = (1, 2),
    min_df      = 10,
    max_df      = 0.8,
)

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
        "text_col":  "review",
        "label_col": "sentiment",
        "label_map": {"positive": 1, "negative": 0},
    },
]

BASE_STOP = set(ENGLISH_STOP_WORDS).union({
    "the", "and", "to", "was", "it", "of", "is", "in",
    "for", "that", "we", "be", "are", "has", "have",
    "this", "with", "but", "not", "he", "she", "they",
    "his", "her", "their", "an", "at", "by", "from",
    "good", "great", "bad", "really", "just", "like",
    "love", "hate", "make", "made", "get", "one", "even",
    "still", "well", "time", "way", "think", "people",
    "much", "many", "back", "watch",
})

DOMAIN_STOP = {
    "IMDB": {
        "movie", "film", "story", "character", "characters",
        "scene", "scenes", "plot", "show", "act", "acting",
        "actor", "actress", "director", "cinema", "screen",
        "role", "cast", "watched", "films", "movies", "series",
        "episode", "sequel", "ending", "version", "seen", "part",
    },
    "Yelp": {
        "restaurant", "food", "place", "order", "ordered", "came",
        "got", "went", "come", "going", "said", "told", "asked",
        "service", "staff", "table", "menu", "wait", "waitress",
        "waiter", "eat", "eating", "lunch", "dinner", "breakfast",
        "price", "prices", "experience", "location", "area",
    },
}


# Text cleaning

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>",      " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"&\w+;",          " ", text)
    text = re.sub(r"\.{2,}",         " ", text)
    text = re.sub(r"[^a-z\s]",       " ", text)
    text = re.sub(r"\s+",            " ", text).strip()
    return text

def tokenize(text: str, stopwords: set) -> str:
    tokens = [w for w in clean_text(text).split() if w not in stopwords and len(w) > 2]
    return " ".join(tokens)


# Load dataset

def load_dataset(cfg: dict, stopwords: set) -> tuple:
    """Return (texts_raw, texts_clean, labels) ready for BERTopic."""
    print(f"Loading {cfg['name']} ...")
    df = pd.read_csv(cfg["url"])

    if cfg["label_map"]:
        df[cfg["label_col"]] = df[cfg["label_col"]].map(cfg["label_map"])

    if SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=SEED).reset_index(drop=True)

    texts_raw   = df[cfg["text_col"]].astype(str).tolist()
    texts_clean = [tokenize(t, stopwords) for t in texts_raw]
    labels      = df[cfg["label_col"]].tolist()

    print(f"{cfg['name']} ready - {len(df):,} rows")
    return texts_raw, texts_clean, labels


# Evaluation metrics

def tc_npmi_for_topic(words: list, topic_docs: list) -> float:
    """Compute TC-NPMI for a single topic (identical to SEMTGPU's compute_topic_coherence)."""
    if len(words) < 2 or len(topic_docs) == 0:
        return 0.0

    n = len(topic_docs)
    df_w = {w: sum(1 for doc in topic_docs if w in doc.lower()) for w in words}

    npmi_scores = []
    for w1, w2 in combinations(words, 2):
        co = sum(1 for doc in topic_docs if w1 in doc.lower() and w2 in doc.lower())
        if co > 0 and df_w[w1] > 0 and df_w[w2] > 0:
            p12  = co / n
            p1   = df_w[w1] / n
            p2   = df_w[w2] / n
            pmi  = np.log((p12 + 1e-10) / (p1 * p2 + 1e-10))
            npmi = pmi / (-np.log(p12 + 1e-10))
            npmi_scores.append(npmi)

    return float(np.mean(npmi_scores)) if npmi_scores else 0.0


def compute_metrics(topic_model, texts_raw: list, topics: list, top_n: int = 10) -> dict:
    """Return n_topics, TC_NPMI, Diversity, Coverage."""
    topic_ids   = [t for t in topic_model.get_topics().keys() if t != -1]
    topic_words = [[w for w, _ in topic_model.get_topic(tid)[:top_n]] for tid in topic_ids]

    # Diversity
    all_words = [w for tw in topic_words for w in tw]
    diversity = len(set(all_words)) / (len(topic_ids) * top_n) if topic_ids else 0.0

    # Coverage
    topics_arr = np.array(topics)
    coverage   = float(np.sum(topics_arr != -1) / len(topics_arr))

    # TC-NPMI
    docs_per_topic = defaultdict(list)
    for doc, tid in zip(texts_raw, topics):
        if tid != -1:
            docs_per_topic[tid].append(doc)

    tc_npmi_vals = [
        tc_npmi_for_topic(words, docs_per_topic.get(tid, []))
        for tid, words in zip(topic_ids, topic_words)
    ]
    tc_npmi_mean = float(np.mean(tc_npmi_vals)) if tc_npmi_vals else 0.0

    return {
        "n_topics" : len(topic_ids),
        "TC_NPMI"  : tc_npmi_mean,
        "Diversity": diversity,
        "Coverage" : coverage,
    }


# Main loop: dataset -> n_topics

all_results = []

for ds_cfg in DATASETS:
    ds_name   = ds_cfg["name"]
    stopwords = BASE_STOP | DOMAIN_STOP.get(ds_name, set())

    print(f"\n[DATASET: {ds_name}]")
    texts_raw, texts_clean, labels = load_dataset(ds_cfg, stopwords)

    # Embed once per dataset
    print(f"Encoding embeddings with {BERT_MODEL} ...")
    embedding_model = SentenceTransformer(BERT_MODEL)
    embeddings = embedding_model.encode(texts_raw, show_progress_bar=True, batch_size=64)
    print("Embeddings done.\n")

    for n_topics in N_TOPICS_LIST:
        print(f"[{ds_name}] n_topics = {n_topics}")

        umap_model     = umap.UMAP(**UMAP_PARAMS)
        hdbscan_model  = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
        vectorizer     = CountVectorizer(**VECTORIZER_PARAMS)

        topic_model = BERTopic(
            embedding_model      = embedding_model,
            umap_model           = umap_model,
            hdbscan_model        = hdbscan_model,
            vectorizer_model     = vectorizer,
            calculate_probabilities = False,
            verbose              = False,
        )

        print("Fitting BERTopic ...")
        topics, _ = topic_model.fit_transform(texts_raw, embeddings=embeddings)

        raw_n = len([t for t in set(topics) if t != -1])
        print(f"Initial topics found: {raw_n}, reducing to {n_topics} ...")

        reduced_model  = topic_model.reduce_topics(texts_raw, nr_topics=n_topics)
        reduced_topics = reduced_model.topics_

        metrics = compute_metrics(reduced_model, texts_raw, reduced_topics)
        print(f"Done. Metrics: {metrics}\n")

        row = {"dataset": ds_name, "nr_topics": n_topics}
        row.update(metrics)
        all_results.append(row)


# Results table

print("\nRESULTS SUMMARY")

results_df = (
    pd.DataFrame(all_results)
    .set_index(["dataset", "nr_topics"])
    .sort_index()
)

print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))

output_csv = "bertopic_results.csv"
results_df.to_csv(output_csv)
print(f"\nSaved -> {output_csv}")