"""
SEMT Training Script - IMDB & Yelp
Loop over 2 datasets x 3 n_clusters configurations.
Results printed as a table and saved to results_summary.csv.
"""

import subprocess
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Clone SEMT repo (skip if already cloned)
subprocess.run(
    ["git", "clone", "https://github.com/zzeiidann/SEMT.git"],
    check=False,
)

from SEMT.SEMT_Gpu import SEMTGPU, CachedBERTDataset  # noqa: E402


# Config

BERT_MODEL      = "bert-base-uncased"
MAX_LENGTH      = 512
DIMS            = [768, 2022, 2022, 256]
N_CLUSTERS_LIST = [30, 50, 80]
PRETRAIN_EPOCHS = 30
PRETRAIN_BATCH  = 128

FIT_PARAMS = dict(
    tol                  = 1e-100,
    update_interval      = 10,
    learning_rate        = 1e-3,
    compute_metrics      = True,
    gamma                = 0.5,
    eta                  = 0.5,
    maxiter              = 100,
    batch_size           = 256,
    val_ratio            = 0.2,
    token_attr_bert_name = BERT_MODEL,
)

# Dataset registry - name, url, text column, label column, label mapping
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


# Load and preprocess dataset

def load_dataset(cfg: dict) -> tuple:
    """Return (texts, labels) Series ready for CachedBERTDataset."""
    print(f"Loading {cfg['name']} ...")
    df = pd.read_csv(cfg["url"])

    if cfg["label_map"]:
        df[cfg["label_col"]] = df[cfg["label_col"]].map(cfg["label_map"])

    df[cfg["text_col"]] = df[cfg["text_col"]].apply(clean_text)

    texts  = df[cfg["text_col"]].reset_index(drop=True)
    labels = df[cfg["label_col"]].reset_index(drop=True)
    print(f"{cfg['name']} ready - {len(df):,} rows")
    return texts, labels


# Main loop: dataset -> n_clusters

all_results = []

for ds_cfg in DATASETS:
    ds_name = ds_cfg["name"]
    print(f"\n[DATASET: {ds_name}]")

    texts, labels = load_dataset(ds_cfg)
    stop_words = BASE_STOP | DOMAIN_STOP.get(ds_name, set())

    print(f"Building CachedBERTDataset ({BERT_MODEL}) ...")
    dataset = CachedBERTDataset(
        texts        = texts,
        labels       = labels,
        bert_model   = BERT_MODEL,
        max_length   = MAX_LENGTH,
        cuda         = True,
        testing_mode = False,
        return_texts = True,
    )
    print("Embeddings cached.\n")

    for n_clusters in N_CLUSTERS_LIST:
        print(f"[{ds_name}] n_clusters = {n_clusters}")

        model = SEMTGPU(dims=DIMS, n_clusters=n_clusters)
        model.set_stop_words(stop_words)

        print(f"Pre-training autoencoder (epochs={PRETRAIN_EPOCHS}) ...")
        model.pretrain_autoencoder(dataset, epochs=PRETRAIN_EPOCHS, batch_size=PRETRAIN_BATCH)

        print("Fitting SEMT ...")
        y_pred, s_probs, metrics = model.fit(dataset=dataset, **FIT_PARAMS)

        row = {"dataset": ds_name, "n_clusters": n_clusters}
        if isinstance(metrics, dict):
            row.update(metrics)
        all_results.append(row)
        print(f"Done. Metrics: {metrics}\n")


# Results table

print("\nRESULTS SUMMARY")

results_df = (
    pd.DataFrame(all_results)
    .set_index(["dataset", "n_clusters"])
    .sort_index()
)

print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))

output_csv = "results_summary.csv"
results_df.to_csv(output_csv)
print(f"\nSaved -> {output_csv}")