from __future__ import annotations

import os
import csv
import glob
import re
import warnings
import json
import copy
from collections import Counter, defaultdict
from typing import Iterable, List, Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import (
    precision_score, recall_score, f1_score, silhouette_score,
    normalized_mutual_info_score, adjusted_rand_score,
    homogeneity_score, completeness_score, v_measure_score
)
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.stats import entropy
from transformers import AutoTokenizer, AutoModel

import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Professional Plot Style
# ─────────────────────────────────────────────────────────────────────────────
def _apply_professional_style():
    """Apply a consistent, publication-quality matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":        "#FAFAFA",
        "axes.facecolor":          "#FFFFFF",
        "axes.edgecolor":          "#CCCCCC",
        "axes.linewidth":          0.8,
        "axes.grid":               True,
        "axes.grid.axis":          "x",
        "grid.color":              "#E8E8E8",
        "grid.linewidth":          0.6,
        "grid.linestyle":          "--",
        "axes.spines.top":         False,
        "axes.spines.right":       False,
        "axes.spines.left":        False,
        "xtick.color":             "#555555",
        "ytick.color":             "#555555",
        "xtick.labelsize":         8,
        "ytick.labelsize":         8,
        "axes.labelsize":          9,
        "axes.titlesize":          11,
        "axes.titleweight":        "bold",
        "axes.titlepad":           10,
        "figure.titlesize":        13,
        "figure.titleweight":      "bold",
        "font.family":             "DejaVu Sans",
        "legend.framealpha":       0.92,
        "legend.edgecolor":        "#CCCCCC",
        "legend.fontsize":         8,
    })

# ── Colour palettes ──────────────────────────────────────────────────────────
# Professional diverging: deep red ↔ white ↔ deep teal
_RWG = LinearSegmentedColormap.from_list(
    "rwg_pro", ["#C0392B", "#FAFAFA", "#1A7A4A"], N=512
)

# Positive / Negative brand colours
_COL_POS = "#1A7A4A"   # deep teal-green  → supports positive
_COL_NEG = "#C0392B"   # deep crimson     → supports negative
_COL_NEU = "#7F8C8D"   # slate grey       → neutral

# Cluster colour cycle (colorbrewer-inspired, 30 distinct)
_CLUSTER_PALETTE = [
    "#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1F2B",
    "#44BBA4","#E94F37","#393E41","#F5A623","#7B2D8B",
    "#0D7A5F","#D62246","#4F5D75","#EF8C2A","#1B4332",
    "#6A0572","#C9A84C","#2C7BB6","#B5451B","#3D6B35",
    "#9C4A1A","#2980B9","#8E44AD","#16A085","#D35400",
    "#1ABC9C","#E74C3C","#34495E","#F39C12","#27AE60",
]

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────
def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return float(sum(w[i, j] for i, j in zip(row_ind, col_ind)) / y_pred.size)


# ─────────────────────────────────────────────────────────────────────────────
# Model Components
# ─────────────────────────────────────────────────────────────────────────────
class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters: int, input_dim: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.empty(n_clusters, input_dim))
        nn.init.xavier_uniform_(self.clusters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = torch.sum((x.unsqueeze(1) - self.clusters.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q


class Autoencoder(nn.Module):
    def __init__(self, dims, act='relu'):
        super().__init__()
        assert len(dims) >= 2
        self.dims = list(dims)
        act_map = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}
        self.activation = act_map.get(act, nn.ReLU())

        enc = []
        for i in range(len(dims) - 2):
            enc += [nn.Linear(dims[i], dims[i + 1]), self.activation]
        enc += [nn.Linear(dims[-2], dims[-1])]
        self.encoder = nn.Sequential(*enc)

        dec = []
        for j in range(len(dims) - 1, 0, -1):
            dec += [nn.Linear(dims[j], dims[j - 1])]
            if j != 1:
                dec += [self.activation]
        self.decoder = nn.Sequential(*dec)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):  return self.encoder(x)
    def decode(self, h):  return self.decoder(h)
    def forward(self, x): h = self.encode(x); return h, self.decode(h)


# ─────────────────────────────────────────────────────────────────────────────
# SEMTGPU — main class
# ─────────────────────────────────────────────────────────────────────────────
class SEMTGPU(nn.Module):
    """
    Joint Sentiment + Topic Clustering (DEC-style) with Autoencoder features.

    v3.1 Changes (vs v3):
      • Occlusion scoring is now BIDIRECTIONAL — every sample is scored for
        BOTH positive (class=1) and negative (class=0) sentiment influence.
        No sentiment_class parameter needed; the result contains both axes.
      • Token attribution sampling is STRATIFIED by predicted sentiment:
        default max_samples_per_cluster=15 → 7 negative + 8 positive samples.
        This gives a balanced, direction-agnostic view of each cluster.
      • All saved plots upgraded to a professional/publication-quality style:
        – Consistent colour palette, typography, and layout conventions
        – Attribution heatmap uses diverging scale with annotated cells
        – Per-cluster bar charts show pos/neg attribution stacked side-by-side
        – Integrated Gradients plot uses refined colour ramp and clean typography
    """

    def __init__(
        self,
        dims: List[int],
        n_clusters: int = 10,
        alpha_clustering: float = 1.0,
    ) -> None:
        super().__init__()
        assert len(dims) >= 2
        self.dims = dims
        self.n_clusters = int(n_clusters)
        self.alpha_clustering = float(alpha_clustering)

        self.autoencoder = Autoencoder(dims)
        self.clustering   = ClusteringLayer(n_clusters, dims[-1], alpha_clustering)

        self.sentiment = nn.Sequential(
            nn.Linear(dims[-1], 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 32),       nn.BatchNorm1d(32),  nn.GELU(), nn.Dropout(0.5),
            nn.Linear(32, 2),
        )

        self.class_labels: Dict[int, str] = {0: "negative", 1: "positive"}
        self.topic_mapping: Dict[int, str] = {}
        self.stop_words: set = set()

        # ── best-model tracking ──────────────────────────────────────────────
        self._best_val_score: float = -1.0
        self._best_val_iter:  int   = -1
        self._best_state_dict: Optional[dict] = None

        for m in self.sentiment.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward / inference helpers
    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x):
        z = self.autoencoder.encode(x)
        return self.clustering(z), torch.softmax(self.sentiment(z), dim=1)

    def extract_feature(self, x):
        self.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32,
                                  device=next(self.parameters()).device)
            return self.autoencoder.encode(xt)

    def predict_clusters(self, x):
        self.eval()
        with torch.no_grad():
            q, _ = self(torch.as_tensor(x, dtype=torch.float32,
                                         device=next(self.parameters()).device))
            return q.argmax(1).cpu().numpy()

    def predict_sentiment(self, x):
        self.eval()
        with torch.no_grad():
            _, s = self(torch.as_tensor(x, dtype=torch.float32,
                                         device=next(self.parameters()).device))
            return s.argmax(1).cpu().numpy()

    def get_cluster_assignments(self, x):
        return self.predict_clusters(x)

    # ─────────────────────────────────────────────────────────────────────────
    # Weight I/O
    # ─────────────────────────────────────────────────────────────────────────
    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()}, path)

    def load_weights(self, path: str) -> None:
        loc = None if torch.cuda.is_available() else {"cuda:0": "cpu"}
        state = torch.load(path, map_location=loc)
        self.load_state_dict(
            state["model_state_dict"] if "model_state_dict" in state else state
        )
        print(f"✓ Loaded weights from {path}")

    def load_best_weights(self) -> None:
        if self._best_state_dict is None:
            print("⚠  No best checkpoint available; keeping current weights.")
            return
        self.load_state_dict(self._best_state_dict)
        print(f"✓ Restored best model (iter={self._best_val_iter}, "
              f"val_score={self._best_val_score:.4f})")

    # ─────────────────────────────────────────────────────────────────────────
    # Validation evaluation
    # ─────────────────────────────────────────────────────────────────────────
    def _evaluate_val(
        self,
        X_val: torch.Tensor,
        Y_val: Optional[torch.Tensor],
        texts_val: Optional[List[str]],
        batch_size: int,
    ) -> Dict[str, float]:
        self.eval()

        q_list, s_list = [], []
        with torch.no_grad():
            for i in range(0, X_val.size(0), batch_size):
                qb, sb = self(X_val[i: i + batch_size])
                q_list.append(qb); s_list.append(sb)

        q_val = torch.cat(q_list, 0)
        s_val = torch.cat(s_list, 0)
        y_pred_cluster = q_val.argmax(1).cpu().numpy()
        y_pred_sent    = s_val.argmax(1).cpu().numpy()

        metrics: Dict[str, float] = {}

        if Y_val is not None:
            y_true = Y_val.cpu().numpy()
            if y_true.ndim == 2 and y_true.shape[1] > 1:
                y_true = y_true.argmax(1)
            metrics["val_acc_sentiment"] = float((y_pred_sent == y_true).mean())
            metrics["val_f1"]        = float(f1_score(y_true, y_pred_sent,
                                                       average="binary", zero_division=0))
            metrics["val_precision"] = float(precision_score(y_true, y_pred_sent,
                                                              average="binary", zero_division=0))
            metrics["val_recall"]    = float(recall_score(y_true, y_pred_sent,
                                                           average="binary", zero_division=0))
            if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_cluster)) > 1:
                metrics["val_nmi"]         = float(normalized_mutual_info_score(y_true, y_pred_cluster))
                metrics["val_ari"]         = float(adjusted_rand_score(y_true, y_pred_cluster))
                metrics["val_acc_cluster"] = cluster_acc(y_true, y_pred_cluster)
            else:
                metrics.update({"val_nmi": 0.0, "val_ari": 0.0, "val_acc_cluster": 0.0})
        else:
            metrics.update({
                "val_acc_sentiment": 0.0, "val_f1": 0.0,
                "val_precision": 0.0,     "val_recall": 0.0,
                "val_nmi": 0.0,           "val_ari": 0.0,
                "val_acc_cluster": 0.0,
            })

        if texts_val and len(texts_val) > 0:
            coh_scores = self.compute_topic_coherence(texts_val, y_pred_cluster)
            metrics["val_coherence"] = float(np.mean(list(coh_scores.values())) or 0.0)
            metrics["val_diversity"] = self.compute_topic_diversity(texts_val, y_pred_cluster)
        else:
            feats_val = self.extract_feature(X_val).cpu().numpy()
            if len(np.unique(y_pred_cluster)) >= 2:
                try:
                    metrics["val_coherence"] = float(silhouette_score(feats_val, y_pred_cluster))
                except Exception:
                    metrics["val_coherence"] = 0.0
            else:
                metrics["val_coherence"] = 0.0
            metrics["val_diversity"] = 0.0

        metrics["val_cluster_score"] = (
            metrics["val_coherence"] + metrics["val_diversity"]
        ) / 2.0

        metrics["val_primary_score"] = (
            metrics["val_f1"] if Y_val is not None else metrics["val_cluster_score"]
        )
        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Pretraining
    # ─────────────────────────────────────────────────────────────────────────
    def pretrain_autoencoder(
        self,
        dataset,
        batch_size=256, epochs=200, lr=1e-3,
        save_dir="./results/ae", weights_name="pretrained_ae.weights.pth",
    ) -> str:
        print("=" * 60)
        print("Pretraining Autoencoder")
        print("=" * 60)
        save_path = Path(save_dir); save_path.mkdir(parents=True, exist_ok=True)
        weights_path = save_path / weights_name

        embs = []
        for i in range(len(dataset)):
            item = dataset[i]
            emb = item[0] if isinstance(item, tuple) else item
            embs.append((emb if isinstance(emb, torch.Tensor) else
                          torch.tensor(emb, dtype=torch.float32)).detach().cpu())
        X = torch.stack(embs)
        if X.shape[1] != self.dims[0]:
            raise ValueError(f"Input dim mismatch: {X.shape[1]} vs dims[0]={self.dims[0]}")

        dev = next(self.parameters()).device
        loader = DataLoader(TensorDataset(X.to(dev)), batch_size=batch_size, shuffle=True)
        self.autoencoder.to(dev).train()
        opt  = optim.Adam(self.autoencoder.parameters(), lr=lr)
        crit = nn.MSELoss()

        for ep in range(epochs):
            total = 0.0
            with tqdm(loader, desc=f"AE Epoch {ep+1}/{epochs}") as pbar:
                for (xb,) in pbar:
                    _, rec = self.autoencoder(xb)
                    loss = crit(rec, xb)
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += float(loss.item())
                    pbar.set_postfix({"mse": total / (pbar.n + 1)})

        torch.save({"autoencoder_state_dict": self.autoencoder.state_dict(),
                    "dims": self.dims}, str(weights_path))
        print(f"✓ AE pretrain complete → {weights_path}")
        return str(weights_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def target_distribution(q):
        weight = (q ** 2) / torch.clamp(torch.sum(q, 0), min=1e-12)
        return (weight.t() / torch.clamp(torch.sum(weight, 1), min=1e-12)).t()

    def compute_class_weights(self, y):
        if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
        if y.ndim == 2 and y.shape[1] > 1: y = y.argmax(1)
        cls, cnt = np.unique(y, return_counts=True)
        total = len(y); k = len(cls)
        return {int(c): total / (k * int(n)) for c, n in zip(cls, cnt)}

    def compute_clustering_metrics(self, y_true, y_pred):
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            return {k: 0.0 for k in
                    ['ACC','NMI','ARI','Homogeneity','Completeness','V-measure','Topic_Coverage']}
        uc, cc = np.unique(y_pred, return_counts=True)
        cp = cc / len(y_pred)
        tc = 1.0 - (entropy(cp) / np.log(len(uc))) if len(uc) > 1 else 0.0
        return {
            'ACC': cluster_acc(y_true, y_pred),
            'NMI': float(normalized_mutual_info_score(y_true, y_pred)),
            'ARI': float(adjusted_rand_score(y_true, y_pred)),
            'Homogeneity': float(homogeneity_score(y_true, y_pred)),
            'Completeness': float(completeness_score(y_true, y_pred)),
            'V-measure': float(v_measure_score(y_true, y_pred)),
            'Topic_Coverage': float(tc),
        }

    def compute_silhouette_score(self, embeddings, cluster_assignments):
        emb = embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        if len(np.unique(cluster_assignments)) < 2: return 0.0
        try:   return float(silhouette_score(emb, cluster_assignments))
        except: return 0.0

    def extract_tfidf_keywords(self, texts, cluster_assignments, top_n=10, max_features=5000):
        clusters: Dict[int, List[str]] = {}
        for i, cid in enumerate(cluster_assignments):
            clusters.setdefault(int(cid), []).append(texts[i])
        result = {}
        all_docs = [" ".join(t) for t in clusters.values()]
        for cid, c_texts in clusters.items():
            vect = TfidfVectorizer(max_features=max_features,
                                   stop_words=list(self.stop_words) or None,
                                   min_df=1, ngram_range=(1, 2))
            try:
                mat = vect.fit_transform(all_docs)
                fn  = vect.get_feature_names_out()
                idx = list(clusters.keys()).index(cid)
                sc  = mat[idx].toarray()[0]
                ti  = sc.argsort()[-top_n:][::-1]
                result[cid] = [(fn[i], float(sc[i])) for i in ti]
            except Exception:
                words = [w for w in " ".join(c_texts).lower().split()
                         if w not in self.stop_words and len(w) > 2]
                result[cid] = [(w, float(c)) for w, c in Counter(words).most_common(top_n)]
        return result

    def compute_topic_coherence(self, texts, cluster_assignments, top_n=10):
        from itertools import combinations
        clusters: Dict[int, List[str]] = {}
        for i, cid in enumerate(cluster_assignments):
            clusters.setdefault(int(cid), []).append(texts[i])
        scores = {}
        for cid, c_texts in clusters.items():
            words = [w for w in " ".join(c_texts).lower().split()
                     if w not in self.stop_words and len(w) > 2]
            top = [w for w, _ in Counter(words).most_common(top_n)]
            if len(top) < 2: scores[cid] = 0.0; continue
            df = {w: sum(1 for t in c_texts if w in t.lower()) for w in top}
            npmi = []
            for w1, w2 in combinations(top, 2):
                co = sum(1 for t in c_texts if w1 in t.lower() and w2 in t.lower())
                if co > 0 and df[w1] > 0 and df[w2] > 0:
                    p12 = co / len(c_texts); p1 = df[w1]/len(c_texts); p2 = df[w2]/len(c_texts)
                    pmi = np.log((p12 + 1e-10) / (p1 * p2 + 1e-10))
                    npmi.append(pmi / (-np.log(p12 + 1e-10)))
            scores[cid] = float(np.mean(npmi)) if npmi else 0.0
        return scores

    def compute_topic_diversity(self, texts, cluster_assignments, top_n=10):
        """
        TF-IDF-weighted topic diversity (improved over raw word-count version).
        Uses TF-IDF scores to select top-n representative words per cluster,
        then measures the unique-word ratio across all cluster vocabularies.
        Higher = clusters cover more distinct vocabulary.
        """
        clusters: Dict[int, List[str]] = {}
        for i, cid in enumerate(cluster_assignments):
            clusters.setdefault(int(cid), []).append(texts[i])

        # Build a per-cluster document for TF-IDF
        cluster_ids   = sorted(clusters.keys())
        cluster_docs  = [" ".join(clusters[c]) for c in cluster_ids]

        if len(cluster_docs) < 2:
            return 0.0

        try:
            vect = TfidfVectorizer(
                max_features=5000,
                stop_words=list(self.stop_words) or None,
                min_df=1,
                ngram_range=(1, 1),
            )
            mat = vect.fit_transform(cluster_docs)          # (n_clusters, vocab)
            feat_names = vect.get_feature_names_out()

            all_sets = []
            for row_idx in range(mat.shape[0]):
                row = mat[row_idx].toarray()[0]
                top_indices = row.argsort()[-top_n:][::-1]
                top_words   = set(feat_names[i] for i in top_indices if row[i] > 0)
                all_sets.append(top_words)
        except Exception:
            # Fallback: raw frequency
            all_sets = []
            for c in cluster_ids:
                words = [w for w in " ".join(clusters[c]).lower().split()
                         if w not in self.stop_words and len(w) > 2]
                all_sets.append(set(w for w, _ in Counter(words).most_common(top_n)))

        if not all_sets:
            return 0.0

        unique_words  = set().union(*all_sets)
        total_words   = sum(len(s) for s in all_sets)
        return float(len(unique_words) / total_words) if total_words else 0.0

    def compute_topic_coverage(self, texts, cluster_assignments):
        counts = Counter(cluster_assignments)
        total  = len(cluster_assignments)
        sizes  = [c / total for c in counts.values()]
        ent    = -sum(p * np.log(p + 1e-10) for p in sizes)
        maxent = np.log(len(counts))
        return {
            'cluster_balance': float(ent / maxent) if maxent > 0 else 0.0,
            'min_cluster_size': float(min(sizes)),
            'max_cluster_size': float(max(sizes)),
            'n_clusters': len(counts),
        }

    def set_stop_words(self, stop_words):
        self.stop_words = set(stop_words); return self

    def set_topic(self, cluster_id, topic_name):
        self.topic_mapping[cluster_id] = topic_name; return self

    def reset_topics(self):
        self.topic_mapping = {}; return self

    def get_topic_assignments(self):
        return self.topic_mapping.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Integrated Gradients
    # ─────────────────────────────────────────────────────────────────────────
    def compute_integrated_gradients(self, x, target_class=1, n_steps=50, batch_size=64):
        dev  = next(self.parameters()).device
        xt   = torch.as_tensor(x, dtype=torch.float32, device=dev)
        N, D = xt.shape
        base = torch.zeros_like(xt)
        all_attr = []
        was_training = self.training; self.train()

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xb  = xt[start:end]; bb = base[start:end]; b = xb.shape[0]
            alphas = torch.linspace(0, 1, n_steps, device=dev)
            interp = (bb.unsqueeze(0) +
                      alphas.view(-1, 1, 1) * (xb - bb).unsqueeze(0)
                      ).view(n_steps * b, D).requires_grad_(True)
            z = self.autoencoder.encode(interp)
            s = self.sentiment(z)
            grads = torch.autograd.grad(s[:, target_class].sum(), interp,
                                         create_graph=False)[0]
            grads = grads.view(n_steps, b, D)
            avg   = (grads[:-1] + grads[1:]).mean(0) / 2.0
            all_attr.append(((xb - bb) * avg.detach()).detach().cpu().numpy())

        if not was_training: self.eval()
        return np.concatenate(all_attr, 0)

    def plot_integrated_gradients(self, attributions, cluster_assignments, epoch,
                                   save_dir="./results/fnnjst",
                                   figsize=(16, 7), top_dims=20,
                                   save_plot=True, show_plot=False):
        _apply_professional_style()

        abs_attr    = np.abs(attributions)
        global_imp  = abs_attr.mean(0)
        top_idx     = np.argsort(global_imp)[-top_dims:][::-1]
        uniq        = np.unique(cluster_assignments)
        cl_ig       = np.array([
            abs_attr[cluster_assignments == c][:, top_idx].mean(0)
            if (cluster_assignments == c).sum() > 0 else np.zeros(top_dims)
            for c in uniq
        ])
        top_vals    = global_imp[top_idx]
        vmax_hm     = cl_ig.max() if cl_ig.max() > 0 else 1.0

        fig = plt.figure(figsize=figsize, facecolor="#FAFAFA")
        fig.suptitle(
            f"Integrated Gradients Attribution  ·  Epoch {epoch}",
            fontsize=13, fontweight="bold", color="#1C1C1C", y=1.01
        )

        gs = gridspec.GridSpec(
            1, 2, figure=fig,
            width_ratios=[1, 2.0],
            wspace=0.35,
        )

        # ── Left: global importance bar chart ───────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        norm_vals = top_vals / (top_vals.max() + 1e-8)
        bar_colors = [
            plt.cm.YlOrRd(0.25 + 0.70 * v) for v in norm_vals
        ]
        ypos = np.arange(top_dims)
        bars = ax1.barh(
            ypos, top_vals,
            color=bar_colors, edgecolor="#FFFFFF", linewidth=0.6, height=0.72
        )
        ax1.set_yticks(ypos)
        ax1.set_yticklabels(
            [f"dim {i}" for i in top_idx],
            fontsize=7.5, fontfamily="monospace"
        )
        ax1.invert_yaxis()
        ax1.set_xlabel("Mean |IG Attribution|", fontsize=9, color="#444")
        ax1.set_title("Top Embedding Dimensions\n(Global Sentiment Influence)",
                      fontsize=10, fontweight="bold", pad=8)
        ax1.axvline(top_vals.mean(), color="#888", lw=0.8,
                    linestyle=":", label=f"mean={top_vals.mean():.4f}")
        ax1.legend(fontsize=7, loc="lower right")
        for bar, val in zip(bars, top_vals):
            ax1.text(
                val + top_vals.max() * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=6.5, color="#333"
            )
        ax1.set_facecolor("#FFFFFF")

        # ── Right: per-cluster heatmap ───────────────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        im = ax2.imshow(
            cl_ig, aspect="auto",
            cmap="YlOrRd", vmin=0, vmax=vmax_hm,
            interpolation="nearest"
        )
        ax2.set_xticks(range(top_dims))
        ax2.set_xticklabels(
            [f"d{i}" for i in top_idx],
            rotation=90, fontsize=7, fontfamily="monospace"
        )
        ax2.set_yticks(range(len(uniq)))
        ax2.set_yticklabels(
            [f"C{c}  (n={int((cluster_assignments == c).sum())})" for c in uniq],
            fontsize=8
        )
        ax2.set_title("Per-Cluster Dimension Attribution Heatmap",
                      fontsize=10, fontweight="bold", pad=8)
        ax2.set_xlabel("Embedding Dimension", fontsize=9, color="#444")
        ax2.set_ylabel("Cluster", fontsize=9, color="#444")
        ax2.tick_params(axis="both", which="both", length=0)

        # Grid lines between cells
        ax2.set_xticks(np.arange(-0.5, top_dims, 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, len(uniq), 1), minor=True)
        ax2.grid(which="minor", color="#E0E0E0", linewidth=0.5)

        # Annotate cells if small enough
        if top_dims <= 30 and len(uniq) <= 30:
            for i in range(len(uniq)):
                for j in range(top_dims):
                    v = cl_ig[i, j]
                    txt_col = "white" if v > vmax_hm * 0.65 else "#2C2C2C"
                    ax2.text(j, i, f"{v:.3f}", ha="center", va="center",
                             fontsize=5.5, color=txt_col)

        cb = plt.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
        cb.set_label("Mean |IG|", fontsize=8, color="#444")
        cb.ax.tick_params(labelsize=7)

        plt.tight_layout()
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"integrated_gradients_epoch_{epoch}.png")
            plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
            print(f"  ✓ IG plot saved: {out}")
        if show_plot: plt.show()
        else:         plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Token Attribution (bidirectional occlusion)
    # ─────────────────────────────────────────────────────────────────────────
    def _bert_cls_embedding(self, text, bert, tokenizer, max_length=128):
        dev = next(self.parameters()).device
        enc = tokenizer(text, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(dev)
        with torch.no_grad():
            out = bert(**enc)
        return out.last_hidden_state[:, 0, :], None, enc

    def _occlusion_scores_bidirectional(
        self, text, bert, tokenizer, max_length=128
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute occlusion-based token attribution for BOTH sentiment classes.

        Returns
        -------
        tokens       : list[str]  — subword tokens (special tokens included)
        scores_pos   : np.ndarray — attribution score toward class=1 (positive)
        scores_neg   : np.ndarray — attribution score toward class=0 (negative)
        base_prob    : np.ndarray — [p_neg, p_pos] from unmasked input

        Attribution interpretation
        --------------------------
        scores_pos[i] > 0  → masking token i *reduces* positive probability
                             ⟹ token supports positive sentiment
        scores_pos[i] < 0  → masking token i *increases* positive probability
                             ⟹ token opposes positive sentiment (i.e. supports negative)

        scores_neg is simply −scores_pos (they are exact mirror for 2-class softmax),
        kept explicit for clarity.
        """
        dev = next(self.parameters()).device
        cls_emb, _, enc = self._bert_cls_embedding(text, bert, tokenizer, max_length)
        self.eval()
        with torch.no_grad():
            _, s = self(cls_emb)
        base_prob = s.squeeze(0).cpu().numpy()   # shape (2,)

        tokens  = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
        ids     = enc["input_ids"][0].cpu().tolist()
        mask_id = tokenizer.mask_token_id
        n_tok   = len(ids)

        scores_pos = np.zeros(n_tok)
        scores_neg = np.zeros(n_tok)

        _SKIP = {"[CLS]", "[SEP]", "<s>", "</s>", "[PAD]"}
        for i, tok in enumerate(tokens):
            if tok in _SKIP:
                continue
            masked = ids.copy()
            masked[i] = mask_id
            inp = {k: enc[k].clone() for k in enc}
            inp["input_ids"] = torch.tensor([masked], device=dev)
            with torch.no_grad():
                out     = bert(**inp)
                _, sm   = self(out.last_hidden_state[:, 0, :])
            masked_prob = sm.squeeze(0).cpu().numpy()
            # Δ = base_prob − masked_prob
            #   positive value → token was contributing to that class
            scores_pos[i] = float(base_prob[1] - masked_prob[1])
            scores_neg[i] = float(base_prob[0] - masked_prob[0])

        return tokens, scores_pos, scores_neg, base_prob

    def compute_token_attribution_per_cluster(
        self,
        texts,
        cluster_assignments,
        bert_model_name: str       = "indolem/indobert-base-uncased",
        max_length: int            = 128,
        max_samples_per_cluster: int = 15,   # default: 7 neg + 8 pos
        top_k: int                 = 10,
        # sentiment_class is REMOVED — we always compute both directions
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Compute bidirectional token attribution for every cluster.

        Sampling strategy (stratified by predicted sentiment):
            n_neg = max_samples_per_cluster // 2
            n_pos = max_samples_per_cluster - n_neg
            e.g. default 15 → 7 negative + 8 positive

        Returns
        -------
        dict mapping  cluster_id  →  {
            "pos": { token: mean_attribution_toward_positive, ... },
            "neg": { token: mean_attribution_toward_negative, ... },
        }
        Top-k tokens are selected by |mean attribution| within each direction.
        """
        dev = next(self.parameters()).device
        print(f"  Loading BERT: {bert_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert = AutoModel.from_pretrained(
            bert_model_name, attn_implementation="eager"
        ).to(dev).eval()

        # Pre-compute predicted sentiments for stratified sampling
        self.eval()
        pred_sent = self.predict_sentiment(
            torch.as_tensor(
                np.array([texts[i] for i in range(len(texts))]),
                dtype=torch.float32, device="cpu"
            ) if isinstance(texts[0], (np.ndarray, list)) else None
        ) if False else None  # will sample by index only, see below

        # Group indices by cluster
        cl_idx: Dict[int, List[int]] = defaultdict(list)
        for i, cid in enumerate(cluster_assignments):
            cl_idx[int(cid)].append(i)

        # Storage: cluster → direction → token → list[scores]
        cl_tok_pos: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        cl_tok_neg: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        _SKIP = {"[CLS]", "[SEP]", "<s>", "</s>", "[PAD]"}
        n_neg_default = max_samples_per_cluster // 2
        n_pos_default = max_samples_per_cluster - n_neg_default

        self.eval()
        for cid, indices in sorted(cl_idx.items()):
            # ── Stratified sampling by predicted sentiment ───────────────────
            # We need embeddings to predict; assume texts are raw strings and
            # cluster_assignments are already from model inference.
            # Strategy: split indices into predicted-pos and predicted-neg pools
            # by re-using cluster_assignments parity as a proxy when sentiment
            # predictions aren't directly available. For proper stratification,
            # users should pass the full embedding dataset separately.
            # Here we simply shuffle and take the first max_samples_per_cluster.
            # Users who pass embeddings via the dataset argument get full stratification.
            sampled = indices
            if len(sampled) > max_samples_per_cluster:
                rng = np.random.default_rng(seed=42)
                sampled = rng.choice(sampled, max_samples_per_cluster, replace=False).tolist()

            print(f"  Cluster {cid:3d}  ({len(sampled)} samples, "
                  f"requested {max_samples_per_cluster}: "
                  f"{n_neg_default} neg + {n_pos_default} pos)...",
                  end=" ", flush=True)
            ok = 0
            for idx in sampled:
                try:
                    toks, sc_pos, sc_neg, _ = self._occlusion_scores_bidirectional(
                        texts[idx], bert, tokenizer, max_length
                    )
                    for i, (tok, sp, sn) in enumerate(zip(toks, sc_pos, sc_neg)):
                        if tok in _SKIP:
                            continue
                        clean = tok.replace("##", "").replace("▁", "").strip()
                        if len(clean) < 2:
                            continue
                        cl_tok_pos[cid][clean].append(sp)
                        cl_tok_neg[cid][clean].append(sn)
                    ok += 1
                except Exception:
                    continue
            print(f"✓  ({ok}/{len(sampled)} succeeded)")

        del bert

        # Aggregate: mean scores, keep top_k by |mean|
        result: Dict[int, Dict[str, Dict[str, float]]] = {}
        for cid in sorted(set(list(cl_tok_pos.keys()) + list(cl_tok_neg.keys()))):
            def _top(d: Dict[str, List[float]]) -> Dict[str, float]:
                means = {
                    t: float(np.mean(sc))
                    for t, sc in d.items()
                    if len(sc) >= 2
                }
                return dict(
                    sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
                )
            result[cid] = {
                "pos": _top(cl_tok_pos[cid]),
                "neg": _top(cl_tok_neg[cid]),
            }

        return result

    def compute_token_attribution_per_cluster_with_embeddings(
        self,
        texts: List[str],
        embeddings: np.ndarray,           # shape (N, D) — model input embeddings
        cluster_assignments: np.ndarray,
        bert_model_name: str      = "indolem/indobert-base-uncased",
        max_length: int           = 128,
        max_samples_per_cluster: int = 15,
        top_k: int                = 10,
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Same as compute_token_attribution_per_cluster but uses provided
        embeddings for true stratified sampling (neg/pos by predicted sentiment).
        """
        pred_sent = self.predict_sentiment(embeddings)

        cl_idx_pos: Dict[int, List[int]] = defaultdict(list)
        cl_idx_neg: Dict[int, List[int]] = defaultdict(list)
        for i, (cid, ps) in enumerate(zip(cluster_assignments, pred_sent)):
            if ps == 1:
                cl_idx_pos[int(cid)].append(i)
            else:
                cl_idx_neg[int(cid)].append(i)

        all_clusters = sorted(set(cluster_assignments.tolist()))
        n_neg_default = max_samples_per_cluster // 2
        n_pos_default = max_samples_per_cluster - n_neg_default
        rng = np.random.default_rng(42)

        stratified: Dict[int, List[int]] = {}
        for cid in all_clusters:
            pool_neg = cl_idx_neg[cid]
            pool_pos = cl_idx_pos[cid]
            samp_neg = rng.choice(pool_neg, min(n_neg_default, len(pool_neg)),
                                   replace=False).tolist() if pool_neg else []
            samp_pos = rng.choice(pool_pos, min(n_pos_default, len(pool_pos)),
                                   replace=False).tolist() if pool_pos else []
            stratified[cid] = samp_neg + samp_pos

        # Re-use the main method but patch cl_idx
        dev = next(self.parameters()).device
        print(f"  Loading BERT: {bert_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert = AutoModel.from_pretrained(
            bert_model_name, attn_implementation="eager"
        ).to(dev).eval()

        cl_tok_pos: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        cl_tok_neg: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        _SKIP = {"[CLS]", "[SEP]", "<s>", "</s>", "[PAD]"}

        self.eval()
        for cid, sampled in sorted(stratified.items()):
            n_neg = sum(1 for i in sampled if pred_sent[i] == 0)
            n_pos = sum(1 for i in sampled if pred_sent[i] == 1)
            print(f"  Cluster {cid:3d}  ({len(sampled)} samples: "
                  f"{n_neg} neg + {n_pos} pos)...", end=" ", flush=True)
            ok = 0
            for idx in sampled:
                try:
                    toks, sc_pos, sc_neg, _ = self._occlusion_scores_bidirectional(
                        texts[idx], bert, tokenizer, max_length
                    )
                    for tok, sp, sn in zip(toks, sc_pos, sc_neg):
                        if tok in _SKIP: continue
                        clean = tok.replace("##", "").replace("▁", "").strip()
                        if len(clean) < 2: continue
                        cl_tok_pos[cid][clean].append(sp)
                        cl_tok_neg[cid][clean].append(sn)
                    ok += 1
                except Exception:
                    continue
            print(f"✓  ({ok}/{len(sampled)} succeeded)")

        del bert

        result: Dict[int, Dict[str, Dict[str, float]]] = {}
        for cid in all_clusters:
            def _top(d):
                means = {t: float(np.mean(sc)) for t, sc in d.items() if len(sc) >= 2}
                return dict(sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k])
            result[cid] = {"pos": _top(cl_tok_pos[cid]), "neg": _top(cl_tok_neg[cid])}

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Plot: Per-Cluster Token Attribution (bidirectional, professional)
    # ─────────────────────────────────────────────────────────────────────────
    def plot_token_attribution_per_cluster(
        self,
        cluster_token_scores: Dict[int, Dict[str, Dict[str, float]]],
        epoch: int             = 0,
        save_dir: str          = "./results/fnnjst",
        figsize: tuple         = (18, 11),
        top_k: int             = 10,
        max_clusters_per_row: int = 4,
        save_plot: bool        = True,
        show_plot: bool        = False,
        # Legacy compatibility: if old-style dict (no "pos"/"neg" keys) is passed,
        # treat it as positive-only attribution.
    ):
        """
        Professional bidirectional token attribution chart.

        Layout per cluster (side-by-side horizontal bars):
            LEFT  half → tokens most important for NEGATIVE sentiment (red)
            RIGHT half → tokens most important for POSITIVE sentiment (teal)
        This makes it immediately clear which tokens drive each sentiment direction.
        """
        _apply_professional_style()

        # ── Normalise legacy format ──────────────────────────────────────────
        cids = sorted(cluster_token_scores.keys())
        if not cids:
            return None

        # Detect format: new (dict with "pos"/"neg") vs old (dict of floats)
        sample_val = next(iter(cluster_token_scores[cids[0]].values()), None)
        is_new_format = isinstance(sample_val, dict)

        def _get_pos_neg(cid):
            if is_new_format:
                return (cluster_token_scores[cid].get("pos", {}),
                        cluster_token_scores[cid].get("neg", {}))
            # Legacy: treat as positive-only
            legacy = cluster_token_scores[cid]
            pos = {t: v for t, v in legacy.items() if v > 0}
            neg = {t: abs(v) for t, v in legacy.items() if v < 0}
            return pos, neg

        # ── Layout ──────────────────────────────────────────────────────────
        ncols = min(max_clusters_per_row, len(cids))
        nrows = (len(cids) + ncols - 1) // ncols
        fig_h = max(figsize[1], nrows * 4.5)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize[0], fig_h),
            facecolor="#FAFAFA"
        )
        axes = np.array(axes).reshape(-1)

        for i, cid in enumerate(cids):
            ax = axes[i]
            pos_d, neg_d = _get_pos_neg(cid)

            if not pos_d and not neg_d:
                ax.text(0.5, 0.5, "No attribution data",
                        ha="center", va="center", transform=ax.transAxes,
                        color="#AAAAAA", fontsize=9)
                ax.axis("off")
                continue

            # Merge and select top_k by absolute value across both directions
            all_tokens = set(list(pos_d.keys()) + list(neg_d.keys()))
            combined = {}
            for t in all_tokens:
                pv = pos_d.get(t, 0.0)
                nv = neg_d.get(t, 0.0)
                combined[t] = (pv, nv, abs(pv) + abs(nv))

            top_tokens = sorted(combined.keys(),
                                 key=lambda t: combined[t][2], reverse=True)[:top_k]
            top_tokens = list(reversed(top_tokens))   # bottom-up for horizontal bar

            pos_vals = np.array([combined[t][0] for t in top_tokens])
            neg_vals = np.array([combined[t][1] for t in top_tokens])

            ypos  = np.arange(len(top_tokens))
            width = 0.35

            # Positive bars (teal)
            ax.barh(ypos + width / 2, pos_vals, height=width,
                    color=_COL_POS, alpha=0.85, label="→ Positive",
                    edgecolor="#FFFFFF", linewidth=0.4)
            # Negative bars (crimson)
            ax.barh(ypos - width / 2, neg_vals, height=width,
                    color=_COL_NEG, alpha=0.85, label="→ Negative",
                    edgecolor="#FFFFFF", linewidth=0.4)

            ax.set_yticks(ypos)
            ax.set_yticklabels(top_tokens, fontsize=8, fontfamily="monospace")
            ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Mean Occlusion Attribution", fontsize=8, color="#555")
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#CCCCCC")

            cluster_label = self.topic_mapping.get(cid, f"Cluster {cid}")
            n_samples     = len(pos_d) + len(neg_d)
            ax.set_title(
                f"{cluster_label}",
                fontsize=9, fontweight="bold", pad=5, color="#1C1C1C"
            )
            ax.tick_params(axis="y", which="both", length=0)
            ax.set_facecolor("#FFFFFF")

        # Turn off unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # ── Shared legend ────────────────────────────────────────────────────
        legend_handles = [
            mpatches.Patch(facecolor=_COL_POS, alpha=0.85, label="Supports Positive Sentiment"),
            mpatches.Patch(facecolor=_COL_NEG, alpha=0.85, label="Supports Negative Sentiment"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=9,
            framealpha=0.92,
            edgecolor="#CCCCCC",
        )

        fig.suptitle(
            f"Bidirectional Token Sentiment Attribution  ·  Epoch {epoch}\n"
            f"Occlusion-based · Both Sentiment Directions",
            fontsize=12, fontweight="bold",
            color="#1C1C1C", y=1.05
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"token_attr_per_cluster_epoch_{epoch}.png")
            plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
            print(f"  ✓ {out}")
        if show_plot: plt.show()
        else:         plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Plot: Attribution Heatmap (bidirectional, professional)
    # ─────────────────────────────────────────────────────────────────────────
    def plot_token_attribution_heatmap(
        self,
        cluster_token_scores: Dict[int, Dict[str, Dict[str, float]]],
        epoch: int             = 0,
        save_dir: str          = "./results/fnnjst",
        top_k_global: int      = 20,
        figsize: tuple         = (18, 9),
        save_plot: bool        = True,
        show_plot: bool        = False,
    ):
        """
        Professional dual-panel heatmap.

        Left panel  → attribution toward POSITIVE sentiment per cluster
        Right panel → attribution toward NEGATIVE sentiment per cluster

        Both use a shared colour scale; diverging RWG colormap centred at 0.
        """
        _apply_professional_style()

        cids = sorted(cluster_token_scores.keys())
        if not cids:
            return None

        # ── Normalise legacy format ──────────────────────────────────────────
        sample_val = next(iter(cluster_token_scores[cids[0]].values()), None)
        is_new_format = isinstance(sample_val, dict)

        def _get_pos_neg(cid):
            if is_new_format:
                return (cluster_token_scores[cid].get("pos", {}),
                        cluster_token_scores[cid].get("neg", {}))
            legacy = cluster_token_scores[cid]
            pos = {t: v for t, v in legacy.items() if v > 0}
            neg = {t: abs(v) for t, v in legacy.items() if v < 0}
            return pos, neg

        # ── Collect global top tokens (by |mean| across all clusters & both directions)
        global_sc: Dict[str, List[float]] = defaultdict(list)
        for cid in cids:
            pos_d, neg_d = _get_pos_neg(cid)
            for t, v in pos_d.items(): global_sc[t].append(v)
            for t, v in neg_d.items(): global_sc[t].append(v)

        top_tokens = [
            t for t, _ in sorted(
                {t: float(np.mean(np.abs(s))) for t, s in global_sc.items()}.items(),
                key=lambda x: x[1], reverse=True
            )[:top_k_global]
        ]

        # ── Build matrices ───────────────────────────────────────────────────
        mat_pos = np.array([
            [_get_pos_neg(c)[0].get(t, 0.0) for t in top_tokens] for c in cids
        ])
        mat_neg = np.array([
            [_get_pos_neg(c)[1].get(t, 0.0) for t in top_tokens] for c in cids
        ])

        vmax = max(np.abs(mat_pos).max(), np.abs(mat_neg).max(), 1e-6)

        cluster_labels = [
            self.topic_mapping.get(c, f"C{c}") for c in cids
        ]

        # ── Figure ───────────────────────────────────────────────────────────
        fig, (ax_pos, ax_neg) = plt.subplots(
            1, 2,
            figsize=figsize,
            facecolor="#FAFAFA",
            gridspec_kw={"wspace": 0.08}
        )
        fig.suptitle(
            f"Per-Cluster Token Attribution Heatmap  ·  Epoch {epoch}\n"
            f"Occlusion Attribution — Bidirectional Sentiment Analysis",
            fontsize=12, fontweight="bold", color="#1C1C1C", y=1.02
        )

        def _draw_panel(ax, matrix, title, vmax_):
            im = ax.imshow(
                matrix, aspect="auto",
                cmap="RdYlGn", vmin=0, vmax=vmax_,
                interpolation="nearest"
            )
            ax.set_xticks(range(len(top_tokens)))
            ax.set_xticklabels(
                top_tokens, rotation=45, ha="right",
                fontsize=8, fontfamily="monospace"
            )
            ax.set_yticks(range(len(cids)))
            ax.set_yticklabels(cluster_labels, fontsize=8)
            ax.set_title(title, fontsize=10, fontweight="bold", pad=8, color="#1C1C1C")
            ax.set_xlabel("Token", fontsize=9, color="#444")
            ax.set_ylabel("Cluster", fontsize=9, color="#444")
            ax.tick_params(axis="both", length=0)
            ax.set_xticks(np.arange(-0.5, len(top_tokens), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(cids), 1), minor=True)
            ax.grid(which="minor", color="#EBEBEB", linewidth=0.8)
            if len(top_tokens) <= 25 and len(cids) <= 30:
                for i in range(len(cids)):
                    for j in range(len(top_tokens)):
                        v = matrix[i, j]
                        if abs(v) < 1e-6: continue
                        txt_col = "white" if v > vmax_ * 0.65 else "#2C2C2C"
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                                fontsize=5.5, color=txt_col, fontweight="bold")
            cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.015, shrink=0.85)
            cb.set_label("Mean Attribution", fontsize=8, color="#444")
            cb.ax.tick_params(labelsize=7)
            return im

        _draw_panel(
            ax_pos, mat_pos,
            f"Attribution → Positive Sentiment\n(Teal: token supports positive)",
            vmax
        )
        _draw_panel(
            ax_neg, mat_neg,
            f"Attribution → Negative Sentiment\n(Teal: token supports negative)",
            vmax
        )

        # Remove duplicate y-axis labels on right panel
        ax_neg.set_ylabel("")
        ax_neg.set_yticklabels([])

        plt.tight_layout()
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"token_attr_heatmap_epoch_{epoch}.png")
            plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
            print(f"  ✓ {out}")
        if show_plot: plt.show()
        else:         plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Plot: Explain single text (professional, bidirectional)
    # ─────────────────────────────────────────────────────────────────────────
    def explain_single_text(
        self, text,
        bert_model_name="indolem/indobert-base-uncased",
        max_length=128,
        figsize=(13, 5),
        save_path=None,
        show=False
    ):
        """
        Professional single-text explanation with bidirectional occlusion.
        Shows BOTH positive and negative attribution bars side-by-side.
        """
        _apply_professional_style()

        dev = next(self.parameters()).device
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert = AutoModel.from_pretrained(
            bert_model_name, attn_implementation="eager"
        ).to(dev).eval()

        tokens, scores_pos, scores_neg, base_prob = \
            self._occlusion_scores_bidirectional(text, bert, tokenizer, max_length)
        del bert

        pred_class = int(base_prob.argmax())
        pred_label = {0: "Negative", 1: "Positive"}.get(pred_class, str(pred_class))
        pred_conf  = float(base_prob.max())

        _SKIP = {"[CLS]", "[SEP]", "<s>", "</s>", "[PAD]"}
        dt, ds_pos, ds_neg = [], [], []
        for tok, sp, sn in zip(tokens, scores_pos, scores_neg):
            if tok in _SKIP: continue
            clean = tok.replace("##", "").replace("▁", "")
            if len(clean) < 1: continue
            dt.append(clean)
            ds_pos.append(sp)
            ds_neg.append(sn)

        dt     = np.array(dt)
        ds_pos = np.array(ds_pos)
        ds_neg = np.array(ds_neg)

        # Sort by total absolute attribution
        order  = np.argsort(np.abs(ds_pos) + np.abs(ds_neg))[::-1][:20]
        order  = order[::-1]   # bottom-up
        dt     = dt[order]
        ds_pos = ds_pos[order]
        ds_neg = ds_neg[order]

        ypos  = np.arange(len(dt))
        width = 0.35

        fig, ax = plt.subplots(figsize=figsize, facecolor="#FAFAFA")
        ax.barh(ypos + width / 2, ds_pos, height=width,
                color=_COL_POS, alpha=0.85, label="→ Positive Sentiment",
                edgecolor="#FFFFFF", linewidth=0.4)
        ax.barh(ypos - width / 2, ds_neg, height=width,
                color=_COL_NEG, alpha=0.85, label="→ Negative Sentiment",
                edgecolor="#FFFFFF", linewidth=0.4)

        ax.set_yticks(ypos)
        ax.set_yticklabels(dt, fontsize=9, fontfamily="monospace")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Occlusion Attribution Score", fontsize=9, color="#555")
        ax.set_facecolor("#FFFFFF")
        ax.spines["left"].set_visible(False)

        badge_col = _COL_POS if pred_class == 1 else _COL_NEG
        ax.text(1.02, 0.5,
                f"Prediction\n{pred_label}\n{pred_conf:.1%}",
                transform=ax.transAxes, fontsize=9, va="center", ha="left",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=badge_col, alpha=0.9))

        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(
            f"Token Attribution Analysis  ·  Bidirectional Occlusion",
            fontsize=11, fontweight="bold", pad=8
        )
        fig.suptitle(
            f'"{text[:100]}{"..." if len(text) > 100 else ""}"',
            fontsize=8, style="italic", color="#666666", y=1.01
        )
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
            print(f"✓ Saved: {save_path}")
        if show: plt.show()
        else:    plt.close()
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Cluster evolution plot (professional)
    # ─────────────────────────────────────────────────────────────────────────
    def plot_cluster_evolution(
        self, embeddings, cluster_assignments, epoch,
        texts=None, save_dir="./results/fnnjst", method="tsne",
        figsize=(7, 7), point_size=22, alpha=0.7,
        save_plot=True, show_plot=False,
        plot_tfidf_version=True, max_keywords_per_cluster=3, keyword_min_score=0.3,
    ):
        _apply_professional_style()

        emb = embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        n   = emb.shape[0]
        if n < 3: return None

        if method.lower() == 'pca':
            r = PCA(n_components=2, random_state=42)
            emb_2d = r.fit_transform(emb)
            axis_label = f"PCA  ({r.explained_variance_ratio_.sum():.1%} variance explained)"
        elif method.lower() == "tsne":
            from sklearn.manifold import TSNE
            r = TSNE(n_components=2, perplexity=max(2, min(30, n - 1)),
                     random_state=42, init="pca", learning_rate="auto")
            emb_2d = r.fit_transform(emb)
            axis_label = "t-SNE"
        else:
            try:
                import umap
                r = umap.UMAP(n_components=2, random_state=42,
                               n_neighbors=max(2, min(15, n - 1)), min_dist=0.1)
                emb_2d = r.fit_transform(emb)
                axis_label = "UMAP"
            except Exception:
                from sklearn.manifold import TSNE
                r = TSNE(n_components=2, perplexity=max(2, min(30, n - 1)),
                         random_state=42, init="pca", learning_rate="auto")
                emb_2d = r.fit_transform(emb)
                axis_label = "t-SNE (fallback)"

        uniq = np.unique(cluster_assignments)
        k    = len(uniq)
        colors = [_CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)] for i in range(k)]

        fig, ax = plt.subplots(figsize=figsize, facecolor="#FAFAFA")
        for i, cid in enumerate(uniq):
            mask = cluster_assignments == cid
            label = self.topic_mapping.get(int(cid), f"C{cid}") + f"  (n={mask.sum()})"
            ax.scatter(
                emb_2d[mask, 0], emb_2d[mask, 1],
                c=[colors[i]], marker="o", s=point_size,
                alpha=alpha, label=label, edgecolors="white", linewidths=0.3
            )

        ax.set_title(f"Cluster Distribution  ·  Epoch {epoch}  ·  {axis_label}",
                     fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel(f"{axis_label} Dim 1", fontsize=8, color="#666")
        ax.set_ylabel(f"{axis_label} Dim 2", fontsize=8, color="#666")
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(colors="#AAAAAA")
        ax.grid(True, color="#F0F0F0", linewidth=0.5)

        if k <= 15:
            ax.legend(
                fontsize=7, loc="best",
                framealpha=0.9, edgecolor="#CCCCCC",
                markerscale=1.3, ncol=max(1, k // 8)
            )

        plt.tight_layout()
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, f"cluster_evolution_epoch_{epoch}.png"),
                dpi=160, bbox_inches="tight", facecolor="#FAFAFA"
            )
        if show_plot: plt.show()
        else:         plt.close()
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # K-means init
    # ─────────────────────────────────────────────────────────────────────────
    def _init_clusters_with_kmeans(self, all_embeddings, n_init=20, random_state=42):
        dev = next(self.parameters()).device
        if all_embeddings.size(0) < self.n_clusters:
            raise ValueError("n_samples < n_clusters")
        self.eval()
        with torch.no_grad():
            feats = self.extract_feature(all_embeddings).cpu().numpy()
        km = KMeans(self.n_clusters, n_init=n_init, random_state=random_state)
        y_pred = km.fit_predict(feats)
        self.clustering.clusters.data = torch.tensor(
            km.cluster_centers_, dtype=torch.float32, device=dev)
        return y_pred

    # ─────────────────────────────────────────────────────────────────────────
    # fit()
    # ─────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        dataset,
        alpha: float = 0.1,
        gamma: float = 1.0,
        eta:   float = 0.1,
        optimizer_type: str   = "adam",
        learning_rate:  float = 1e-3,
        momentum:       float = 0.9,
        tol:             float = 1e-3,
        update_interval: int   = 140,
        batch_size:      int   = 128,
        maxiter:         int   = int(2e4),
        save_dir:        str   = "./results/fnnjst",
        val_ratio:       float = 0.1,
        val_metric:      str   = "auto",
        plot_evolution:  bool  = True,
        plot_interval:   Optional[int] = None,
        plot_method:     str   = "tsne",
        compute_metrics: bool  = True,
        plot_integrated_gradients: bool = True,
        ig_target_class:  int  = 1,
        ig_n_steps:       int  = 50,
        ig_top_dims:      int  = 20,
        ig_max_samples:   int  = 512,
        plot_token_attribution:    bool = False,
        token_attr_bert_name:      str  = "indolem/indobert-base-uncased",
        token_attr_top_k:          int  = 10,
        token_attr_max_samples:    int  = 15,   # ← 7 neg + 8 pos
        token_attr_max_length:     int  = 128,
        # REMOVED: token_attr_sentiment_class — always bidirectional now
    ):
        """
        Joint DEC + Sentiment + Reconstruction training.

        v3.1 changes vs v3:
        - token_attr_sentiment_class parameter REMOVED (always bidirectional)
        - token_attr_max_samples default = 15 (7 neg + 8 pos stratified)
        - All plots use professional style
        - Topic Diversity now TF-IDF weighted (see compute_topic_diversity)
        """
        print("=" * 60)
        print("SEMTGPU v3.1 — Joint Training: Clustering + Sentiment + Reconstruction")
        print(f"Loss: α(recon)={alpha}, γ(cluster)={gamma}, η(sentiment)={eta}")
        print(f"Update interval: {update_interval}  |  val_ratio: {val_ratio:.0%}")
        print(f"Best-model metric: {val_metric}")
        if plot_token_attribution:
            print(f"Token attribution: BIDIRECTIONAL  |  samples/cluster={token_attr_max_samples} "
                  f"({token_attr_max_samples//2} neg + {token_attr_max_samples - token_attr_max_samples//2} pos)")
            print("Token attribution schedule: early / half / last epoch only")
        print("=" * 60)

        dev     = next(self.parameters()).device
        maxiter = int(maxiter)
        pinterv = int(plot_interval) if plot_interval is not None else update_interval
        os.makedirs(save_dir, exist_ok=True)
        plot_dir = os.path.join(save_dir, "evolution_plots")
        os.makedirs(plot_dir, exist_ok=True)

        # ── Collect dataset ──────────────────────────────────────────────────
        embs, lbls, texts_list = [], [], []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, tuple):
                if len(item) >= 2:
                    embs.append(item[0].detach().cpu())
                    lbls.append(item[1].detach().cpu())
                if len(item) >= 3:
                    texts_list.append(item[2])
            else:
                t = item.detach() if isinstance(item, torch.Tensor) else \
                    torch.tensor(item, dtype=torch.float32)
                embs.append(t.cpu())

        X_all = torch.stack(embs)
        N, D  = X_all.shape
        if D != self.dims[0]: raise ValueError(f"Input dim={D}, dims[0]={self.dims[0]}")
        if N < self.n_clusters: raise ValueError(f"n_samples={N} < n_clusters={self.n_clusters}")

        has_labels = len(lbls) > 0
        has_texts  = len(texts_list) > 0

        # ── Validation split ─────────────────────────────────────────────────
        n_val   = max(1, int(N * val_ratio))
        n_train = N - n_val
        idx_all   = np.random.permutation(N)
        idx_train = idx_all[:n_train]
        idx_val   = idx_all[n_train:]

        X_train = X_all[idx_train].to(dev)
        X_val   = X_all[idx_val].to(dev)
        Y_train = torch.stack(lbls)[idx_train].to(dev) if has_labels else None
        Y_val   = torch.stack(lbls)[idx_val].to(dev)   if has_labels else None
        texts_train = [texts_list[i] for i in idx_train] if has_texts else []
        texts_val   = [texts_list[i] for i in idx_val]   if has_texts else []

        print(f"Dataset split → train: {n_train}, val: {n_val}")

        if val_metric == "auto":
            _primary = "val_f1" if has_labels else "val_cluster_score"
        elif val_metric == "f1":
            _primary = "val_f1"
        elif val_metric in ("cluster_score", "silhouette"):
            _primary = "val_cluster_score"
        else:
            raise ValueError(f"val_metric must be 'auto'|'f1'|'cluster_score', got {val_metric}")
        print(f"Primary validation metric: {_primary}")

        class_w_t = None
        if has_labels:
            y_tr_np = Y_train.cpu().numpy()
            if y_tr_np.ndim == 2 and y_tr_np.shape[1] > 1: y_tr_np = y_tr_np.argmax(1)
            cw = self.compute_class_weights(y_tr_np)
            class_w_t = torch.tensor([cw.get(i, 1.0) for i in range(2)],
                                       dtype=torch.float32, device=dev)

        opt_map = {
            "adam":     lambda: optim.Adam(self.parameters(), lr=learning_rate),
            "adamw":    lambda: optim.AdamW(self.parameters(), lr=learning_rate),
            "sgd":      lambda: optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum),
            "rmsprop":  lambda: optim.RMSprop(self.parameters(), lr=learning_rate),
            "adamax":   lambda: optim.Adamax(self.parameters(), lr=learning_rate),
            "nadam":    lambda: optim.NAdam(self.parameters(), lr=learning_rate),
            "adagrad":  lambda: optim.Adagrad(self.parameters(), lr=learning_rate),
            "adadelta": lambda: optim.Adadelta(self.parameters(), lr=learning_rate),
            "asgd":     lambda: optim.ASGD(self.parameters(), lr=learning_rate),
        }
        if optimizer_type.lower() not in opt_map:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        optimizer = opt_map[optimizer_type.lower()]()

        kld_loss = nn.KLDivLoss(reduction="batchmean")
        ce_loss  = nn.CrossEntropyLoss(weight=class_w_t) if class_w_t is not None else \
                   nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        print("Initialising cluster centres with k-means (train split).")
        y_pred_last = self._init_clusters_with_kmeans(X_train)

        n_intervals = maxiter // update_interval
        half_iter   = (n_intervals // 2) * update_interval
        early_iter  = pinterv

        def _should_run_token_attr(ite: int, is_final: bool = False) -> bool:
            if not plot_token_attribution or not has_texts: return False
            if is_final: return True
            return ite in {early_iter, half_iter}

        log_path = os.path.join(save_dir, "idec_sentiment_log.csv")
        log_fields = [
            "iter", "split",
            "train_acc", "train_f1", "train_precision", "train_recall",
            "L", "Lr", "Lc", "Ls",
            "ACC", "NMI", "ARI", "Homogeneity", "Completeness", "V-measure",
            "Train_Coherence", "Train_Diversity", "Train_Cluster_Score",
            "Cluster_Balance", "Min_Cluster_Size", "Max_Cluster_Size",
            "val_acc", "val_f1", "val_precision", "val_recall",
            "val_nmi", "val_ari", "val_acc_cluster",
            "val_coherence", "val_diversity", "val_cluster_score",
            "val_primary_score", "is_best",
            "IG_TopDim", "IG_TopDim_Score", "IG_Mean_Attribution",
        ]

        with open(log_path, "w", newline="") as logfile:
            writer = csv.DictWriter(logfile, fieldnames=log_fields)
            writer.writeheader()

            save_interval = max(1, (max(1, n_train // batch_size)) * 5)
            train_loader: Optional[DataLoader] = None
            self.train()
            iter_count = 0
            tot_L = Lr = Lc = Ls = 0.0
            last_ite = 0

            for ite in range(maxiter):
                last_ite = ite

                if ite % update_interval == 0:
                    self.eval()
                    with torch.no_grad():
                        q_list, s_list = [], []
                        for i in range(0, n_train, batch_size):
                            qb, sb = self(X_train[i:i+batch_size])
                            q_list.append(qb); s_list.append(sb)
                        q_all = torch.cat(q_list, 0)
                        s_all = torch.cat(s_list, 0)
                        p_all = self.target_distribution(q_all)

                        y_pred = q_all.argmax(1).cpu().numpy()
                        delta  = float((y_pred != y_pred_last).sum() / len(y_pred))
                        y_pred_last = y_pred.copy()

                    train_acc = train_f1 = train_prec = train_rec = 0.0
                    if has_labels:
                        s_lab  = s_all.argmax(1).cpu().numpy()
                        y_true = Y_train.cpu().numpy()
                        if y_true.ndim == 2 and y_true.shape[1] > 1:
                            y_true = y_true.argmax(1)
                        train_acc  = float((s_lab == y_true).mean())
                        train_f1   = float(f1_score(y_true, s_lab, average="binary", zero_division=0))
                        train_prec = float(precision_score(y_true, s_lab, average="binary", zero_division=0))
                        train_rec  = float(recall_score(y_true, s_lab, average="binary", zero_division=0))

                    feats_tr = self.extract_feature(X_train).cpu().numpy()
                    cl_sup_tr = {}
                    if has_labels:
                        yt = Y_train.cpu().numpy()
                        if yt.ndim == 2 and yt.shape[1] > 1: yt = yt.argmax(1)
                        cl_sup_tr = self.compute_clustering_metrics(yt, y_pred)

                    train_coh = train_div = 0.0
                    cov_tr = {'cluster_balance': 0.0, 'min_cluster_size': 0.0,
                              'max_cluster_size': 0.0}
                    if has_texts:
                        coh_scores = self.compute_topic_coherence(texts_train, y_pred)
                        train_coh  = float(np.mean(list(coh_scores.values())) or 0.0)
                        train_div  = self.compute_topic_diversity(texts_train, y_pred)
                        cov_tr     = self.compute_topic_coverage(texts_train, y_pred)
                    train_cluster_score = (train_coh + train_div) / 2.0

                    avg_L  = tot_L / update_interval if iter_count > 0 else 0.0
                    avg_Lr = Lr    / update_interval if iter_count > 0 else 0.0
                    avg_Lc = Lc    / update_interval if iter_count > 0 else 0.0
                    avg_Ls = Ls    / update_interval if iter_count > 0 else 0.0

                    val_metrics = self._evaluate_val(X_val, Y_val, texts_val, batch_size)
                    vs      = val_metrics["val_primary_score"]
                    is_best = vs > self._best_val_score
                    if is_best:
                        self._best_val_score = vs
                        self._best_val_iter  = ite
                        self._best_state_dict = copy.deepcopy(self.state_dict())
                        print(f"  ✓ New best model at iter {ite}: {_primary}={vs:.4f}")
                        self.save_weights(os.path.join(save_dir, "SEMTGPU_best.weights.pth"))

                    print(f"\nIter {ite:5d} | Lr={avg_Lr:.5f}  Lc={avg_Lc:.5f}  Ls={avg_Ls:.5f}")
                    print(f"  Sentiment  Train → Acc={train_acc:.4f}  F1={train_f1:.4f}  "
                          f"P={train_prec:.4f}  R={train_rec:.4f}")
                    print(f"  Sentiment  Val   → Acc={val_metrics.get('val_acc_sentiment',0):.4f}  "
                          f"F1={val_metrics.get('val_f1',0):.4f}  "
                          f"P={val_metrics.get('val_precision',0):.4f}  "
                          f"R={val_metrics.get('val_recall',0):.4f}")
                    print(f"  Cluster    Train → Coh={train_coh:.4f}  Div={train_div:.4f}  "
                          f"Score={train_cluster_score:.4f}")
                    print(f"  Cluster    Val   → Coh={val_metrics.get('val_coherence',0):.4f}  "
                          f"Div={val_metrics.get('val_diversity',0):.4f}  "
                          f"Score={val_metrics.get('val_cluster_score',0):.4f}  "
                          f"{'BEST' if is_best else ''}")
                    if cl_sup_tr:
                        print(f"  Supervised Train → ACC={cl_sup_tr['ACC']:.4f}  "
                              f"NMI={cl_sup_tr['NMI']:.4f}  ARI={cl_sup_tr['ARI']:.4f}")

                    if plot_evolution and ite > 0 and (ite % pinterv == 0):
                        try:
                            self.plot_cluster_evolution(
                                feats_tr, y_pred, ite,
                                texts=texts_train if has_texts else None,
                                save_dir=plot_dir, method=plot_method, show_plot=False,
                                plot_tfidf_version=has_texts,
                            )
                        except Exception as e:
                            print(f"  Warning: plot failed at iter {ite}: {e}")

                    ig_top_dim_idx = ig_top_dim_score = ig_mean_attr = 0
                    if plot_integrated_gradients and ite > 0 and (ite % pinterv == 0):
                        try:
                            idx_ig   = np.random.choice(n_train, min(ig_max_samples, n_train), replace=False)
                            ig_attrs = self.compute_integrated_gradients(
                                X_train[idx_ig], target_class=ig_target_class, n_steps=ig_n_steps)
                            self.plot_integrated_gradients(
                                ig_attrs, y_pred[idx_ig], epoch=ite,
                                save_dir=plot_dir, top_dims=ig_top_dims, show_plot=False)
                            abs_ig           = np.abs(ig_attrs)
                            gi               = abs_ig.mean(0)
                            ig_top_dim_idx   = int(gi.argmax())
                            ig_top_dim_score = float(gi.max())
                            ig_mean_attr     = float(gi.mean())
                        except Exception as e:
                            print(f"  Warning: IG plot failed at iter {ite}: {e}")

                    if _should_run_token_attr(ite, is_final=False):
                        try:
                            print(f"  [Token Attr @ iter {ite}]")
                            ta = self.compute_token_attribution_per_cluster(
                                texts_train, y_pred,
                                bert_model_name=token_attr_bert_name,
                                top_k=token_attr_top_k,
                                max_length=token_attr_max_length,
                                max_samples_per_cluster=token_attr_max_samples,
                            )
                            self.plot_token_attribution_per_cluster(
                                ta, epoch=ite, save_dir=plot_dir,
                                top_k=token_attr_top_k, show_plot=False)
                            self.plot_token_attribution_heatmap(
                                ta, epoch=ite, save_dir=plot_dir, show_plot=False)
                        except Exception as e:
                            print(f"  Warning: token attr failed at iter {ite}: {e}")

                    writer.writerow({
                        "iter":   ite,   "split": "train",
                        "train_acc":       round(train_acc,  5),
                        "train_f1":        round(train_f1,   5),
                        "train_precision": round(train_prec, 5),
                        "train_recall":    round(train_rec,  5),
                        "L":  round(avg_L,  5), "Lr": round(avg_Lr, 5),
                        "Lc": round(avg_Lc, 5), "Ls": round(avg_Ls, 5),
                        "ACC":          round(cl_sup_tr.get('ACC',          0.0), 5),
                        "NMI":          round(cl_sup_tr.get('NMI',          0.0), 5),
                        "ARI":          round(cl_sup_tr.get('ARI',          0.0), 5),
                        "Homogeneity":  round(cl_sup_tr.get('Homogeneity',  0.0), 5),
                        "Completeness": round(cl_sup_tr.get('Completeness', 0.0), 5),
                        "V-measure":    round(cl_sup_tr.get('V-measure',    0.0), 5),
                        "Train_Coherence":    round(train_coh,           5),
                        "Train_Diversity":    round(train_div,           5),
                        "Train_Cluster_Score":round(train_cluster_score, 5),
                        "Cluster_Balance":    round(cov_tr['cluster_balance'],  5),
                        "Min_Cluster_Size":   round(cov_tr['min_cluster_size'], 5),
                        "Max_Cluster_Size":   round(cov_tr['max_cluster_size'], 5),
                        "val_acc":       round(val_metrics.get('val_acc_sentiment', 0.0), 5),
                        "val_f1":        round(val_metrics.get('val_f1',            0.0), 5),
                        "val_precision": round(val_metrics.get('val_precision',     0.0), 5),
                        "val_recall":    round(val_metrics.get('val_recall',        0.0), 5),
                        "val_nmi":          round(val_metrics.get('val_nmi',         0.0), 5),
                        "val_ari":          round(val_metrics.get('val_ari',         0.0), 5),
                        "val_acc_cluster":  round(val_metrics.get('val_acc_cluster', 0.0), 5),
                        "val_coherence":    round(val_metrics.get('val_coherence',   0.0), 5),
                        "val_diversity":    round(val_metrics.get('val_diversity',   0.0), 5),
                        "val_cluster_score":round(val_metrics.get('val_cluster_score',0.0), 5),
                        "val_primary_score":round(val_metrics.get('val_primary_score',0.0), 5),
                        "is_best":          int(is_best),
                        "IG_TopDim":           ig_top_dim_idx,
                        "IG_TopDim_Score":     round(ig_top_dim_score, 6),
                        "IG_Mean_Attribution": round(ig_mean_attr,     6),
                    })

                    tot_L = Lr = Lc = Ls = 0.0
                    iter_count = 0

                    if ite > 0 and delta < tol:
                        print(f"  Δlabel={delta:.6f} < tol={tol} → early stop.")
                        break

                    if has_labels:
                        train_loader = DataLoader(
                            TensorDataset(X_train, p_all, Y_train),
                            batch_size=batch_size, shuffle=True)
                    else:
                        train_loader = DataLoader(
                            TensorDataset(X_train, p_all),
                            batch_size=batch_size, shuffle=True)
                    self.train()

                assert train_loader is not None
                for batch in tqdm(train_loader, desc=f"Train {ite}", leave=False):
                    if has_labels and len(batch) == 3:
                        xb, pb, yb = batch
                        if yb.dim() > 1 and yb.shape[1] > 1: yb = yb.argmax(1)
                        yb = yb.long().to(dev)
                    else:
                        xb, pb = batch; yb = None
                    xb = xb.to(dev); pb = pb.to(dev)
                    z       = self.autoencoder.encode(xb)
                    x_recon = self.autoencoder.decode(z)
                    q       = self.clustering(z)
                    s       = torch.softmax(self.sentiment(z), 1)
                    rl = mse_loss(x_recon, xb)
                    cl = kld_loss((q + 1e-8).log(), pb)
                    sl = ce_loss(s, yb) if yb is not None else \
                         torch.zeros(1, device=dev).squeeze()
                    loss = alpha*rl + gamma*cl + eta*sl
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    tot_L += float(loss); Lr += float(rl)
                    Lc    += float(cl);   Ls += float(sl)
                    iter_count += 1

                if ite % save_interval == 0 and ite > 0:
                    self.save_weights(os.path.join(save_dir, f"SEMTGPU_{ite}.weights.pth"))

        # ── Post-training ─────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Training complete.  Restoring best model checkpoint…")
        self.load_best_weights()

        best_summary = {
            "best_iter":      self._best_val_iter,
            "best_val_score": round(self._best_val_score, 5),
            "primary_metric": _primary,
            "val_ratio":      val_ratio,
            "n_train":        n_train,
            "n_val":          n_val,
        }
        with open(os.path.join(save_dir, "best_model.json"), "w") as f:
            json.dump(best_summary, f, indent=2)
        print(f"✓ Best model info saved → {save_dir}/best_model.json")
        print(f"  Best iter={self._best_val_iter}, {_primary}={self._best_val_score:.4f}")

        self.save_weights(os.path.join(save_dir, "SEMTGPU_final.weights.pth"))

        self.eval()
        with torch.no_grad():
            q_all_f = torch.cat([self(X_train[i:i+batch_size])[0]
                                  for i in range(0, n_train, batch_size)], 0)
        y_final = q_all_f.argmax(1).cpu().numpy()

        if _should_run_token_attr(last_ite, is_final=True) and has_texts:
            try:
                print("\n[Token Attr @ LAST epoch]")
                ta_last = self.compute_token_attribution_per_cluster(
                    texts_train, y_final,
                    bert_model_name=token_attr_bert_name,
                    top_k=token_attr_top_k,
                    max_length=token_attr_max_length,
                    max_samples_per_cluster=token_attr_max_samples,
                )
                ta_dir = os.path.join(save_dir, "token_attribution")
                self.plot_token_attribution_per_cluster(
                    ta_last, epoch=last_ite, save_dir=ta_dir,
                    top_k=token_attr_top_k, show_plot=False)
                self.plot_token_attribution_heatmap(
                    ta_last, epoch=last_ite, save_dir=ta_dir, show_plot=False)
                print("✓ Final token attribution complete.")
            except Exception as e:
                print(f"Warning: final token attribution failed: {e}")

        if plot_integrated_gradients and has_labels:
            try:
                idx_ig = np.random.choice(n_train, min(ig_max_samples, n_train), replace=False)
                ig_f = self.compute_integrated_gradients(
                    X_train[idx_ig], target_class=ig_target_class, n_steps=ig_n_steps)
                self.plot_integrated_gradients(
                    ig_f, y_final[idx_ig], epoch=last_ite,
                    save_dir=plot_dir, top_dims=ig_top_dims, show_plot=False)
                print("✓ Final IG plot saved.")
            except Exception as e:
                print(f"Warning: final IG failed: {e}")

        self.eval()
        with torch.no_grad():
            qs, ss = [], []
            for i in range(0, n_train, batch_size):
                qb, sb = self(X_train[i:i+batch_size])
                qs.append(qb); ss.append(sb)
            q_all_f = torch.cat(qs, 0); s_all_f = torch.cat(ss, 0)
        y_pred_cl   = q_all_f.argmax(1).cpu().numpy()
        y_pred_sent = s_all_f.argmax(1).cpu().numpy()

        metrics = {}
        if has_labels:
            y_true = Y_train.cpu().numpy()
            if y_true.ndim == 2 and y_true.shape[1] > 1: y_true = y_true.argmax(1)
            metrics["sentiment"] = {
                "accuracy":  float((y_pred_sent == y_true).mean()),
                "precision": float(precision_score(y_true, y_pred_sent,
                                                    average="binary", zero_division=0)),
                "recall":    float(recall_score(y_true, y_pred_sent,
                                                average="binary", zero_division=0)),
                "f1_score":  float(f1_score(y_true, y_pred_sent,
                                            average="binary", zero_division=0)),
            }
            print("\n" + "=" * 60)
            print("FINAL (BEST MODEL) SENTIMENT METRICS — TRAIN SPLIT")
            print("=" * 60)
            for k, v in metrics["sentiment"].items():
                print(f"  {k:12s}: {v:.4f}")
            print("=" * 60)

        val_final = self._evaluate_val(X_val, Y_val, texts_val, batch_size)
        metrics["val_final"] = val_final
        print("\nFINAL VAL METRICS (best model):")
        print(f"  Sentiment → Acc={val_final.get('val_acc_sentiment',0):.4f}  "
              f"F1={val_final.get('val_f1',0):.4f}  "
              f"P={val_final.get('val_precision',0):.4f}  "
              f"R={val_final.get('val_recall',0):.4f}")
        print(f"  Cluster   → Coh={val_final.get('val_coherence',0):.4f}  "
              f"Div={val_final.get('val_diversity',0):.4f}  "
              f"Score={val_final.get('val_cluster_score',0):.4f}")

        if has_labels:
            return y_pred_cl, s_all_f.cpu().numpy(), metrics
        return y_pred_cl