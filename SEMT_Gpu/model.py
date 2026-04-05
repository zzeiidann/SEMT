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
from matplotlib.colors import LinearSegmentedColormap

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
# Device
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_RWG = LinearSegmentedColormap.from_list(
    "rwg", ["#d62728", "#f7f7f7", "#2ca02c"], N=256
)
warnings.filterwarnings("ignore", category=UserWarning)


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

    v3 Changes (vs v2):
      • Train metrics now include F1 / Precision / Recall (consistent with val)
      • Val cluster metric changed: silhouette (geometric) → coherence + diversity (semantic)
      • val_metric="auto" now uses val_f1 (if labels) else val_cluster_score
        where val_cluster_score = mean(coherence, diversity)
      • Print output aligned: train and val show the same metric set
      • CSV log updated accordingly
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
        """
        Full pass over validation set.

        Sentiment metrics (if labels available):
            val_acc_sentiment, val_f1, val_precision, val_recall
            val_nmi, val_ari, val_acc_cluster

        Cluster quality metrics (semantic, text-based):
            val_coherence  — mean NPMI topic coherence across clusters
            val_diversity  — unique top-words ratio across clusters
            val_cluster_score = mean(coherence, diversity)

        Fallback when no texts provided:
            val_coherence = silhouette score (geometric fallback)
            val_diversity = 0.0

        Primary metric for best-model selection:
            val_f1           (if labels available)
            val_cluster_score (otherwise)
        """
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

        # ── Sentiment metrics ────────────────────────────────────────────────
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

        # ── Semantic cluster quality (coherence + diversity) ─────────────────
        # Preferred over silhouette for text: coherence measures whether top
        # words in a cluster co-occur (topic is real), diversity measures
        # whether clusters cover distinct vocabulary (no redundancy).
        if texts_val and len(texts_val) > 0:
            coh_scores = self.compute_topic_coherence(texts_val, y_pred_cluster)
            metrics["val_coherence"] = float(np.mean(list(coh_scores.values())) or 0.0)
            metrics["val_diversity"] = self.compute_topic_diversity(texts_val, y_pred_cluster)
        else:
            # Geometric fallback when no texts available
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

        # ── Primary score for best-model selection ───────────────────────────
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
        clusters: Dict[int, List[str]] = {}
        for i, cid in enumerate(cluster_assignments):
            clusters.setdefault(int(cid), []).append(texts[i])
        all_sets = []
        for c_texts in clusters.values():
            words = [w for w in " ".join(c_texts).lower().split()
                     if w not in self.stop_words and len(w) > 2]
            all_sets.append(set(w for w, _ in Counter(words).most_common(top_n)))
        if len(all_sets) < 2: return 0.0
        uniq  = set().union(*all_sets)
        total = sum(len(s) for s in all_sets)
        return float(len(uniq) / total) if total else 0.0

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
                                   figsize=(14, 6), top_dims=20,
                                   save_plot=True, show_plot=False):
        abs_attr = np.abs(attributions)
        global_imp = abs_attr.mean(0)
        top_idx  = np.argsort(global_imp)[-top_dims:][::-1]
        uniq     = np.unique(cluster_assignments)
        cl_ig    = np.array([
            abs_attr[cluster_assignments == c][:, top_idx].mean(0)
            if (cluster_assignments == c).sum() > 0 else np.zeros(top_dims)
            for c in uniq
        ])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                        gridspec_kw={'width_ratios': [1, 1.6]})
        fig.suptitle(f"Integrated Gradients — Epoch {epoch}", fontsize=13,
                     fontweight='bold', y=1.02)

        ni = global_imp[top_idx] / (global_imp[top_idx].max() + 1e-8)
        bars = ax1.barh([f"dim {i}" for i in top_idx], global_imp[top_idx],
                        color=plt.cm.RdYlGn(ni), edgecolor='white', linewidth=0.5)
        ax1.set_xlabel("Mean |IG|", fontsize=10)
        ax1.set_title("Top Dimensions\n(Global Sentiment Impact)", fontsize=11)
        ax1.invert_yaxis(); ax1.spines[['top','right']].set_visible(False)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        for bar, val in zip(bars, global_imp[top_idx]):
            ax1.text(val + global_imp[top_idx].max() * 0.01,
                     bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', ha='left', fontsize=7)

        vmax = cl_ig.max() if cl_ig.max() > 0 else 1.0
        im = ax2.imshow(cl_ig, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
        ax2.set_xticks(range(top_dims))
        ax2.set_xticklabels([f"d{i}" for i in top_idx], rotation=90, fontsize=7)
        ax2.set_yticks(range(len(uniq)))
        ax2.set_yticklabels([f"C{c} ({(cluster_assignments==c).sum()})" for c in uniq], fontsize=9)
        ax2.set_title("Per-Cluster Attribution Heatmap", fontsize=11)
        ax2.set_xlabel("Embedding Dimension", fontsize=10)
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).set_label("Mean |IG|", fontsize=9)
        if top_dims <= 30 and len(uniq) <= 20:
            for i in range(len(uniq)):
                for j in range(top_dims):
                    v = cl_ig[i, j]
                    ax2.text(j, i, f'{v:.3f}', ha='center', va='center',
                             fontsize=6, color='white' if v > vmax*0.6 else '#333')
        plt.tight_layout()
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"integrated_gradients_epoch_{epoch}.png")
            plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  ✓ IG plot saved: {out}")
        if show_plot: plt.show()
        else:         plt.close()
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Token Attribution (occlusion)
    # ─────────────────────────────────────────────────────────────────────────
    def _bert_cls_embedding(self, text, bert, tokenizer, max_length=128):
        dev = next(self.parameters()).device
        enc = tokenizer(text, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(dev)
        with torch.no_grad():
            out = bert(**enc)
        return out.last_hidden_state[:, 0, :], None, enc

    def _occlusion_scores(self, text, bert, tokenizer, sentiment_class=1, max_length=128):
        dev = next(self.parameters()).device
        cls_emb, _, enc = self._bert_cls_embedding(text, bert, tokenizer, max_length)
        self.eval()
        with torch.no_grad():
            _, s = self(cls_emb)
        base_prob = s.squeeze(0).cpu().numpy()
        tokens    = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
        ids       = enc["input_ids"][0].cpu().tolist()
        mask_id   = tokenizer.mask_token_id
        scores    = np.zeros(len(ids))
        for i, tok in enumerate(tokens):
            if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]"): continue
            masked    = ids.copy(); masked[i] = mask_id
            inp       = {k: enc[k].clone() for k in enc}
            inp["input_ids"] = torch.tensor([masked], device=dev)
            with torch.no_grad():
                out = bert(**inp)
                _, sm = self(out.last_hidden_state[:, 0, :])
            scores[i] = float(base_prob[sentiment_class] - sm.squeeze(0).cpu().numpy()[sentiment_class])
        return tokens, scores, base_prob

    def compute_token_attribution_per_cluster(
        self, texts, cluster_assignments,
        bert_model_name="indolem/indobert-base-uncased",
        sentiment_class=1, top_k=10, max_length=128, max_samples_per_cluster=50,
    ):
        dev = next(self.parameters()).device
        print(f"  Loading BERT: {bert_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert = AutoModel.from_pretrained(bert_model_name,
                                          attn_implementation="eager").to(dev).eval()
        cl_idx = defaultdict(list)
        for i, cid in enumerate(cluster_assignments): cl_idx[int(cid)].append(i)
        cl_tok: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.eval()
        for cid, indices in sorted(cl_idx.items()):
            if len(indices) > max_samples_per_cluster:
                indices = list(np.random.choice(indices, max_samples_per_cluster, replace=False))
            print(f"  Cluster {cid} ({len(indices)} samples)...", end=" ", flush=True)
            for idx in indices:
                try:
                    toks, scs, _ = self._occlusion_scores(
                        texts[idx], bert, tokenizer, sentiment_class, max_length)
                    for tok, sc in zip(toks, scs):
                        if tok in ("[CLS]","[SEP]","<s>","</s>","[PAD]"): continue
                        clean = tok.replace("##","").replace("▁","").strip()
                        if len(clean) < 2: continue
                        cl_tok[cid][clean].append(sc)
                except Exception: continue
            print("✓")
        result = {}
        for cid, td in cl_tok.items():
            means = {t: float(np.mean(sc)) for t, sc in td.items() if len(sc) >= 2}
            result[cid] = dict(sorted(means.items(), key=lambda x: abs(x[1]),
                                       reverse=True)[:top_k])
        del bert
        return result

    def plot_token_attribution_per_cluster(
        self, cluster_token_scores, sentiment_class=1, epoch=0,
        save_dir="./results/fnnjst", figsize=(16, 10), top_k=10,
        max_clusters_per_row=4, save_plot=True, show_plot=False,
    ):
        sent_label = {0:"Negative",1:"Positive"}.get(sentiment_class, str(sentiment_class))
        cids = sorted(cluster_token_scores.keys())
        if not cids: return None
        ncols = min(max_clusters_per_row, len(cids))
        nrows = (len(cids) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(figsize[0], figsize[1]*nrows/max(2,nrows)),
                                  facecolor="white")
        axes = np.array(axes).reshape(-1)
        all_vals = [v for d in cluster_token_scores.values() for v in d.values()]
        vmax = max(abs(v) for v in all_vals) if all_vals else 1.0
        for i, cid in enumerate(cids):
            ax = axes[i]; td = cluster_token_scores[cid]
            if not td:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="#aaa"); ax.axis("off"); continue
            items = sorted(td.items(), key=lambda x: x[1], reverse=True)[:top_k]
            toks_ = [t for t,_ in items]; scs_ = [s for _,s in items]
            colors = [_RWG(0.5 + 0.5*(s/vmax)) for s in scs_]
            yp = np.arange(len(toks_))
            ax.barh(yp, scs_, color=colors, edgecolor="white", linewidth=0.3, height=0.7)
            ax.set_yticks(yp); ax.set_yticklabels(toks_, fontsize=8, fontfamily="monospace")
            ax.invert_yaxis(); ax.axvline(0, color="#888", linewidth=0.7, linestyle="--")
            ax.spines[["top","right"]].set_visible(False)
            ax.set_xlabel("Attribution", fontsize=7, color="#555")
            ax.set_title(f"{self.topic_mapping.get(cid, f'C{cid}')}  (n≈{len(td)})",
                         fontsize=9, fontweight="bold", pad=5)
            for bar, val in zip(ax.patches, scs_):
                ax.text(val + vmax*0.01 if val >= 0 else val - vmax*0.01,
                        bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center",
                        ha="left" if val >= 0 else "right", fontsize=6)
        for j in range(i+1, len(axes)): axes[j].axis("off")
        fig.legend(handles=[mpatches.Patch(color="#2ca02c", label=f"Supports {sent_label}"),
                             mpatches.Patch(color="#d62728", label=f"Opposes {sent_label}")],
                   loc="upper right", fontsize=9, framealpha=0.9, ncol=2)
        fig.suptitle(f"Token Attribution per Cluster — Epoch {epoch}", fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"token_attr_per_cluster_epoch_{epoch}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print(f"  ✓ {out}")
        if show_plot: plt.show()
        else:         plt.close()
        return fig

    def plot_token_attribution_heatmap(
        self, cluster_token_scores, sentiment_class=1, epoch=0,
        save_dir="./results/fnnjst", top_k_global=20, figsize=(16,8),
        save_plot=True, show_plot=False,
    ):
        sent_label = {0:"Negative",1:"Positive"}.get(sentiment_class, str(sentiment_class))
        global_sc: Dict[str, List[float]] = defaultdict(list)
        for td in cluster_token_scores.values():
            for tok, sc in td.items(): global_sc[tok].append(sc)
        top_tokens = [t for t, _ in sorted(
            {t: float(np.mean(s)) for t, s in global_sc.items()}.items(),
            key=lambda x: abs(x[1]), reverse=True)[:top_k_global]]
        cids   = sorted(cluster_token_scores.keys())
        matrix = np.array([[cluster_token_scores[c].get(t, 0.0) for t in top_tokens] for c in cids])
        vmax   = np.abs(matrix).max() if matrix.any() else 1.0
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        im = ax.imshow(matrix, cmap=_RWG, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(top_tokens)))
        ax.set_xticklabels(top_tokens, rotation=45, ha="right", fontsize=9, fontfamily="monospace")
        ax.set_yticks(range(len(cids)))
        ax.set_yticklabels([self.topic_mapping.get(c, f"C{c}") for c in cids], fontsize=9)
        if len(top_tokens) <= 25 and len(cids) <= 25:
            for i in range(len(cids)):
                for j in range(len(top_tokens)):
                    v = matrix[i,j]
                    if v != 0.0:
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, fontweight="bold",
                                color="white" if abs(v) > vmax*0.5 else "#333")
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02).set_label(f"Mean Attr → {sent_label}", fontsize=10)
        ax.set_title(f"Per-Cluster Token Attribution Heatmap — Epoch {epoch}", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Token", fontsize=10); ax.set_ylabel("Cluster", fontsize=10)
        ax.set_xticks(np.arange(-0.5, len(top_tokens), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(cids), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2); ax.tick_params(which="minor", length=0)
        plt.tight_layout()
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"token_attr_heatmap_epoch_{epoch}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); print(f"  ✓ {out}")
        if show_plot: plt.show()
        else:         plt.close()
        return fig

    def explain_single_text(self, text, bert_model_name="indolem/indobert-base-uncased",
                             sentiment_class=1, max_length=128, figsize=(12,4),
                             save_path=None, show=False):
        dev = next(self.parameters()).device
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert = AutoModel.from_pretrained(bert_model_name,
                                          attn_implementation="eager").to(dev).eval()
        tokens, scores, base_prob = self._occlusion_scores(
            text, bert, tokenizer, sentiment_class, max_length)
        del bert
        pred_class = int(base_prob.argmax())
        pred_label = {0:"Negative",1:"Positive"}.get(pred_class, str(pred_class))
        pred_conf  = float(base_prob.max())
        sent_label = {0:"Negative",1:"Positive"}.get(sentiment_class, str(sentiment_class))
        dt, ds = [], []
        for tok, sc in zip(tokens, scores):
            if tok in ("[CLS]","[SEP]","<s>","</s>","[PAD]"): continue
            dt.append(tok.replace("##","").replace("▁","")); ds.append(sc)
        dt = np.array(dt); ds = np.array(ds)
        vmax = max(abs(ds).max(), 1e-6)
        fig = plt.figure(figsize=figsize, facecolor="white")
        gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3,1], hspace=0.45)
        ax_bar = fig.add_subplot(gs[0]); ax_txt = fig.add_subplot(gs[1])
        ax_bar.barh(np.arange(len(dt)), ds,
                    color=[_RWG(0.5+0.5*(s/vmax)) for s in ds],
                    edgecolor="white", linewidth=0.4, height=0.7)
        ax_bar.set_yticks(np.arange(len(dt))); ax_bar.set_yticklabels(dt, fontsize=9, fontfamily="monospace")
        ax_bar.axvline(0, color="#555", linewidth=0.8, linestyle="--"); ax_bar.invert_yaxis()
        ax_bar.spines[["top","right"]].set_visible(False)
        ax_bar.set_xlabel("Attribution Score (Occlusion)", fontsize=9)
        ax_bar.set_title(f"Token Attribution → {sent_label}  |  Pred: {pred_label} ({pred_conf:.1%})",
                         fontsize=10, fontweight="bold")
        badge_c = "#2ca02c" if pred_class==1 else "#d62728"
        ax_bar.text(1.01, 0.5, f"{pred_label}\n{pred_conf:.1%}",
                    transform=ax_bar.transAxes, fontsize=8, va="center", ha="left",
                    color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_c, alpha=0.9))
        ax_txt.set_xlim(0,1); ax_txt.set_ylim(0,1); ax_txt.axis("off")
        x, y = 0.01, 0.75
        for tok, sc in zip(dt, ds):
            w = len(tok)*0.013 + 0.015
            if x+w > 0.98: x, y = 0.01, y-0.45
            if y < 0.05:   break
            intensity = abs(sc)/vmax
            if sc > 0: bg=plt.cm.Greens(0.2+0.6*intensity); fg="#1a5c1a" if intensity>0.5 else "#333"
            else:      bg=plt.cm.Reds(0.2+0.6*intensity);   fg="#7a0c0c" if intensity>0.5 else "#333"
            ax_txt.text(x+w/2, y, tok, ha="center", va="center", fontsize=8.5,
                        color=fg, fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=bg, edgecolor="none", alpha=0.85))
            x += w + 0.005
        ax_txt.set_title("Token Highlight", fontsize=9, color="#555", pad=4)
        fig.suptitle(f'"{text[:90]}{"..." if len(text)>90 else ""}"',
                     fontsize=8, style="italic", color="#666", y=1.01)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {save_path}")
        if show: plt.show()
        else:    plt.close()
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Cluster evolution plot helpers
    # ─────────────────────────────────────────────────────────────────────────
    def plot_cluster_evolution(
        self, embeddings, cluster_assignments, epoch,
        texts=None, save_dir="./results/fnnjst", method="tsne",
        figsize=(6,6), point_size=20, alpha=0.7,
        save_plot=True, show_plot=False,
        plot_tfidf_version=True, max_keywords_per_cluster=3, keyword_min_score=0.3,
    ):
        emb = embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        n   = emb.shape[0]
        if n < 3: return None

        if method.lower() == 'pca':
            from sklearn.decomposition import PCA
            r = PCA(n_components=2, random_state=42); emb_2d = r.fit_transform(emb)
            label = f"PCA ({r.explained_variance_ratio_.sum():.1%} var)"
        elif method.lower() == "tsne":
            from sklearn.manifold import TSNE
            r = TSNE(n_components=2, perplexity=max(2,min(30,n-1)),
                     random_state=42, init="pca", learning_rate="auto")
            emb_2d = r.fit_transform(emb); label = "t-SNE"
        else:
            try:
                import umap
                r = umap.UMAP(n_components=2, random_state=42,
                               n_neighbors=max(2,min(15,n-1)), min_dist=0.1)
                emb_2d = r.fit_transform(emb); label = "UMAP"
            except Exception:
                from sklearn.manifold import TSNE
                r = TSNE(n_components=2, perplexity=max(2,min(30,n-1)),
                         random_state=42, init="pca", learning_rate="auto")
                emb_2d = r.fit_transform(emb); label = "t-SNE (fallback)"

        uniq = np.unique(cluster_assignments); k = len(uniq)
        if k <= 10:   colors = plt.cm.tab10(np.linspace(0,1,10))
        elif k <= 20: colors = plt.cm.tab20(np.linspace(0,1,20))
        else:         colors = plt.cm.hsv(np.linspace(0,1,k))

        fig1, ax1 = plt.subplots(figsize=figsize)
        for i, cid in enumerate(uniq):
            mask = cluster_assignments == cid
            ax1.scatter(emb_2d[mask,0], emb_2d[mask,1], c=[colors[i]],
                        marker="x", s=point_size, alpha=alpha, label=f"C{cid}")
        ax1.set_title(f"Epoch {epoch} ({label})", fontsize=14, fontweight="bold")
        ax1.set_xticks([]); ax1.set_yticks([])
        for sp in ax1.spines.values(): sp.set_visible(False)
        ax1.grid(True, alpha=0.3); ax1.set_facecolor("white")
        plt.tight_layout()
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"cluster_evolution_epoch_{epoch}.png"),
                        dpi=150, bbox_inches="tight", facecolor="white")
        if show_plot: plt.show()
        else:         plt.close()
        return fig1

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

    def fit(
        self,
        dataset,
        # ── Loss weights ────────────────────────────────────────────────────
        alpha: float = 0.1,
        gamma: float = 1.0,
        eta:   float = 0.1,
        # ── Optimizer ───────────────────────────────────────────────────────
        optimizer_type: str   = "adam",
        learning_rate:  float = 1e-3,
        momentum:       float = 0.9,
        # ── Training control ────────────────────────────────────────────────
        tol:             float = 1e-3,
        update_interval: int   = 140,
        batch_size:      int   = 128,
        maxiter:         int   = int(2e4),
        save_dir:        str   = "./results/fnnjst",
        # ── Validation split ─────────────────────────────────────────────────
        val_ratio:       float = 0.1,
        val_metric:      str   = "auto",
        # ── Plot control ─────────────────────────────────────────────────────
        plot_evolution:  bool  = True,
        plot_interval:   Optional[int] = None,
        plot_method:     str   = "tsne",
        compute_metrics: bool  = True,
        # ── Integrated Gradients ─────────────────────────────────────────────
        plot_integrated_gradients: bool = True,
        ig_target_class:  int  = 1,
        ig_n_steps:       int  = 50,
        ig_top_dims:      int  = 20,
        ig_max_samples:   int  = 512,
        # ── Token Attribution ────────────────────────────────────────────────
        plot_token_attribution:    bool = False,
        token_attr_bert_name:      str  = "indolem/indobert-base-uncased",
        token_attr_sentiment_class: int = 1,
        token_attr_top_k:          int  = 10,
        token_attr_max_samples:    int  = 30,
        token_attr_max_length:     int  = 128,
    ):
        """
        Joint DEC + Sentiment + Reconstruction training.

        Metric consistency (v3):
        ─────────────────────────
        Train and validation now expose the SAME metric set:
            Sentiment : Acc, F1, Precision, Recall
            Cluster   : Coherence (NPMI), Diversity (unique top-word ratio)
                        combined into cluster_score = mean(coherence, diversity)

        Silhouette is only used as a geometric fallback when no texts are
        supplied. For text data, coherence + diversity are more meaningful:
            - Coherence  → are the top words in a cluster actually co-occurring?
            - Diversity  → are the clusters covering distinct vocabulary?

        Best-model selection:
        ─────────────────────
        val_f1 (if labels)  |  val_cluster_score (if no labels, text available)
        val_coherence/silhouette (fallback, no texts)
        """
        print("=" * 60)
        print("SEMTGPU v3 — Joint Training: Clustering + Sentiment + Reconstruction")
        print(f"Loss: α(recon)={alpha}, γ(cluster)={gamma}, η(sentiment)={eta}")
        print(f"Update interval: {update_interval}  |  val_ratio: {val_ratio:.0%}")
        print(f"Best-model metric: {val_metric}")
        if plot_token_attribution:
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

        # ── Determine primary val metric ──────────────────────────────────────
        if val_metric == "auto":
            _primary = "val_f1" if has_labels else "val_cluster_score"
        elif val_metric == "f1":
            _primary = "val_f1"
        elif val_metric in ("cluster_score", "silhouette"):
            # Accept legacy "silhouette" keyword, redirect to semantic score
            _primary = "val_cluster_score"
        else:
            raise ValueError(
                f"val_metric must be 'auto'|'f1'|'cluster_score', got {val_metric}"
            )
        print(f"Primary validation metric: {_primary}")

        # ── Class weights on train split ──────────────────────────────────────
        class_w_t = None
        if has_labels:
            y_tr_np = Y_train.cpu().numpy()
            if y_tr_np.ndim == 2 and y_tr_np.shape[1] > 1: y_tr_np = y_tr_np.argmax(1)
            cw = self.compute_class_weights(y_tr_np)
            class_w_t = torch.tensor([cw.get(i, 1.0) for i in range(2)],
                                       dtype=torch.float32, device=dev)

        # ── Optimizer ─────────────────────────────────────────────────────────
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

        # ── KMeans init on train ──────────────────────────────────────────────
        print("Initialising cluster centres with k-means (train split).")
        y_pred_last = self._init_clusters_with_kmeans(X_train)

        # ── Token attribution schedule ────────────────────────────────────────
        n_intervals = maxiter // update_interval
        half_iter   = (n_intervals // 2) * update_interval
        early_iter  = pinterv

        def _should_run_token_attr(ite: int, is_final: bool = False) -> bool:
            if not plot_token_attribution or not has_texts: return False
            if is_final: return True
            return ite in {early_iter, half_iter}

        # ── CSV log ───────────────────────────────────────────────────────────
        log_path = os.path.join(save_dir, "idec_sentiment_log.csv")
        # v3: replaced val_silhouette with val_coherence, val_diversity, val_cluster_score
        #     added train_f1, train_precision, train_recall for consistency
        log_fields = [
            "iter", "split",
            # Train sentiment (now full set, consistent with val)
            "train_acc", "train_f1", "train_precision", "train_recall",
            # Train losses
            "L", "Lr", "Lc", "Ls",
            # Train clustering (supervised)
            "ACC", "NMI", "ARI", "Homogeneity", "Completeness", "V-measure",
            # Train cluster quality (semantic)
            "Train_Coherence", "Train_Diversity", "Train_Cluster_Score",
            "Cluster_Balance", "Min_Cluster_Size", "Max_Cluster_Size",
            # Val sentiment
            "val_acc", "val_f1", "val_precision", "val_recall",
            # Val clustering (supervised)
            "val_nmi", "val_ari", "val_acc_cluster",
            # Val cluster quality (semantic)
            "val_coherence", "val_diversity", "val_cluster_score",
            # Best-model
            "val_primary_score", "is_best",
            # IG
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

                # ── Evaluation + target distribution ─────────────────────────
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

                    # ── Train sentiment metrics (full set, consistent with val) ─
                    train_acc = train_f1 = train_prec = train_rec = 0.0
                    if has_labels:
                        s_lab  = s_all.argmax(1).cpu().numpy()
                        y_true = Y_train.cpu().numpy()
                        if y_true.ndim == 2 and y_true.shape[1] > 1:
                            y_true = y_true.argmax(1)
                        train_acc  = float((s_lab == y_true).mean())
                        train_f1   = float(f1_score(y_true, s_lab,
                                                     average="binary", zero_division=0))
                        train_prec = float(precision_score(y_true, s_lab,
                                                           average="binary", zero_division=0))
                        train_rec  = float(recall_score(y_true, s_lab,
                                                        average="binary", zero_division=0))

                    # ── Train clustering metrics (supervised) ─────────────────
                    feats_tr = self.extract_feature(X_train).cpu().numpy()
                    cl_sup_tr = {}
                    if has_labels:
                        yt = Y_train.cpu().numpy()
                        if yt.ndim == 2 and yt.shape[1] > 1: yt = yt.argmax(1)
                        cl_sup_tr = self.compute_clustering_metrics(yt, y_pred)

                    # ── Train cluster quality (semantic) ──────────────────────
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

                    # ── Validation evaluation ─────────────────────────────────
                    val_metrics = self._evaluate_val(X_val, Y_val, texts_val, batch_size)
                    vs      = val_metrics["val_primary_score"]
                    is_best = vs > self._best_val_score
                    if is_best:
                        self._best_val_score = vs
                        self._best_val_iter  = ite
                        self._best_state_dict = copy.deepcopy(self.state_dict())
                        print(f"  ✓ New best model at iter {ite}: {_primary}={vs:.4f}")
                        self.save_weights(os.path.join(save_dir, "SEMTGPU_best.weights.pth"))

                    # ── Aligned print (train vs val, same metric set) ─────────
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

                    # ── Scatter plot ──────────────────────────────────────────
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

                    # ── Integrated Gradients plot ─────────────────────────────
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

                    # ── Token Attribution (scheduled: early / half only) ───────
                    if _should_run_token_attr(ite, is_final=False):
                        try:
                            print(f"  [Token Attr @ iter {ite}]")
                            ta = self.compute_token_attribution_per_cluster(
                                texts_train, y_pred,
                                bert_model_name=token_attr_bert_name,
                                sentiment_class=token_attr_sentiment_class,
                                top_k=token_attr_top_k,
                                max_length=token_attr_max_length,
                                max_samples_per_cluster=token_attr_max_samples,
                            )
                            self.plot_token_attribution_per_cluster(
                                ta, sentiment_class=token_attr_sentiment_class,
                                epoch=ite, save_dir=plot_dir,
                                top_k=token_attr_top_k, show_plot=False)
                            self.plot_token_attribution_heatmap(
                                ta, sentiment_class=token_attr_sentiment_class,
                                epoch=ite, save_dir=plot_dir, show_plot=False)
                        except Exception as e:
                            print(f"  Warning: token attr failed at iter {ite}: {e}")

                    # ── CSV row ───────────────────────────────────────────────
                    writer.writerow({
                        "iter":   ite,
                        "split":  "train",
                        # Train sentiment
                        "train_acc":       round(train_acc,  5),
                        "train_f1":        round(train_f1,   5),
                        "train_precision": round(train_prec, 5),
                        "train_recall":    round(train_rec,  5),
                        # Losses
                        "L":  round(avg_L,  5),
                        "Lr": round(avg_Lr, 5),
                        "Lc": round(avg_Lc, 5),
                        "Ls": round(avg_Ls, 5),
                        # Train clustering supervised
                        "ACC":         round(cl_sup_tr.get('ACC',         0.0), 5),
                        "NMI":         round(cl_sup_tr.get('NMI',         0.0), 5),
                        "ARI":         round(cl_sup_tr.get('ARI',         0.0), 5),
                        "Homogeneity": round(cl_sup_tr.get('Homogeneity', 0.0), 5),
                        "Completeness":round(cl_sup_tr.get('Completeness',0.0), 5),
                        "V-measure":   round(cl_sup_tr.get('V-measure',   0.0), 5),
                        # Train cluster quality (semantic)
                        "Train_Coherence":    round(train_coh,           5),
                        "Train_Diversity":    round(train_div,           5),
                        "Train_Cluster_Score":round(train_cluster_score, 5),
                        "Cluster_Balance":    round(cov_tr['cluster_balance'],   5),
                        "Min_Cluster_Size":   round(cov_tr['min_cluster_size'],  5),
                        "Max_Cluster_Size":   round(cov_tr['max_cluster_size'],  5),
                        # Val sentiment
                        "val_acc":       round(val_metrics.get('val_acc_sentiment', 0.0), 5),
                        "val_f1":        round(val_metrics.get('val_f1',            0.0), 5),
                        "val_precision": round(val_metrics.get('val_precision',     0.0), 5),
                        "val_recall":    round(val_metrics.get('val_recall',        0.0), 5),
                        # Val clustering supervised
                        "val_nmi":         round(val_metrics.get('val_nmi',         0.0), 5),
                        "val_ari":         round(val_metrics.get('val_ari',         0.0), 5),
                        "val_acc_cluster": round(val_metrics.get('val_acc_cluster', 0.0), 5),
                        # Val cluster quality (semantic)
                        "val_coherence":     round(val_metrics.get('val_coherence',     0.0), 5),
                        "val_diversity":     round(val_metrics.get('val_diversity',     0.0), 5),
                        "val_cluster_score": round(val_metrics.get('val_cluster_score', 0.0), 5),
                        # Best-model
                        "val_primary_score": round(val_metrics.get('val_primary_score', 0.0), 5),
                        "is_best":           int(is_best),
                        # IG
                        "IG_TopDim":           ig_top_dim_idx,
                        "IG_TopDim_Score":     round(ig_top_dim_score, 6),
                        "IG_Mean_Attribution": round(ig_mean_attr,     6),
                    })

                    tot_L = Lr = Lc = Ls = 0.0
                    iter_count = 0

                    if ite > 0 and delta < tol:
                        print(f"  Δlabel={delta:.6f} < tol={tol} → early stop.")
                        break

                    # Rebuild train loader with updated P
                    if has_labels:
                        train_loader = DataLoader(
                            TensorDataset(X_train, p_all, Y_train),
                            batch_size=batch_size, shuffle=True)
                    else:
                        train_loader = DataLoader(
                            TensorDataset(X_train, p_all),
                            batch_size=batch_size, shuffle=True)
                    self.train()

                # ── Train step ────────────────────────────────────────────────
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

        # ── Final cluster assignments ─────────────────────────────────────────
        self.eval()
        with torch.no_grad():
            q_all_f = torch.cat([self(X_train[i:i+batch_size])[0]
                                  for i in range(0, n_train, batch_size)], 0)
        y_final = q_all_f.argmax(1).cpu().numpy()

        # ── Token attribution at LAST epoch ───────────────────────────────────
        if _should_run_token_attr(last_ite, is_final=True) and has_texts:
            try:
                print("\n[Token Attr @ LAST epoch]")
                ta_last = self.compute_token_attribution_per_cluster(
                    texts_train, y_final,
                    bert_model_name=token_attr_bert_name,
                    sentiment_class=token_attr_sentiment_class,
                    top_k=token_attr_top_k,
                    max_length=token_attr_max_length,
                    max_samples_per_cluster=token_attr_max_samples,
                )
                ta_dir = os.path.join(save_dir, "token_attribution")
                self.plot_token_attribution_per_cluster(
                    ta_last, sentiment_class=token_attr_sentiment_class,
                    epoch=last_ite, save_dir=ta_dir,
                    top_k=token_attr_top_k, show_plot=False)
                self.plot_token_attribution_heatmap(
                    ta_last, sentiment_class=token_attr_sentiment_class,
                    epoch=last_ite, save_dir=ta_dir, show_plot=False)
                print("✓ Final token attribution complete.")
            except Exception as e:
                print(f"Warning: final token attribution failed: {e}")

        # ── Final IG ──────────────────────────────────────────────────────────
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

        # ── Final metrics ──────────────────────────────────────────────────────
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

        # ── Final val metrics on best model ───────────────────────────────────
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