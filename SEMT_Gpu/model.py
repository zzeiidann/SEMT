from __future__ import annotations

import os
import csv
import glob
import re
import warnings
from collections import Counter, defaultdict
from typing import Iterable, List, Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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


# --------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# ── Custom colormaps for token attribution ──────────────────────────────────
_RWG = LinearSegmentedColormap.from_list(
    "rwg", ["#d62728", "#f7f7f7", "#2ca02c"], N=256
)
warnings.filterwarnings("ignore", category=UserWarning)
# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------
def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return float(sum(w[i, j] for i, j in zip(row_ind, col_ind)) / y_pred.size)


# --------------------------------------------------------------------------------------
# Model Components
# --------------------------------------------------------------------------------------
class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters: int, input_dim: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.empty(n_clusters, input_dim))
        self._init_weights()

    def _init_weights(self) -> None:
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
        assert len(dims) >= 2, "dims minimal [input_dim, latent]"
        self.dims = list(dims)

        act_map = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}
        self.activation = act_map.get(act, nn.ReLU())

        enc = []
        for i in range(len(dims)-2):
            enc += [nn.Linear(dims[i], dims[i+1]), self.activation]
        enc += [nn.Linear(dims[-2], dims[-1])]
        self.encoder = nn.Sequential(*enc)

        dec = []
        for j in range(len(dims)-1, 0, -1):
            dec += [nn.Linear(dims[j], dims[j-1])]
            if j != 1:
                dec += [self.activation]
        self.decoder = nn.Sequential(*dec)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, h):
        return self.decoder(h)

    def forward(self, x):
        h = self.encode(x)
        return h, self.decode(h)


# --------------------------------------------------------------------------------------
# SEMTGPU (Main)
# --------------------------------------------------------------------------------------
class SEMTGPU(nn.Module):
    """
    Joint Sentiment + Topic Clustering (DEC-style) with Autoencoder features.

    Loss weights:
        alpha: reconstruction loss weight
        gamma: clustering loss weight
        eta: sentiment loss weight
    """

    def __init__(self, dims: List[int], n_clusters: int = 10, alpha_clustering: float = 1.0) -> None:
        super().__init__()

        assert len(dims) >= 2, "dims must be [input_dim, ..., latent_dim]"
        self.dims = dims
        self.n_clusters = int(n_clusters)
        self.alpha_clustering = float(alpha_clustering)

        self.autoencoder = Autoencoder(dims)
        self.clustering = ClusteringLayer(n_clusters=n_clusters, input_dim=dims[-1], alpha=alpha_clustering)

        self.sentiment = nn.Sequential(
            nn.Linear(dims[-1], 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2),
        )

        self.class_labels: Dict[int, str] = {0: "negative", 1: "positive"}
        self.topic_mapping: Dict[int, str] = {}
        self.stop_words: set[str] = set()

        self._init_head()

    def _init_head(self) -> None:
        for m in self.sentiment.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.autoencoder.encode(x)
        q = self.clustering(z)
        s = torch.softmax(self.sentiment(z), dim=1)
        return q, s

    def extract_feature(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            z = self.autoencoder.encode(xt)
        return z

    def load_weights(self, path: str) -> None:
        loc = {"cuda:0": "cpu"} if not torch.cuda.is_available() else None
        state = torch.load(path, map_location=loc)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.load_state_dict(state["model_state_dict"])
        else:
            self.load_state_dict(state)
        print(f"✓ Loaded weights from {path}")

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()}, path)

    def pretrain_autoencoder(
        self,
        dataset: Iterable,
        batch_size: int = 256,
        epochs: int = 200,
        lr: float = 1e-3,
        save_dir: str = "./results/ae",
        weights_name: str = "pretrained_ae.weights.pth",
    ) -> str:
        print("=" * 60)
        print("Pretraining Autoencoder")
        print("=" * 60)

        save_path = Path(save_dir or "./results/ae")
        save_path.mkdir(parents=True, exist_ok=True)
        weights_path = save_path / weights_name

        embs: List[torch.Tensor] = []
        for i in range(len(dataset)):
            item = dataset[i]
            emb = item[0] if (isinstance(item, tuple) and len(item) >= 1) else item
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            else:
                emb = emb.detach()
            embs.append(emb.cpu())

        X = torch.stack(embs)
        n, d = X.shape

        if d != self.dims[0]:
            raise ValueError(
                f"Input dim mismatch: embeddings dim={d}, tapi dims[0]={self.dims[0]}."
            )

        dev = next(self.parameters()).device
        X = X.to(dev)
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

        self.autoencoder.to(dev).train()
        opt = optim.Adam(self.autoencoder.parameters(), lr=lr)
        crit = nn.MSELoss()

        for ep in range(epochs):
            total = 0.0
            with tqdm(loader, desc=f"AE Epoch {ep+1}/{epochs}") as pbar:
                for (xb,) in pbar:
                    _, rec = self.autoencoder(xb)
                    loss = crit(rec, xb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += float(loss.item())
                    pbar.set_postfix({"mse": total / (pbar.n + 1)})

        torch.save(
            {"autoencoder_state_dict": self.autoencoder.state_dict(), "dims": self.dims},
            str(weights_path),
        )
        print(f"✓ Autoencoder pretraining complete. Saved to: {weights_path}")
        return str(weights_path)

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        weight = (q ** 2) / torch.clamp(torch.sum(q, dim=0), min=1e-12)
        p = (weight.t() / torch.clamp(torch.sum(weight, dim=1), min=1e-12)).t()
        return p

    def compute_class_weights(self, y: Union[np.ndarray, torch.Tensor]) -> Dict[int, float]:
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if y.ndim == 2 and y.shape[1] > 1:
            y = y.argmax(axis=1)
        cls, cnt = np.unique(y, return_counts=True)
        total = len(y)
        k = len(cls)
        weights: Dict[int, float] = {int(c): total / (k * int(n)) for c, n in zip(cls, cnt)}
        return weights

    def predict_clusters(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            q, _ = self(xt)
            return q.argmax(dim=1).cpu().numpy()

    def predict_sentiment(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            _, s = self(xt)
            return s.argmax(dim=1).cpu().numpy()

    def predict(
        self,
        inputs: Union[str, List[str], np.ndarray, torch.Tensor],
        bert_model: Optional[Union[str, AutoModel]] = None,
    ) -> List[Dict[str, Union[str, int]]]:
        self.eval()
        dev = next(self.parameters()).device

        if isinstance(inputs, str) or (isinstance(inputs, list) and inputs and isinstance(inputs[0], str)):
            texts = [inputs] if isinstance(inputs, str) else inputs
            model_name = bert_model if isinstance(bert_model, str) else "indolem/indobert-base-uncased"
            tok = AutoTokenizer.from_pretrained(model_name)
            toks = tok(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(dev)
            with torch.no_grad():
                if isinstance(bert_model, AutoModel):
                    bert = bert_model.to(dev)
                else:
                    bert = AutoModel.from_pretrained(model_name).to(dev)
                out = bert(**toks)
            X = out.last_hidden_state[:, 0, :]
        else:
            X = torch.as_tensor(inputs, dtype=torch.float32, device=dev)

        with torch.no_grad():
            q, s = self(X)

        c = q.argmax(dim=1).cpu().numpy()
        y = s.argmax(dim=1).cpu().numpy()

        results: List[Dict[str, Union[str, int]]] = []
        for i in range(len(y)):
            cid = int(c[i])
            topic = self.topic_mapping.get(cid, cid)
            results.append({"sentiment": self.class_labels[int(y[i])], "topic": topic})
        return results

    # -------------------------
    # INTERPRETABILITY METRICS
    # -------------------------

    def extract_tfidf_keywords(
        self,
        texts: List[str],
        cluster_assignments: np.ndarray,
        top_n: int = 10,
        max_features: int = 5000,
    ) -> Dict[int, List[Tuple[str, float]]]:
        clusters: Dict[int, List[str]] = {}
        for i, cid in enumerate(cluster_assignments):
            clusters.setdefault(int(cid), []).append(texts[i])

        cluster_keywords = {}

        for cid, cluster_texts in clusters.items():
            if not cluster_texts:
                cluster_keywords[cid] = []
                continue

            all_cluster_docs = [" ".join(txts) for txts in clusters.values()]

            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words=list(self.stop_words) if self.stop_words else None,
                min_df=1,
                ngram_range=(1, 2),
            )

            try:
                tfidf_matrix = vectorizer.fit_transform(all_cluster_docs)
                feature_names = vectorizer.get_feature_names_out()

                cluster_idx = list(clusters.keys()).index(cid)
                scores = tfidf_matrix[cluster_idx].toarray()[0]

                top_indices = scores.argsort()[-top_n:][::-1]
                keywords = [(feature_names[i], float(scores[i])) for i in top_indices]
                cluster_keywords[cid] = keywords

            except Exception as e:
                print(f"Warning: TF-IDF failed for cluster {cid}: {e}")
                cluster_doc = " ".join(cluster_texts)
                words = cluster_doc.lower().split()
                words = [w for w in words if w not in self.stop_words and len(w) > 2]
                word_counts = Counter(words)
                cluster_keywords[cid] = [(w, float(c)) for w, c in word_counts.most_common(top_n)]

        return cluster_keywords

    def compute_clustering_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            return {
                'ACC': 0.0, 'NMI': 0.0, 'ARI': 0.0,
                'Homogeneity': 0.0, 'Completeness': 0.0, 'V-measure': 0.0,
                'Topic_Coverage': 0.0,
            }

        unique_clusters, cluster_counts = np.unique(y_pred, return_counts=True)
        cluster_probs = cluster_counts / len(y_pred)
        from scipy.stats import entropy
        topic_coverage = 1.0 - (entropy(cluster_probs) / np.log(len(unique_clusters))) if len(unique_clusters) > 1 else 0.0

        return {
            'ACC': cluster_acc(y_true, y_pred),
            'NMI': float(normalized_mutual_info_score(y_true, y_pred)),
            'ARI': float(adjusted_rand_score(y_true, y_pred)),
            'Homogeneity': float(homogeneity_score(y_true, y_pred)),
            'Completeness': float(completeness_score(y_true, y_pred)),
            'V-measure': float(v_measure_score(y_true, y_pred)),
            'Topic_Coverage': float(topic_coverage),
        }

    def compute_variance_explained(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        n_components: int = 2,
    ) -> Dict[str, float]:
        if isinstance(embeddings, torch.Tensor):
            emb = embeddings.detach().cpu().numpy()
        else:
            emb = embeddings

        pca = PCA(n_components=min(n_components, emb.shape[0], emb.shape[1]))
        pca.fit(emb)

        return {
            'variance_explained': float(pca.explained_variance_ratio_[:n_components].sum()),
            'component_variances': pca.explained_variance_ratio_[:n_components].tolist(),
        }

    def select_best_visualization_method(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        methods: List[str] = ['pca', 'tsne', 'umap'],
        variance_threshold: float = 0.7,
    ) -> Tuple[str, Dict[str, float]]:
        if isinstance(embeddings, torch.Tensor):
            emb = embeddings.detach().cpu().numpy()
        else:
            emb = embeddings

        n = emb.shape[0]
        variance_info = self.compute_variance_explained(emb, n_components=2)
        var_explained = variance_info['variance_explained']
        metrics = {'pca_variance_2d': var_explained, 'n_samples': n}

        if var_explained >= variance_threshold:
            print(f"✓ PCA explains {var_explained:.2%} variance → using PCA for visualization")
            return 'pca', metrics

        if 'umap' in methods and n > 1000:
            print(f"✓ Large dataset (n={n}) with PCA variance {var_explained:.2%} → using UMAP")
            return 'umap', metrics
        elif 'tsne' in methods:
            print(f"✓ Medium dataset (n={n}) with PCA variance {var_explained:.2%} → using t-SNE")
            return 'tsne', metrics
        else:
            print(f"✓ Fallback to PCA (variance: {var_explained:.2%})")
            return 'pca', metrics

    def compute_silhouette_score(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        cluster_assignments: np.ndarray,
    ) -> float:
        if isinstance(embeddings, torch.Tensor):
            emb = embeddings.detach().cpu().numpy()
        else:
            emb = embeddings

        if len(np.unique(cluster_assignments)) < 2:
            return 0.0

        try:
            score = silhouette_score(emb, cluster_assignments)
            return float(score)
        except Exception as e:
            print(f"Warning: silhouette score computation failed: {e}")
            return 0.0

    def compute_topic_coherence(
        self,
        texts: List[str],
        cluster_assignments: np.ndarray,
        top_n: int = 10,
    ) -> Dict[int, float]:
        from itertools import combinations

        clusters: Dict[int, List[str]] = {}
        for i, cid in enumerate(cluster_assignments):
            clusters.setdefault(int(cid), []).append(texts[i])

        coherence_scores = {}

        for cid, cluster_texts in clusters.items():
            words = " ".join(cluster_texts).lower().split()
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            word_counts = Counter(words)
            top_words = [w for w, _ in word_counts.most_common(top_n)]

            if len(top_words) < 2:
                coherence_scores[cid] = 0.0
                continue

            doc_freq = {}
            for word in top_words:
                doc_freq[word] = sum(1 for text in cluster_texts if word in text.lower())

            npmi_scores = []
            for w1, w2 in combinations(top_words, 2):
                co_occur = sum(1 for text in cluster_texts if w1 in text.lower() and w2 in text.lower())
                if co_occur > 0 and doc_freq[w1] > 0 and doc_freq[w2] > 0:
                    p_w1_w2 = co_occur / len(cluster_texts)
                    p_w1 = doc_freq[w1] / len(cluster_texts)
                    p_w2 = doc_freq[w2] / len(cluster_texts)
                    pmi = np.log((p_w1_w2 + 1e-10) / (p_w1 * p_w2 + 1e-10))
                    npmi = pmi / (-np.log(p_w1_w2 + 1e-10))
                    npmi_scores.append(npmi)

            coherence_scores[cid] = float(np.mean(npmi_scores)) if npmi_scores else 0.0

        return coherence_scores

    def compute_topic_diversity(
        self,
        texts: List[str],
        cluster_assignments: np.ndarray,
        top_n: int = 10,
    ) -> float:
        clusters: Dict[int, List[str]] = {}
        for i, cid in enumerate(cluster_assignments):
            clusters.setdefault(int(cid), []).append(texts[i])

        all_top_words = []
        for cid, cluster_texts in clusters.items():
            words = " ".join(cluster_texts).lower().split()
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            word_counts = Counter(words)
            top_words = set([w for w, _ in word_counts.most_common(top_n)])
            all_top_words.append(top_words)

        if len(all_top_words) < 2:
            return 0.0

        unique_words = set()
        total_words = 0
        for word_set in all_top_words:
            unique_words.update(word_set)
            total_words += len(word_set)

        diversity = len(unique_words) / total_words if total_words > 0 else 0.0
        return float(diversity)

    def compute_topic_coverage(
        self,
        texts: List[str],
        cluster_assignments: np.ndarray,
    ) -> Dict[str, float]:
        cluster_counts = Counter(cluster_assignments)
        total = len(cluster_assignments)
        sizes = [count / total for count in cluster_counts.values()]
        entropy_val = -sum(p * np.log(p + 1e-10) for p in sizes)
        max_entropy = np.log(len(cluster_counts))
        balance = entropy_val / max_entropy if max_entropy > 0 else 0.0

        return {
            'cluster_balance': float(balance),
            'min_cluster_size': float(min(sizes)),
            'max_cluster_size': float(max(sizes)),
            'n_clusters': len(cluster_counts),
        }

    def compute_comprehensive_metrics(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        texts: List[str],
        cluster_assignments: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        top_n_words: int = 10,
    ) -> Dict[str, any]:
        print("\n" + "=" * 60)
        print("Computing Comprehensive Interpretability Metrics")
        print("=" * 60)

        clustering_metrics = {}
        if true_labels is not None:
            clustering_metrics = self.compute_clustering_metrics(true_labels, cluster_assignments)

        best_method, viz_metrics = self.select_best_visualization_method(embeddings)
        silhouette = self.compute_silhouette_score(embeddings, cluster_assignments)
        coherence = self.compute_topic_coherence(texts, cluster_assignments, top_n=top_n_words)
        diversity = self.compute_topic_diversity(texts, cluster_assignments, top_n=top_n_words)
        coverage = self.compute_topic_coverage(texts, cluster_assignments)
        tfidf_keywords = self.extract_tfidf_keywords(texts, cluster_assignments, top_n=top_n_words)

        metrics = {
            'clustering_supervised': clustering_metrics,
            'visualization': {'best_method': best_method, **viz_metrics},
            'clustering_unsupervised': {'silhouette_score': silhouette},
            'topic_coherence': coherence,
            'topic_coherence_mean': float(np.mean(list(coherence.values()))) if coherence else 0.0,
            'topic_diversity': diversity,
            'topic_coverage': coverage,
            'tfidf_keywords': tfidf_keywords,
        }

        print(f"\n Visualization Method: {best_method.upper()}")
        print(f"   PCA Variance (2D): {viz_metrics['pca_variance_2d']:.2%}")

        if clustering_metrics:
            print(f"\n Clustering Quality (Supervised):")
            print(f"   ACC:           {clustering_metrics['ACC']:.4f}")
            print(f"   NMI:           {clustering_metrics['NMI']:.4f}")
            print(f"   ARI:           {clustering_metrics['ARI']:.4f}")
            print(f"   V-measure:     {clustering_metrics['V-measure']:.4f}")

        print(f"\n Clustering Quality (Unsupervised):")
        print(f"   Silhouette:    {silhouette:.4f}")
        print(f"   Topic_Coverage:{clustering_metrics.get('Topic_Coverage', 0.0):.4f}")

        print(f"\n Topic Quality:")
        print(f"   Mean Coherence: {metrics['topic_coherence_mean']:.4f}")
        print(f"   Topic Diversity: {diversity:.4f}")
        print(f"   Cluster Balance: {coverage['cluster_balance']:.4f}")
        print(f"   Min/Max Cluster: {coverage['min_cluster_size']:.1%} / {coverage['max_cluster_size']:.1%}")

        print(f"\n Top TF-IDF Keywords per Cluster:")
        for cid, keywords in tfidf_keywords.items():
            top_5 = ", ".join([f"{w}({s:.3f})" for w, s in keywords[:5]])
            print(f"   Cluster {cid}: {top_5}")

        print("=" * 60 + "\n")
        return metrics

    # =========================================================
    # 🆕 INTEGRATED GRADIENTS
    # =========================================================

    def compute_integrated_gradients(
        self,
        x: Union[np.ndarray, torch.Tensor],
        target_class: int = 1,
        n_steps: int = 50,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients for the sentiment head.

        Shows WHICH embedding dimensions most influence the sentiment prediction.
        Uses the straight-line path from a zero baseline to the actual input.

        Reference:
            Sundararajan et al. (2017). "Axiomatic Attribution for Deep Networks". ICML.
            Formula: IG_i(x) = (x_i - x'_i) * ∫₀¹ [∂F/∂x_i (x' + α(x-x')) dα]

        Why zero baseline?
            The zero vector is the standard baseline for embedding spaces — it represents
            "no information", analogous to a neutral/absent signal.

        Args:
            x:            Input embeddings, shape (N, D). Usually BERT [CLS] embeddings.
            target_class: Sentiment class to explain. 0=negative, 1=positive.
            n_steps:      Number of integration steps (Riemann approx). Default 50 is
                          sufficient for most cases; use 100+ for publication-quality.
            batch_size:   Batch size for gradient computation (reduce if OOM).

        Returns:
            attributions: np.ndarray, shape (N, D).
                          Positive values → dimension pushes toward target_class.
                          Negative values → dimension pushes away from target_class.
        """
        dev = next(self.parameters()).device
        xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
        N, D = xt.shape

        # Baseline: zero vector (standard for embedding inputs)
        baseline = torch.zeros_like(xt)

        all_attributions = []
        # Switch to train mode to allow gradient flow through BatchNorm
        was_training = self.training
        self.train()

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_batch = xt[start:end]          # (b, D)
            base_batch = baseline[start:end]  # (b, D)
            b = x_batch.shape[0]

            # Build interpolated inputs: x' + α*(x - x'), for α ∈ [0, 1]
            # Shape: (n_steps, b, D)
            alphas = torch.linspace(0, 1, n_steps, device=dev)
            interp = base_batch.unsqueeze(0) + alphas.view(-1, 1, 1) * (
                x_batch - base_batch
            ).unsqueeze(0)
            interp = interp.view(n_steps * b, D).requires_grad_(True)

            # Forward: input → encoder → sentiment head → target logit
            z_interp = self.autoencoder.encode(interp)
            s_interp = self.sentiment(z_interp)       # (n_steps*b, 2)
            logits = s_interp[:, target_class]         # (n_steps*b,)

            # Backprop only to interp (not model weights)
            grads = torch.autograd.grad(
                outputs=logits.sum(),
                inputs=interp,
                create_graph=False,
                retain_graph=False,
            )[0]  # (n_steps*b, D)

            # Reshape and average with trapezoidal rule: (grads[0]+grads[-1])/2 + inner
            grads = grads.view(n_steps, b, D)
            avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2.0  # (b, D)

            # Final IG: (x - baseline) * avg_grads
            ig = (x_batch - base_batch) * avg_grads.detach()  # (b, D)
            all_attributions.append(ig.detach().cpu().numpy())

        # Restore original mode
        if not was_training:
            self.eval()

        return np.concatenate(all_attributions, axis=0)  # (N, D)

    def plot_integrated_gradients(
        self,
        attributions: np.ndarray,
        cluster_assignments: np.ndarray,
        epoch: int,
        save_dir: str = "./results/fnnjst",
        figsize: Tuple[int, int] = (14, 6),
        top_dims: int = 20,
        save_plot: bool = True,
        show_plot: bool = False,
    ) -> plt.Figure:
        """
        Visualize Integrated Gradients as two panels:

        Panel A — Global bar chart:
            Mean |IG| per embedding dimension across ALL samples.
            Tells you "globally, which BERT embedding dimensions drive sentiment?"

        Panel B — Per-cluster heatmap:
            Mean |IG| for the top dimensions, broken down by cluster.
            Tells you "does each topic rely on different dimensions for sentiment?"

        Args:
            attributions:        Output of compute_integrated_gradients(), shape (N, D).
            cluster_assignments: Cluster labels, shape (N,).
            epoch:               Current training epoch (for filename & title).
            save_dir:            Directory to save plot.
            top_dims:            How many top embedding dimensions to display.
            save_plot:           Whether to save figure to disk.
            show_plot:           Whether to call plt.show() (use False in training loops).

        Returns:
            matplotlib Figure object.

        Output file:
            <save_dir>/integrated_gradients_epoch_<epoch>.png
        """
        N, D = attributions.shape
        abs_attr = np.abs(attributions)   # (N, D) — unsigned importance

        # --- Global mean |IG| per dimension ---
        global_importance = abs_attr.mean(axis=0)                        # (D,)
        top_idx = np.argsort(global_importance)[-top_dims:][::-1]        # descending

        # --- Per-cluster mean |IG| for the top dims ---
        uniq = np.unique(cluster_assignments)
        cluster_ig = np.zeros((len(uniq), top_dims))
        for i, cid in enumerate(uniq):
            mask = cluster_assignments == cid
            if mask.sum() > 0:
                cluster_ig[i] = abs_attr[mask][:, top_idx].mean(axis=0)

        # ---- Figure layout ----
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={'width_ratios': [1, 1.6]}
        )
        fig.suptitle(
            f"Integrated Gradients — Epoch {epoch}\n"
            f"(Sentiment attribution over BERT embedding dimensions)",
            fontsize=13, fontweight='bold', y=1.02
        )

        # ── Panel A: Global bar chart ──────────────────────────────────
        norm_importance = global_importance[top_idx]
        norm_importance = norm_importance / (norm_importance.max() + 1e-8)
        colors_bar = plt.cm.RdYlGn(norm_importance)

        bars = ax1.barh(
            y=[f"dim {idx}" for idx in top_idx],
            width=global_importance[top_idx],
            color=colors_bar,
            edgecolor='white',
            linewidth=0.5,
        )
        ax1.set_xlabel("Mean |IG| Attribution", fontsize=10)
        ax1.set_title(
            "Top Embedding Dimensions\n(Global Sentiment Impact)",
            fontsize=11
        )
        ax1.invert_yaxis()
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')

        # Value labels on bars
        for bar, val in zip(bars, global_importance[top_idx]):
            ax1.text(
                val + global_importance[top_idx].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=7, color='#444'
            )

        # ── Panel B: Per-cluster heatmap ──────────────────────────────
        vmax = cluster_ig.max() if cluster_ig.max() > 0 else 1.0
        im = ax2.imshow(
            cluster_ig, aspect='auto', cmap='YlOrRd',
            vmin=0, vmax=vmax, interpolation='nearest'
        )
        ax2.set_xticks(range(top_dims))
        ax2.set_xticklabels(
            [f"d{idx}" for idx in top_idx],
            rotation=90, fontsize=7
        )
        ax2.set_yticks(range(len(uniq)))
        ax2.set_yticklabels(
            [f"C{c} ({(cluster_assignments == c).sum()})" for c in uniq],
            fontsize=9
        )
        ax2.set_title(
            "Per-Cluster Attribution Heatmap\n"
            "(rows = clusters, cols = top dims, values = mean |IG|)",
            fontsize=11
        )
        ax2.set_xlabel("Embedding Dimension (index)", fontsize=10)

        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Mean |IG|", fontsize=9)

        # Cell annotations (only if grid is small enough to read)
        if top_dims <= 30 and len(uniq) <= 20:
            for i in range(len(uniq)):
                for j in range(top_dims):
                    val = cluster_ig[i, j]
                    txt_color = 'white' if val > vmax * 0.6 else '#333'
                    ax2.text(
                        j, i, f'{val:.3f}',
                        ha='center', va='center',
                        fontsize=6, color=txt_color
                    )

        plt.tight_layout()

        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"integrated_gradients_epoch_{epoch}.png")
            plt.savefig(out, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"  ✓ IG plot saved: {out}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    # -------------------------
    # Training (fit)
    # -------------------------
    def fit(
        self,
        dataset: Iterable,
        alpha: float = 0.1,
        gamma: float = 1.0,
        eta: float = 0.1,
        optimizer_type: str = "adam",
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        tol: float = 1e-3,
        update_interval: int = 140,
        batch_size: int = 128,
        maxiter: int = int(2e4),
        save_dir: str = "./results/fnnjst",
        plot_evolution: bool = True,
        plot_interval: Optional[int] = None,
        plot_method: str = "tsne",
        compute_metrics: bool = True,
        # ── Integrated Gradients options ──────────────────────────
        plot_integrated_gradients: bool = True,
        ig_target_class: int = 1,
        ig_n_steps: int = 50,
        ig_top_dims: int = 20,
        ig_max_samples: int = 512,
        # ── 🆕 Token Attribution per Cluster ─────────────────────
        plot_token_attribution: bool = False,
        token_attr_bert_name: str = "indolem/indobert-base-uncased",
        token_attr_sentiment_class: int = 1,
        token_attr_top_k: int = 10,
        token_attr_max_samples: int = 30,
        token_attr_max_length: int = 128,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
        """
        Joint training (DEC + Sentiment + Reconstruction).

        Loss = alpha * L_reconstruction + gamma * L_clustering + eta * L_sentiment

        New parameters (Integrated Gradients):
            plot_integrated_gradients: Generate IG plots every plot_interval iterations.
            ig_target_class:           Sentiment class to attribute (0=neg, 1=pos).
            ig_n_steps:                IG integration steps (50 is fine, 100 for papers).
            ig_top_dims:               Top-K embedding dims to display in IG plot.
            ig_max_samples:            Max samples used for IG (subsample for speed).
                                       Full dataset can be slow; 512 is usually enough.
        """
        print("=" * 60)
        print("Joint Training: Clustering + Sentiment + Reconstruction")
        print(f"Loss weights — Alpha (recon): {alpha}, Gamma (cluster): {gamma}, Eta (sentiment): {eta}")
        print(f"Update interval: {update_interval}")
        if plot_integrated_gradients:
            print(f"IG: target_class={ig_target_class}, n_steps={ig_n_steps}, "
                  f"top_dims={ig_top_dims}, max_samples={ig_max_samples}")
        print("=" * 60)

        dev = next(self.parameters()).device
        maxiter = int(maxiter)
        plot_interval = int(plot_interval) if plot_interval is not None else update_interval

        os.makedirs(save_dir, exist_ok=True)
        plot_dir = None
        if plot_evolution or plot_integrated_gradients:
            plot_dir = os.path.join(save_dir, "evolution_plots")
            os.makedirs(plot_dir, exist_ok=True)

        # Collect data
        embs: List[torch.Tensor] = []
        lbls: List[torch.Tensor] = []
        texts_list: List[str] = []

        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, tuple):
                if len(item) == 2:
                    embs.append(item[0].detach().cpu())
                    lbls.append(item[1].detach().cpu())
                elif len(item) == 3:
                    embs.append(item[0].detach().cpu())
                    lbls.append(item[1].detach().cpu())
                    texts_list.append(item[2])
            else:
                t = item.detach() if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.float32)
                embs.append(t.cpu())

        X = torch.stack(embs).to(dev)
        N, D = X.shape
        if D != self.dims[0]:
            raise ValueError(f"Input dim = {D}, but dims[0] = {self.dims[0]}.")
        if N < self.n_clusters:
            raise ValueError(f"n_samples ({N}) < n_clusters ({self.n_clusters}).")

        has_labels = len(lbls) > 0
        Y = torch.stack(lbls).to(dev) if has_labels else None
        has_texts = len(texts_list) > 0

        class_w_t: Optional[torch.Tensor] = None
        if has_labels:
            y_np = Y.detach().cpu().numpy()
            if y_np.ndim == 2 and y_np.shape[1] > 1:
                y_np = y_np.argmax(axis=1)
            cw = self.compute_class_weights(y_np)
            class_w_t = torch.tensor([cw.get(i, 1.0) for i in range(2)], dtype=torch.float32, device=dev)

        opt_map = {
            "sgd": lambda: optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum),
            "adam": lambda: optim.Adam(self.parameters(), lr=learning_rate),
            "adamw": lambda: optim.AdamW(self.parameters(), lr=learning_rate),
            "rmsprop": lambda: optim.RMSprop(self.parameters(), lr=learning_rate, alpha=0.99),
            "adagrad": lambda: optim.Adagrad(self.parameters(), lr=learning_rate),
            "adamax": lambda: optim.Adamax(self.parameters(), lr=learning_rate),
            "asgd": lambda: optim.ASGD(self.parameters(), lr=learning_rate),
            "adadelta": lambda: optim.Adadelta(self.parameters(), lr=learning_rate),
            "nadam": lambda: optim.NAdam(self.parameters(), lr=learning_rate),
        }
        if optimizer_type.lower() not in opt_map:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        optimizer = opt_map[optimizer_type.lower()]()

        kld_loss = nn.KLDivLoss(reduction="batchmean")
        ce_loss = nn.CrossEntropyLoss(weight=class_w_t) if class_w_t is not None else nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        print("Initializing cluster centers with k-means.")
        y_pred_last = self._init_clusters_with_kmeans(X)

        valid_methods = ['tsne', 'pca', 'umap']
        if plot_method not in valid_methods:
            print(f"Warning: plot_method '{plot_method}' not in {valid_methods}. Using 'tsne'.")
            plot_method = 'tsne'
        best_viz_method = plot_method
        if plot_method == 'tsne' and compute_metrics and has_texts:
            feats_initial = self.extract_feature(X).cpu().numpy()
            best_viz_method, _ = self.select_best_visualization_method(feats_initial)

        # Initial plot
        if plot_evolution and plot_dir:
            try:
                feats0 = self.extract_feature(X).cpu().numpy()
                self.plot_cluster_evolution(
                    feats0, y_pred_last, 0,
                    texts=texts_list if has_texts else None,
                    save_dir=plot_dir,
                    method=best_viz_method,
                    show_plot=False,
                    plot_tfidf_version=has_texts,
                )
            except Exception as e:
                print(f"Warning: initial plot failed: {e}")

        # 🆕 Initial IG plot
        if plot_integrated_gradients and plot_dir and has_labels:
            try:
                idx_ig = np.random.choice(N, min(ig_max_samples, N), replace=False)
                ig_attrs = self.compute_integrated_gradients(
                    X[idx_ig], target_class=ig_target_class, n_steps=ig_n_steps
                )
                self.plot_integrated_gradients(
                    ig_attrs, y_pred_last[idx_ig], epoch=0,
                    save_dir=plot_dir, top_dims=ig_top_dims, show_plot=False,
                )
            except Exception as e:
                print(f"Warning: initial IG plot failed: {e}")

        log_path = os.path.join(save_dir, "idec_sentiment_log.csv")
        log_fieldnames = [
            "iter", "acc_sentiment", "L", "Lr", "Lc", "Ls",
            "ACC", "NMI", "ARI", "Homogeneity", "Completeness", "V-measure",
            "Silhouette",
            "Topic_Coherence", "Topic_Diversity", "Cluster_Balance",
            "Min_Cluster_Size", "Max_Cluster_Size",
            # 🆕 IG summary columns
            "IG_TopDim", "IG_TopDim_Score", "IG_Mean_Attribution",
        ]

        with open(log_path, "w", newline="") as logfile:
            writer = csv.DictWriter(logfile, fieldnames=log_fieldnames)
            writer.writeheader()

            save_interval = max(1, (max(1, N // batch_size)) * 5)
            train_loader: Optional[DataLoader] = None
            self.train()
            iter_count = 0
            tot_L = Lr = Lc = Ls = 0.0

            for ite in range(maxiter):
                if ite % update_interval == 0:
                    self.eval()
                    with torch.no_grad():
                        q_list, s_list = [], []
                        for i in range(0, N, batch_size):
                            qb, sb = self(X[i: i + batch_size])
                            q_list.append(qb)
                            s_list.append(sb)
                        q_all = torch.cat(q_list, dim=0)
                        s_all = torch.cat(s_list, dim=0)
                        p_all = self.target_distribution(q_all)

                        y_pred = q_all.argmax(dim=1).cpu().numpy()
                        delta = float((y_pred != y_pred_last).sum() / len(y_pred))
                        y_pred_last = y_pred.copy()

                        # Scatter plot
                        if plot_evolution and plot_dir and ite > 0 and (ite % plot_interval == 0):
                            try:
                                feats = self.extract_feature(X).cpu().numpy()
                                self.plot_cluster_evolution(
                                    feats, y_pred, ite,
                                    texts=texts_list if has_texts else None,
                                    save_dir=plot_dir,
                                    method=best_viz_method,
                                    show_plot=False,
                                    plot_tfidf_version=has_texts,
                                )
                            except Exception as e:
                                print(f"Warning: plot at iter {ite} failed: {e}")

                        acc_s = 0.0
                        if has_labels:
                            s_lab = s_all.argmax(dim=1).cpu().numpy()
                            y_true = Y.detach().cpu().numpy()
                            if y_true.ndim == 2 and y_true.shape[1] > 1:
                                y_true = y_true.argmax(axis=1)
                            acc_s = float((s_lab == y_true).mean())

                        feats_current = self.extract_feature(X).cpu().numpy()
                        clustering_sup = {}
                        if has_labels:
                            y_true_cluster = Y.detach().cpu().numpy()
                            if y_true_cluster.ndim == 2 and y_true_cluster.shape[1] > 1:
                                y_true_cluster = y_true_cluster.argmax(axis=1)
                            clustering_sup = self.compute_clustering_metrics(y_true_cluster, y_pred)

                        silhouette_current = self.compute_silhouette_score(feats_current, y_pred)
                        coherence_current = 0.0
                        diversity_current = 0.0
                        coverage_current = {'cluster_balance': 0.0, 'min_cluster_size': 0.0, 'max_cluster_size': 0.0}

                        if has_texts:
                            coherence_dict = self.compute_topic_coherence(texts_list, y_pred, top_n=10)
                            coherence_current = float(np.mean(list(coherence_dict.values()))) if coherence_dict else 0.0
                            diversity_current = self.compute_topic_diversity(texts_list, y_pred, top_n=10)
                            coverage_current = self.compute_topic_coverage(texts_list, y_pred)

                    avg_L = tot_L / update_interval if iter_count > 0 else 0.0
                    avg_Lr = Lr / update_interval if iter_count > 0 else 0.0
                    avg_Lc = Lc / update_interval if iter_count > 0 else 0.0
                    avg_Ls = Ls / update_interval if iter_count > 0 else 0.0

                    # ── IG plot — HARUS di luar torch.no_grad() agar grad bisa ngalir ──
                    ig_top_dim_idx = -1
                    ig_top_dim_score = 0.0
                    ig_mean_attr = 0.0

                    if plot_integrated_gradients and plot_dir and ite > 0 and (ite % plot_interval == 0):
                        try:
                            idx_ig = np.random.choice(N, min(ig_max_samples, N), replace=False)
                            ig_attrs = self.compute_integrated_gradients(
                                X[idx_ig],
                                target_class=ig_target_class,
                                n_steps=ig_n_steps,
                            )
                            self.plot_integrated_gradients(
                                ig_attrs, y_pred[idx_ig], epoch=ite,
                                save_dir=plot_dir, top_dims=ig_top_dims,
                                show_plot=False,
                            )
                            abs_ig = np.abs(ig_attrs)
                            global_imp = abs_ig.mean(axis=0)
                            ig_top_dim_idx = int(global_imp.argmax())
                            ig_top_dim_score = float(global_imp.max())
                            ig_mean_attr = float(global_imp.mean())
                        except Exception as e:
                            print(f"Warning: IG plot at iter {ite} failed: {e}")

                    # ── Token attribution per cluster (outside no_grad, needs BERT) ──
                    if (plot_token_attribution and has_texts and ite > 0
                            and (ite % plot_interval == 0)):
                        try:
                            print(f"  Computing token attribution per cluster (iter {ite})...")
                            ta_scores = self.compute_token_attribution_per_cluster(
                                texts=texts_list,
                                cluster_assignments=y_pred,
                                bert_model_name=token_attr_bert_name,
                                sentiment_class=token_attr_sentiment_class,
                                top_k=token_attr_top_k,
                                max_length=token_attr_max_length,
                                max_samples_per_cluster=token_attr_max_samples,
                            )
                            # Bar chart grid per cluster
                            self.plot_token_attribution_per_cluster(
                                ta_scores,
                                sentiment_class=token_attr_sentiment_class,
                                epoch=ite,
                                save_dir=plot_dir,
                                top_k=token_attr_top_k,
                                show_plot=False,
                            )
                            # Heatmap (rows=cluster, cols=tokens)
                            self.plot_token_attribution_heatmap(
                                ta_scores,
                                sentiment_class=token_attr_sentiment_class,
                                epoch=ite,
                                save_dir=plot_dir,
                                show_plot=False,
                            )
                        except Exception as e:
                            print(f"  Warning: token attribution at iter {ite} failed: {e}")

                    log_row = {
                        "iter": ite,
                        "acc_sentiment": round(acc_s, 5),
                        "L": round(avg_L, 5),
                        "Lr": round(avg_Lr, 5),
                        "Lc": round(avg_Lc, 5),
                        "Ls": round(avg_Ls, 5),
                        "ACC": round(clustering_sup.get('ACC', 0.0), 5),
                        "NMI": round(clustering_sup.get('NMI', 0.0), 5),
                        "ARI": round(clustering_sup.get('ARI', 0.0), 5),
                        "Homogeneity": round(clustering_sup.get('Homogeneity', 0.0), 5),
                        "Completeness": round(clustering_sup.get('Completeness', 0.0), 5),
                        "V-measure": round(clustering_sup.get('V-measure', 0.0), 5),
                        "Silhouette": round(silhouette_current, 5),
                        "Topic_Coherence": round(coherence_current, 5),
                        "Topic_Diversity": round(diversity_current, 5),
                        "Cluster_Balance": round(coverage_current['cluster_balance'], 5),
                        "Min_Cluster_Size": round(coverage_current['min_cluster_size'], 5),
                        "Max_Cluster_Size": round(coverage_current['max_cluster_size'], 5),
                        # 🆕 IG columns
                        "IG_TopDim": ig_top_dim_idx,
                        "IG_TopDim_Score": round(ig_top_dim_score, 6),
                        "IG_Mean_Attribution": round(ig_mean_attr, 6),
                    }
                    writer.writerow(log_row)

                    print(f"Iter {ite}: Lr={avg_Lr:.5f}, Lc={avg_Lc:.5f}, Ls={avg_Ls:.5f}, Acc={acc_s:.5f}; L={avg_L:.5f}")
                    if clustering_sup:
                        print(f"  Clustering: ACC={clustering_sup['ACC']:.4f}, NMI={clustering_sup['NMI']:.4f}, ARI={clustering_sup['ARI']:.4f}")
                    print(f"  Topic: Coherence={coherence_current:.4f}, Diversity={diversity_current:.4f}, Silhouette={silhouette_current:.4f}")
                    if ig_top_dim_idx >= 0:
                        print(f"  IG: top_dim=d{ig_top_dim_idx} ({ig_top_dim_score:.4f}), mean_attr={ig_mean_attr:.4f}")

                    tot_L = Lr = Lc = Ls = 0.0
                    iter_count = 0

                    if ite > 0 and delta < tol:
                        print(f"delta_label {delta:.6f} < tol {tol}. Stop.")
                        break

                    if has_labels:
                        train_loader = DataLoader(TensorDataset(X, p_all, Y), batch_size=batch_size, shuffle=True)
                    else:
                        train_loader = DataLoader(TensorDataset(X, p_all), batch_size=batch_size, shuffle=True)

                    self.train()

                assert train_loader is not None
                for batch in tqdm(train_loader, desc=f"Train {ite}", leave=False):
                    if has_labels and len(batch) == 3:
                        xb, pb, yb = batch
                        if yb.dim() > 1 and yb.shape[1] > 1:
                            yb = yb.argmax(dim=1)
                        yb = yb.long().to(dev)
                    else:
                        xb, pb = batch
                        yb = None

                    xb = xb.to(dev)
                    pb = pb.to(dev)

                    z = self.autoencoder.encode(xb)
                    x_recon = self.autoencoder.decode(z)
                    q = self.clustering(z)
                    s = torch.softmax(self.sentiment(z), dim=1)

                    recon_loss = mse_loss(x_recon, xb)
                    c_loss = kld_loss((q + 1e-8).log(), pb)
                    s_loss = torch.tensor(0.0, device=dev)
                    if yb is not None:
                        s_loss = ce_loss(s, yb)

                    loss = alpha * recon_loss + gamma * c_loss + eta * s_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tot_L += float(loss.item())
                    Lr += float(recon_loss.item())
                    Lc += float(c_loss.item())
                    Ls += float(s_loss.item())
                    iter_count += 1

                if ite % save_interval == 0 and ite > 0:
                    self.save_weights(os.path.join(save_dir, f"SEMTGPU_{ite}.weights.pth"))

        # Final plots
        if plot_evolution and plot_dir:
            try:
                feats_f = self.extract_feature(X).cpu().numpy()
                y_final = self.get_cluster_assignments(X)
                self.plot_cluster_evolution(
                    feats_f, y_final, ite,
                    texts=texts_list if has_texts else None,
                    save_dir=plot_dir,
                    method=best_viz_method,
                    show_plot=False,
                    plot_tfidf_version=has_texts,
                )
            except Exception as e:
                print(f"Warning: final plot failed: {e}")

        # 🆕 Final IG plot
        if plot_integrated_gradients and plot_dir and has_labels:
            try:
                y_final = self.get_cluster_assignments(X)
                idx_ig = np.random.choice(N, min(ig_max_samples, N), replace=False)
                ig_attrs_final = self.compute_integrated_gradients(
                    X[idx_ig], target_class=ig_target_class, n_steps=ig_n_steps
                )
                self.plot_integrated_gradients(
                    ig_attrs_final, y_final[idx_ig], epoch=ite,
                    save_dir=plot_dir, top_dims=ig_top_dims, show_plot=False,
                )
                print(f"✓ Final IG plot saved.")
            except Exception as e:
                print(f"Warning: final IG plot failed: {e}")

        self.save_weights(os.path.join(save_dir, "SEMTGPU_final.weights.pth"))

        self.eval()
        with torch.no_grad():
            q_all, s_all = self(X)
            y_pred_cluster = q_all.argmax(dim=1).cpu().numpy()
            y_pred_sentiment = s_all.argmax(dim=1).cpu().numpy()

            metrics = {}

            if has_labels:
                y_true = Y.detach().cpu().numpy()
                if y_true.ndim == 2 and y_true.shape[1] > 1:
                    y_true = y_true.argmax(axis=1)

                metrics['sentiment'] = {
                    'accuracy': float((y_pred_sentiment == y_true).mean()),
                    'precision': float(precision_score(y_true, y_pred_sentiment, average='binary', zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred_sentiment, average='binary', zero_division=0)),
                    'f1_score': float(f1_score(y_true, y_pred_sentiment, average='binary', zero_division=0)),
                }

                print("\n" + "=" * 60)
                print("FINAL SENTIMENT CLASSIFICATION METRICS")
                print("=" * 60)
                print(f"Accuracy:  {metrics['sentiment']['accuracy']:.4f}")
                print(f"Precision: {metrics['sentiment']['precision']:.4f}")
                print(f"Recall:    {metrics['sentiment']['recall']:.4f}")
                print(f"F1 Score:  {metrics['sentiment']['f1_score']:.4f}")
                print("=" * 60)

            if compute_metrics and has_texts:
                feats_final = self.extract_feature(X).cpu().numpy()
                true_cluster_labels = None
                if has_labels:
                    true_cluster_labels = Y.detach().cpu().numpy()
                    if true_cluster_labels.ndim == 2 and true_cluster_labels.shape[1] > 1:
                        true_cluster_labels = true_cluster_labels.argmax(axis=1)

                interp_metrics = self.compute_comprehensive_metrics(
                    feats_final, texts_list, y_pred_cluster,
                    true_labels=true_cluster_labels, top_n_words=10,
                )
                metrics['interpretability'] = interp_metrics

                metrics_path = os.path.join(save_dir, "interpretability_metrics.json")
                import json
                with open(metrics_path, 'w') as f:
                    json_metrics = {}
                    for k, v in interp_metrics.items():
                        if isinstance(v, dict):
                            json_metrics[k] = {str(kk): vv for kk, vv in v.items()}
                        else:
                            json_metrics[k] = v
                    json.dump(json_metrics, f, indent=2)
                print(f"✓ Interpretability metrics saved to: {metrics_path}")

            # 🆕 Final token attribution per cluster
            if plot_token_attribution and has_texts:
                try:
                    print("\nComputing final token attribution per cluster...")
                    ta_final = self.compute_token_attribution_per_cluster(
                        texts=texts_list,
                        cluster_assignments=y_pred_cluster,
                        bert_model_name=token_attr_bert_name,
                        sentiment_class=token_attr_sentiment_class,
                        top_k=token_attr_top_k,
                        max_length=token_attr_max_length,
                        max_samples_per_cluster=token_attr_max_samples,
                    )
                    self.plot_token_attribution_per_cluster(
                        ta_final,
                        sentiment_class=token_attr_sentiment_class,
                        epoch=ite,
                        save_dir=os.path.join(save_dir, "token_attribution"),
                        top_k=token_attr_top_k,
                        show_plot=False,
                    )
                    self.plot_token_attribution_heatmap(
                        ta_final,
                        sentiment_class=token_attr_sentiment_class,
                        epoch=ite,
                        save_dir=os.path.join(save_dir, "token_attribution"),
                        show_plot=False,
                    )
                    if 'interpretability' not in metrics:
                        metrics['interpretability'] = {}
                    metrics['interpretability']['token_attribution'] = {
                        str(cid): {
                            'top_tokens': list(scores.keys()),
                            'scores': list(scores.values()),
                        }
                        for cid, scores in ta_final.items()
                    }
                    print("✓ Final token attribution complete.")
                except Exception as e:
                    print(f"Warning: final token attribution failed: {e}")

            if has_labels:
                return y_pred_cluster, s_all.cpu().numpy(), metrics

            return y_pred_cluster

    # -------------------------
    # Helpers
    # -------------------------
    def _init_clusters_with_kmeans(
        self, all_embeddings: torch.Tensor, n_init: int = 20, random_state: int = 42
    ) -> np.ndarray:
        dev = next(self.parameters()).device
        N = all_embeddings.size(0)
        if N < self.n_clusters:
            raise ValueError(f"n_samples ({N}) < n_clusters ({self.n_clusters}).")
        self.eval()
        with torch.no_grad():
            feats = self.extract_feature(all_embeddings).cpu().numpy()
        km = KMeans(n_clusters=self.n_clusters, n_init=n_init, random_state=random_state)
        y_pred = km.fit_predict(feats)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=dev)
        self.clustering.clusters.data = centers
        return y_pred

    def get_cluster_assignments(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            q, _ = self(xt)
            return q.argmax(dim=1).cpu().numpy()

    def set_stop_words(self, stop_words: Union[Iterable[str], set[str]]) -> "SEMTGPU":
        self.stop_words = set(stop_words) if not isinstance(stop_words, set) else stop_words
        return self

    def map_texts_to_clusters(
        self, texts: List[str], cluster_assignments: np.ndarray
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[Tuple[str, int]]]]:
        clusters: Dict[int, List[str]] = {}
        n = min(len(texts), len(cluster_assignments))
        for i in range(n):
            cid = int(cluster_assignments[i])
            clusters.setdefault(cid, []).append(texts[i])

        common: Dict[int, List[Tuple[str, int]]] = {}
        for cid, txts in clusters.items():
            words = " ".join(txts).lower().split()
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            cnt = Counter(words)
            common[cid] = cnt.most_common(20)
        return clusters, common

    def analyze_clusters(self, x: Union[np.ndarray, torch.Tensor], texts: List[str]) -> pd.DataFrame:
        assigns = self.get_cluster_assignments(x)
        text_clusters, cluster_words = self.map_texts_to_clusters(texts, assigns)
        df = pd.DataFrame(
            [
                {
                    "Cluster": cid,
                    "Common Words": ", ".join([f"{w} ({c})" for w, c in words[:10]]),
                    "Text Count": len(text_clusters[cid]),
                }
                for cid, words in cluster_words.items()
            ]
        ).sort_values(by="Cluster").reset_index(drop=True)
        return df

    def set_topic(self, cluster_id: int, topic_name: str) -> "SEMTGPU":
        if not (isinstance(cluster_id, int) and 0 <= cluster_id < self.n_clusters):
            raise ValueError(f"cluster_id must be in [0, {self.n_clusters-1}]")
        if not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string")
        self.topic_mapping[cluster_id] = topic_name
        print(f"✓ Assigned topic '{topic_name}' to cluster {cluster_id}")
        return self

    def reset_topics(self) -> "SEMTGPU":
        self.topic_mapping = {}
        print("✓ All topic assignments reset")
        return self

    def get_topic_assignments(self) -> Dict[int, str]:
        return self.topic_mapping.copy()

    # -------------------------
    # Plots
    # -------------------------
    def plot_sentiment_by_topic(
        self,
        data: pd.DataFrame,
        x: Union[np.ndarray, torch.Tensor],
        figsize: Tuple[int, int] = (15, 6),
        palette: str = "Set1",
        negative_color: Optional[str] = None,
        positive_color: Optional[str] = None,
    ):
        sns.set_style("whitegrid")
        cluster = self.get_cluster_assignments(x)
        df = data.copy()
        df["cluster"] = cluster

        tab = df.groupby(["cluster", "sentiment"]).size().unstack(fill_value=0)
        total = tab.values.sum()
        pct = (tab / total) * 100

        pct["topic"] = pct.index.map(lambda cid: self.topic_mapping.get(cid, f"Cluster {cid}"))
        melted = pct.reset_index().melt(
            id_vars=["cluster", "topic"], value_vars=[0, 1], var_name="sentiment", value_name="percentage"
        ).sort_values("cluster")

        colors = sns.color_palette(palette, 2)
        neg_c = negative_color or colors[0]
        pos_c = positive_color or colors[1]
        labels = {0: self.class_labels.get(0, "Negative"), 1: self.class_labels.get(1, "Positive")}

        fig, ax = plt.subplots(figsize=figsize)
        topics = melted["topic"].drop_duplicates().tolist()
        bottoms = {t: 0 for t in topics}

        for s_val, color, label in zip([0, 1], [neg_c, pos_c], [labels[0], labels[1]]):
            sub = melted[melted["sentiment"] == s_val]
            y_vals = sub["topic"].tolist()
            x_vals = sub["percentage"].tolist()
            lefts = [bottoms[y] for y in y_vals]
            ax.barh(y=y_vals, width=x_vals, left=lefts, label=label, color=color, edgecolor="white", linewidth=0.5)
            for y, w, l in zip(y_vals, x_vals, lefts):
                if w > 3:
                    ax.text(l + w / 2, y, f"{w:.1f}%", va="center", ha="center", color="white", fontsize=9, fontweight="bold")
            for y, w in zip(y_vals, x_vals):
                bottoms[y] += w

        ax.set_xlabel("Percentage of All Reviews", fontsize=11)
        ax.set_ylabel("Cluster Topic", fontsize=11)
        ax.set_title("Sentiment Distribution by Cluster Topic", fontsize=14, fontweight="bold")
        ax.legend(title="Sentiment", frameon=True, fancybox=True, framealpha=0.9, shadow=True, fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        sns.despine()
        plt.tight_layout()
        return fig

    def plot_cluster_evolution(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        cluster_assignments: np.ndarray,
        epoch: int,
        texts: Optional[List[str]] = None,
        save_dir: str = "./results/fnnjst",
        method: str = "tsne",
        figsize: Tuple[int, int] = (6, 6),
        point_size: int = 20,
        alpha: float = 0.7,
        save_plot: bool = True,
        show_plot: bool = False,
        plot_tfidf_version: bool = True,
        max_keywords_per_cluster: int = 3,
        keyword_min_score: float = 0.3,
    ):
        if isinstance(embeddings, torch.Tensor):
            emb = embeddings.detach().cpu().numpy()
        else:
            emb = embeddings

        n = emb.shape[0]
        if n < 3:
            print(f"Skip plot at epoch {epoch}: n_samples={n} < 3")
            return None

        if method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            emb_2d = reducer.fit_transform(emb)
            method_label = f"PCA ({reducer.explained_variance_ratio_.sum():.1%} var)"
        elif method.lower() == "tsne":
            from sklearn.manifold import TSNE
            perplexity = max(2, min(30, n - 1))
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
            emb_2d = reducer.fit_transform(emb)
            method_label = "t-SNE"
        elif method.lower() == "umap":
            try:
                import umap
                n_neighbors = max(2, min(15, n - 1))
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
                emb_2d = reducer.fit_transform(emb)
                method_label = "UMAP"
            except Exception:
                from sklearn.manifold import TSNE
                perplexity = max(2, min(30, n - 1))
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
                emb_2d = reducer.fit_transform(emb)
                method_label = "t-SNE (fallback)"
        else:
            raise ValueError("method must be 'pca', 'tsne' or 'umap'")

        uniq = np.unique(cluster_assignments)
        k = len(uniq)
        if k <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
        elif k <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, k))

        # Plot 1: Regular scatter
        fig1, ax1 = plt.subplots(figsize=figsize)
        for i, cid in enumerate(uniq):
            mask = cluster_assignments == cid
            pts = emb_2d[mask]
            ax1.scatter(pts[:, 0], pts[:, 1], c=[colors[i]], marker="x", s=point_size, alpha=alpha, label=f"Cluster {cid}")
        ax1.set_title(f"Epoch {epoch} ({method_label})", fontsize=14, fontweight="bold")
        ax1.set_xticks([]), ax1.set_yticks([])
        for sp in ax1.spines.values():
            sp.set_visible(False)
        ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax1.set_facecolor("white")
        plt.tight_layout()

        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            out1 = os.path.join(save_dir, f"cluster_evolution_epoch_{epoch}.png")
            plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        if show_plot:
            plt.show()
        else:
            plt.close()

        # Plot 2: TF-IDF annotated
        if plot_tfidf_version and texts is not None and len(texts) == len(cluster_assignments):
            fig2, ax2 = plt.subplots(figsize=figsize)
            for i, cid in enumerate(uniq):
                mask = cluster_assignments == cid
                pts = emb_2d[mask]
                ax2.scatter(pts[:, 0], pts[:, 1], c=[colors[i]], marker="x", s=point_size, alpha=alpha)

            try:
                tfidf_keywords = self.extract_tfidf_keywords(
                    texts, cluster_assignments, top_n=max_keywords_per_cluster * 3, max_features=5000
                )
                for cid in uniq:
                    mask = cluster_assignments == cid
                    centroid = emb_2d[mask].mean(axis=0)
                    if cid in tfidf_keywords:
                        keywords = tfidf_keywords[cid]
                        scores = [score for _, score in keywords]
                        if scores:
                            adaptive_threshold = max(keyword_min_score, np.percentile(scores, 75) if len(scores) >= 4 else 0)
                            filtered_keywords = [(w, s) for w, s in keywords if s >= adaptive_threshold][:max_keywords_per_cluster]
                            all_words_count = {}
                            for other_cid, other_kw in tfidf_keywords.items():
                                for word, _ in other_kw[:10]:
                                    all_words_count[word] = all_words_count.get(word, 0) + 1
                            max_cluster_frequency = max(2, k * 0.5)
                            final_keywords = [(w, s) for w, s in filtered_keywords if all_words_count.get(w, 0) <= max_cluster_frequency]
                            if not final_keywords and filtered_keywords:
                                final_keywords = [filtered_keywords[0]]
                            if final_keywords:
                                keyword_text = "\n".join([word for word, _ in final_keywords])
                                ax2.annotate(
                                    keyword_text, xy=centroid, xytext=(5, 5),
                                    textcoords='offset points', fontsize=8, fontweight='bold',
                                    color=colors[list(uniq).index(cid)],
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                              edgecolor=colors[list(uniq).index(cid)], alpha=0.8, linewidth=1.5),
                                    ha='left', va='bottom', zorder=1000
                                )
                                ax2.scatter([centroid[0]], [centroid[1]], c=[colors[list(uniq).index(cid)]],
                                            marker='*', s=200, edgecolors='black', linewidths=1, zorder=999)
            except Exception as e:
                print(f"Warning: TF-IDF annotation failed at epoch {epoch}: {e}")

            ax2.set_title(f"Epoch {epoch} - TF-IDF Keywords ({method_label})", fontsize=14, fontweight="bold")
            ax2.set_xticks([]), ax2.set_yticks([])
            for sp in ax2.spines.values():
                sp.set_visible(False)
            ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
            ax2.set_facecolor("white")
            plt.tight_layout()

            if save_plot:
                out2 = os.path.join(save_dir, f"cluster_evolution_epoch_{epoch}_tfidf.png")
                plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
            if show_plot:
                plt.show()
            else:
                plt.close()
            return fig1, fig2

        return fig1

    def create_evolution_grid(
        self,
        save_dir: str = "./results/fnnjst",
        epochs_to_show: Optional[List[int]] = None,
        grid_cols: int = 3,
        figsize: Tuple[int, int] = (15, 10),
        plot_type: str = "both",
    ):
        if plot_type == "both":
            fig_regular = self._create_single_grid(save_dir, epochs_to_show, grid_cols, figsize, suffix="")
            fig_tfidf = self._create_single_grid(save_dir, epochs_to_show, grid_cols, figsize, suffix="_tfidf")
            return fig_regular, fig_tfidf
        elif plot_type == "tfidf":
            return self._create_single_grid(save_dir, epochs_to_show, grid_cols, figsize, suffix="_tfidf")
        else:
            return self._create_single_grid(save_dir, epochs_to_show, grid_cols, figsize, suffix="")

    def _create_single_grid(self, save_dir, epochs_to_show, grid_cols, figsize, suffix=""):
        import matplotlib.image as mpimg
        pattern = f"cluster_evolution_epoch_*{suffix}.png"
        files = glob.glob(os.path.join(save_dir, pattern))
        if not files:
            print(f"No cluster evolution plots found with pattern: {pattern}")
            return None

        epoch_files = []
        for f in files:
            m = re.search(r"epoch_(\d+)" + re.escape(suffix) + r"\.png", f)
            if m:
                epoch_files.append((int(m.group(1)), f))
        epoch_files.sort(key=lambda x: x[0])

        if epochs_to_show is not None:
            epoch_files = [ef for ef in epoch_files if ef[0] in epochs_to_show]
        if not epoch_files:
            print("No matching epoch plots found")
            return None

        n = len(epoch_files)
        rows = (n + grid_cols - 1) // grid_cols
        fig, axes = plt.subplots(rows, grid_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        plot_type_label = "TF-IDF Annotated" if suffix == "_tfidf" else "Regular"
        for i, (ep, fp) in enumerate(epoch_files):
            img = mpimg.imread(fp)
            axes[i].imshow(img)
            axes[i].set_title(f"Epoch {ep} ({plot_type_label})", fontsize=12, fontweight="bold")
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        out = os.path.join(save_dir, f"cluster_evolution_grid{suffix}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"✓ Evolution grid saved: {out}")
        return fig


    # =========================================================
    # TOKEN ATTRIBUTION (per-cluster, paper-quality)
    # =========================================================

    def _bert_cls_embedding(
        self,
        text: str,
        bert: "AutoModel",
        tokenizer: "AutoTokenizer",
        max_length: int = 128,
    ) -> Tuple[torch.Tensor, tuple, dict]:
        """Internal: tokenize → BERT → return (cls_emb, None, enc).
        output_attentions dihapus karena tidak kompatibel dengan sdpa attention.
        Occlusion method tidak butuh attention weights.
        """
        dev = next(self.parameters()).device
        enc = tokenizer(
            text, return_tensors="pt",
            padding=True, truncation=True, max_length=max_length
        ).to(dev)
        with torch.no_grad():
            out = bert(**enc)
        return out.last_hidden_state[:, 0, :], None, enc

    def _occlusion_scores(
        self,
        text: str,
        bert: "AutoModel",
        tokenizer: "AutoTokenizer",
        sentiment_class: int = 1,
        max_length: int = 128,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Leave-one-out occlusion: mask satu token pakai [MASK],
        ukur delta P(sentiment_class).

        Returns: tokens, scores (n_tokens,), base_prob (2,)
        """
        dev = next(self.parameters()).device
        cls_emb, _, enc = self._bert_cls_embedding(text, bert, tokenizer, max_length)

        # baseline prob
        self.eval()
        with torch.no_grad():
            _, s = self(cls_emb)
        base_prob = s.squeeze(0).cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
        input_ids = enc["input_ids"][0].cpu().tolist()
        mask_id = tokenizer.mask_token_id
        scores = np.zeros(len(input_ids))

        for i, tok in enumerate(tokens):
            if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]"):
                continue
            masked_ids = input_ids.copy()
            masked_ids[i] = mask_id
            inp = {k: enc[k].clone() for k in enc}
            inp["input_ids"] = torch.tensor([masked_ids], device=dev)
            with torch.no_grad():
                out = bert(**inp)
                cls_m = out.last_hidden_state[:, 0, :]
                _, sm = self(cls_m)
            masked_prob = sm.squeeze(0).cpu().numpy()
            scores[i] = float(base_prob[sentiment_class] - masked_prob[sentiment_class])

        return tokens, scores, base_prob

    def compute_token_attribution_per_cluster(
        self,
        texts: List[str],
        cluster_assignments: np.ndarray,
        bert_model_name: str = "indolem/indobert-base-uncased",
        sentiment_class: int = 1,
        top_k: int = 10,
        max_length: int = 128,
        max_samples_per_cluster: int = 50,
    ) -> Dict[int, Dict[str, float]]:
        """
        Hitung mean occlusion attribution per token, dikelompokkan per cluster.

        Ini backbone dari semua visualisasi token attribution.
        Hasilnya: {cluster_id: {token: mean_attribution_score}}

        Parameters
        ----------
        texts                   : raw text, len == len(cluster_assignments)
        cluster_assignments     : output dari predict_clusters() atau fit()
        bert_model_name         : HuggingFace model name
        sentiment_class         : 0=negative, 1=positive
        top_k                   : simpan top-K token per cluster
        max_samples_per_cluster : subsample per cluster agar ga lama

        Returns
        -------
        Dict[cluster_id, Dict[token, mean_score]]
        """
        dev = next(self.parameters()).device
        print(f"  Loading BERT tokenizer & model: {bert_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert = AutoModel.from_pretrained(
            bert_model_name,
            attn_implementation="eager",
        ).to(dev).eval()

        # Group indices by cluster
        cluster_indices: Dict[int, List[int]] = defaultdict(list)
        for i, cid in enumerate(cluster_assignments):
            cluster_indices[int(cid)].append(i)

        cluster_token_scores: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        self.eval()
        total_clusters = len(cluster_indices)

        for cid, indices in sorted(cluster_indices.items()):
            # Subsample if too many
            if len(indices) > max_samples_per_cluster:
                indices = list(np.random.choice(indices, max_samples_per_cluster, replace=False))

            print(f"  Cluster {cid} ({len(indices)} samples)...", end=" ", flush=True)

            for idx in indices:
                try:
                    tokens, scores, _ = self._occlusion_scores(
                        texts[idx], bert, tokenizer,
                        sentiment_class=sentiment_class,
                        max_length=max_length,
                    )
                    for tok, sc in zip(tokens, scores):
                        if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]"):
                            continue
                        clean = tok.replace("##", "").replace("▁", "").strip()
                        if len(clean) < 2:
                            continue
                        cluster_token_scores[cid][clean].append(sc)
                except Exception:
                    continue

            print("✓")

        # Aggregate: mean per token per cluster, keep top_k
        result: Dict[int, Dict[str, float]] = {}
        for cid, tok_dict in cluster_token_scores.items():
            mean_scores = {
                tok: float(np.mean(sc_list))
                for tok, sc_list in tok_dict.items()
                if len(sc_list) >= 2
            }
            # Sort by absolute value, keep top_k
            top = sorted(mean_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
            result[cid] = dict(top)

        del bert  # free GPU memory
        return result

    def plot_token_attribution_per_cluster(
        self,
        cluster_token_scores: Dict[int, Dict[str, float]],
        sentiment_class: int = 1,
        epoch: int = 0,
        save_dir: str = "./results/fnnjst",
        figsize: Tuple[int, int] = (16, 10),
        top_k: int = 10,
        max_clusters_per_row: int = 4,
        save_plot: bool = True,
        show_plot: bool = False,
    ) -> plt.Figure:
        """
        Publication-quality figure: grid of bar charts, satu panel per cluster.

        Setiap panel menampilkan top token beserta attribution score-nya.
        Warna hijau = mendukung sentiment_class, merah = melawan.

        Parameters
        ----------
        cluster_token_scores : output dari compute_token_attribution_per_cluster()
        epoch                : current training epoch (untuk filename & title)
        max_clusters_per_row : jumlah kolom dalam grid
        """
        sent_label = {0: "Negative", 1: "Positive"}.get(sentiment_class, str(sentiment_class))
        cluster_ids = sorted(cluster_token_scores.keys())
        n_clusters = len(cluster_ids)

        if n_clusters == 0:
            print("No cluster token scores to plot.")
            return None

        ncols = min(max_clusters_per_row, n_clusters)
        nrows = (n_clusters + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize[0], figsize[1] * nrows / max(2, nrows)),
            facecolor="white"
        )
        axes = np.array(axes).reshape(-1)

        # Global vmax for consistent color scale
        all_vals = [v for d in cluster_token_scores.values() for v in d.values()]
        vmax = max(abs(v) for v in all_vals) if all_vals else 1.0

        for i, cid in enumerate(cluster_ids):
            ax = axes[i]
            tok_dict = cluster_token_scores[cid]

            if not tok_dict:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#aaa")
                ax.axis("off")
                continue

            # Sort by score value (not abs) for readability
            items = sorted(tok_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            tokens_ = [t for t, _ in items]
            scores_ = [s for _, s in items]

            # Color: green if positive attribution, red if negative
            colors = [
                "#2ca02c" if s > 0 else "#d62728"
                for s in scores_
            ]
            # Intensity by magnitude
            colors = [
                _RWG(0.5 + 0.5 * (s / vmax))
                for s in scores_
            ]

            y_pos = np.arange(len(tokens_))
            ax.barh(y_pos, scores_, color=colors, edgecolor="white",
                    linewidth=0.3, height=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tokens_, fontsize=8, fontfamily="monospace")
            ax.invert_yaxis()
            ax.axvline(0, color="#888", linewidth=0.7, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#ddd")
            ax.tick_params(colors="#555", labelsize=8)
            ax.set_xlabel("Attribution", fontsize=7, color="#555")

            topic = self.topic_mapping.get(cid, f"C{cid}")
            n_txt = sum(1 for v in tok_dict.values())
            ax.set_title(f"{topic}  (n≈{n_txt})", fontsize=9,
                         fontweight="bold", color="#222", pad=5)

            # Annotate bar values
            for bar, val in zip(ax.patches, scores_):
                ax.text(
                    val + vmax * 0.01 if val >= 0 else val - vmax * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=6, color="#444"
                )

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Legend
        pos_p = mpatches.Patch(color="#2ca02c", label=f"Supports {sent_label}")
        neg_p = mpatches.Patch(color="#d62728", label=f"Opposes {sent_label}")
        fig.legend(
            handles=[pos_p, neg_p],
            loc="upper right", fontsize=9,
            framealpha=0.9, ncol=2,
        )

        fig.suptitle(
            f"Token Attribution per Cluster  —  Epoch {epoch}\n"
            f"(Method: Occlusion/LOO  |  Target: {sent_label}  |  "
            f"{n_clusters} clusters)",
            fontsize=12, fontweight="bold", y=1.01
        )

        plt.tight_layout()

        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"token_attr_per_cluster_epoch_{epoch}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            print(f"  ✓ Token attr plot saved: {out}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_token_attribution_heatmap(
        self,
        cluster_token_scores: Dict[int, Dict[str, float]],
        sentiment_class: int = 1,
        epoch: int = 0,
        save_dir: str = "./results/fnnjst",
        top_k_global: int = 20,
        figsize: Tuple[int, int] = (16, 8),
        save_plot: bool = True,
        show_plot: bool = False,
    ) -> plt.Figure:
        """
        Heatmap rows=cluster, cols=top tokens (global union).
        Figure ini sangat cocok untuk paper karena menunjukkan
        perbedaan karakteristik token antar cluster dalam satu panel.

        Parameters
        ----------
        top_k_global : ambil top-K token berdasarkan mean attribution global
        """
        sent_label = {0: "Negative", 1: "Positive"}.get(sentiment_class, str(sentiment_class))

        # Global token importance (mean across all clusters)
        global_scores: Dict[str, List[float]] = defaultdict(list)
        for cid, tok_dict in cluster_token_scores.items():
            for tok, sc in tok_dict.items():
                global_scores[tok].append(sc)

        global_mean = {
            tok: float(np.mean(scs))
            for tok, scs in global_scores.items()
        }
        top_tokens = [
            t for t, _ in sorted(
                global_mean.items(), key=lambda x: abs(x[1]), reverse=True
            )[:top_k_global]
        ]

        cluster_ids = sorted(cluster_token_scores.keys())
        matrix = np.zeros((len(cluster_ids), len(top_tokens)))

        for i, cid in enumerate(cluster_ids):
            for j, tok in enumerate(top_tokens):
                matrix[i, j] = cluster_token_scores[cid].get(tok, 0.0)

        vmax = np.abs(matrix).max() if matrix.any() else 1.0

        fig, ax = plt.subplots(figsize=figsize, facecolor="white")

        im = ax.imshow(
            matrix, cmap=_RWG, vmin=-vmax, vmax=vmax,
            aspect="auto", interpolation="nearest"
        )

        ax.set_xticks(range(len(top_tokens)))
        ax.set_xticklabels(top_tokens, rotation=45, ha="right",
                           fontsize=9, fontfamily="monospace")

        ylabels = [
            f"{self.topic_mapping.get(cid, f'C{cid}')}"
            for cid in cluster_ids
        ]
        ax.set_yticks(range(len(cluster_ids)))
        ax.set_yticklabels(ylabels, fontsize=9)

        # Cell annotations
        if len(top_tokens) <= 25 and len(cluster_ids) <= 25:
            for i in range(len(cluster_ids)):
                for j in range(len(top_tokens)):
                    val = matrix[i, j]
                    if val == 0.0:
                        continue
                    txt_color = "white" if abs(val) > vmax * 0.5 else "#333"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=txt_color, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(f"Mean Attribution → {sent_label}", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(
            f"Per-Cluster Token Attribution Heatmap  —  Epoch {epoch}\n"
            f"(Green=supports {sent_label}, Red=opposes | "
            f"top-{top_k_global} global tokens | method=occlusion)",
            fontsize=11, fontweight="bold", pad=12
        )
        ax.set_xlabel("Token / Kata", fontsize=10)
        ax.set_ylabel("Cluster / Topik", fontsize=10)

        # Grid
        ax.set_xticks(np.arange(-0.5, len(top_tokens), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(cluster_ids), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)

        plt.tight_layout()

        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"token_attr_heatmap_epoch_{epoch}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            print(f"  ✓ Token attr heatmap saved: {out}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def explain_single_text(
        self,
        text: str,
        bert_model_name: str = "indolem/indobert-base-uncased",
        sentiment_class: int = 1,
        max_length: int = 128,
        figsize: Tuple[int, int] = (12, 4),
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Visualisasi token attribution untuk SATU kalimat.
        Dua panel: bar chart + highlighted text.
        Cocok untuk lampiran / case study di paper.
        """
        dev = next(self.parameters()).device
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert = AutoModel.from_pretrained(
            bert_model_name,
            attn_implementation="eager",
        ).to(dev).eval()

        tokens, scores, base_prob = self._occlusion_scores(
            text, bert, tokenizer, sentiment_class, max_length
        )
        del bert

        pred_class = int(base_prob.argmax())
        pred_label = {0: "Negative", 1: "Positive"}.get(pred_class, str(pred_class))
        pred_conf = float(base_prob.max())
        sent_label = {0: "Negative", 1: "Positive"}.get(sentiment_class, str(sentiment_class))

        # Filter special tokens
        disp_toks, disp_scores = [], []
        for tok, sc in zip(tokens, scores):
            if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]"):
                continue
            clean = tok.replace("##", "").replace("▁", "")
            disp_toks.append(clean)
            disp_scores.append(sc)
        disp_toks = np.array(disp_toks)
        disp_scores = np.array(disp_scores)

        vmax = max(abs(disp_scores).max(), 1e-6)

        fig = plt.figure(figsize=figsize, facecolor="white")
        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.45)
        ax_bar = fig.add_subplot(gs[0])
        ax_txt = fig.add_subplot(gs[1])

        # Panel A: bar chart
        bar_colors = [_RWG(0.5 + 0.5 * (s / vmax)) for s in disp_scores]
        y_pos = np.arange(len(disp_toks))
        ax_bar.barh(y_pos, disp_scores, color=bar_colors,
                    edgecolor="white", linewidth=0.4, height=0.7)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(disp_toks, fontsize=9, fontfamily="monospace")
        ax_bar.axvline(0, color="#555", linewidth=0.8, linestyle="--")
        ax_bar.invert_yaxis()
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.set_xlabel("Attribution Score (Occlusion)", fontsize=9)
        ax_bar.set_title(
            f"Token Attribution → {sent_label}  |  Pred: {pred_label} ({pred_conf:.1%})",
            fontsize=10, fontweight="bold"
        )
        badge_c = "#2ca02c" if pred_class == 1 else "#d62728"
        ax_bar.text(1.01, 0.5, f"{pred_label}\n{pred_conf:.1%}",
                    transform=ax_bar.transAxes, fontsize=8,
                    va="center", ha="left", color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_c, alpha=0.9))

        # Panel B: highlighted text
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)
        ax_txt.axis("off")

        x, y = 0.01, 0.75
        char_w = 0.013
        for tok, sc in zip(disp_toks, disp_scores):
            w = len(tok) * char_w + 0.015
            if x + w > 0.98:
                x, y = 0.01, y - 0.45
            if y < 0.05:
                break
            intensity = abs(sc) / vmax
            if sc > 0:
                bg = plt.cm.Greens(0.2 + 0.6 * intensity)
                fg = "#1a5c1a" if intensity > 0.5 else "#333"
            else:
                bg = plt.cm.Reds(0.2 + 0.6 * intensity)
                fg = "#7a0c0c" if intensity > 0.5 else "#333"
            ax_txt.text(
                x + w / 2, y, tok, ha="center", va="center",
                fontsize=8.5, color=fg, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=bg,
                          edgecolor="none", alpha=0.85)
            )
            x += w + 0.005
        ax_txt.set_title("Token Highlight", fontsize=9, color="#555", pad=4)

        fig.suptitle(
            f'"{text[:90]}{"..." if len(text) > 90 else ""}"',
            fontsize=8, style="italic", color="#666", y=1.01
        )
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"✓ Saved: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()

        return fig