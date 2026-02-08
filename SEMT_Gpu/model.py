"""
SEMT-GPU: Feedforward Neural Network (PyTorch) for
Joint Sentiment Analysis and Topic Clustering

- Autoencoder for representation learning
- Student-t Clustering layer (DEC-style) for soft assignments
- Sentiment head (binary) with class-weighted CE
- Robust training loop (fit) dengan target distribution
- Comprehensive interpretability metrics with paper references:

DIMENSIONALITY REDUCTION & VISUALIZATION:
  * PCA: Jolliffe (2002). "Principal Component Analysis"
  * t-SNE: van der Maaten & Hinton (2008). "Visualizing Data using t-SNE"
  * UMAP: McInnes et al. (2018). "UMAP: Uniform Manifold Approximation and Projection"

CLUSTERING EVALUATION (Supervised):
  * ACC: Hungarian algorithm for optimal matching
  * NMI: Strehl & Ghosh (2002). "Cluster ensembles"
  * ARI: Hubert & Arabie (1985). "Comparing partitions"
  * Purity: Zhao & Karypis (2001). "Criterion functions for document clustering"
  * V-measure: Rosenberg & Hirschberg (2007). "V-Measure: A conditional entropy-based external cluster evaluation measure"

CLUSTERING EVALUATION (Unsupervised):
  * Silhouette: Rousseeuw (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis"

TOPIC INTERPRETABILITY:
  * TF-IDF Keywords: Ramos (2003). "Using TF-IDF to determine word relevance in document queries"
  * Topic Coherence (NPMI): Mimno et al. (2011). "Optimizing Semantic Coherence in Topic Models"
                           Röder et al. (2015). "Exploring the Space of Topic Coherence Measures"
  * Topic Diversity: Dieng et al. (2020). "Topic Modeling in Embedding Spaces"
                    Bianchi et al. (2021). "Cross-lingual Contextualized Topic Models with Zero-shot Learning"
  * Coverage: Entropy-based cluster balance metric
"""

from __future__ import annotations

import os
import csv
import glob
import re
from collections import Counter
from typing import Iterable, List, Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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


# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------
def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Clustering accuracy via Hungarian algorithm (for evaluation use only).

    Args:
        y_true: shape (n_samples,)
        y_pred: shape (n_samples,)
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # linear_assignment returns (row_ind, col_ind) tuple
    row_ind, col_ind = linear_assignment(w.max() - w)
    return float(sum(w[i, j] for i, j in zip(row_ind, col_ind)) / y_pred.size)

# --------------------------------------------------------------------------------------
# Model Components
# --------------------------------------------------------------------------------------
class ClusteringLayer(nn.Module):
    """
    Student's t-distribution soft assignment layer (DEC).
    """

    def __init__(self, n_clusters: int, input_dim: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.empty(n_clusters, input_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.clusters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # q_ij ∝ (1 + ||x - μ_j||^2 / α)^(-(α+1)/2)
        dist = torch.sum((x.unsqueeze(1) - self.clusters.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q


class Autoencoder(nn.Module):
    """
    Symmetric fully-connected autoencoder: dims = [input_dim, h1, ..., z]
    - Encoder: activations di semua layer kecuali bottleneck
    - Decoder: mirror; activations di semua layer kecuali layer output
    """
    def __init__(self, dims, act='relu'):
        super().__init__()
        assert len(dims) >= 2, "dims minimal [input_dim, latent]"
        self.dims = list(dims)

        act_map = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}
        self.activation = act_map.get(act, nn.ReLU())

        # ---- Encoder ----
        enc = []
        for i in range(len(dims)-2):                 # semua hidden dgn aktivasi
            enc += [nn.Linear(dims[i], dims[i+1]), self.activation]
        enc += [nn.Linear(dims[-2], dims[-1])]       # bottleneck (tanpa aktivasi)
        self.encoder = nn.Sequential(*enc)

        # ---- Decoder (benar-benar simetris, tanpa duplikasi) ----
        dec = []
        for j in range(len(dims)-1, 0, -1):          # z->h_{L-1}->...->in
            dec += [nn.Linear(dims[j], dims[j-1])]
            if j != 1:                                # semua kecuali layer output
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
        self.alpha_clustering = float(alpha_clustering)  # parameter untuk clustering layer (student-t)

        # Core
        self.autoencoder = Autoencoder(dims)
        self.clustering = ClusteringLayer(n_clusters=n_clusters, input_dim=dims[-1], alpha=alpha_clustering)

        # Sentiment head (binary)
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

        # Metadata
        self.class_labels: Dict[int, str] = {0: "negative", 1: "positive"}
        self.topic_mapping: Dict[int, str] = {}
        self.stop_words: set[str] = set()

        self._init_head()

    # -------------------------
    # Basics
    # -------------------------
    def _init_head(self) -> None:
        for m in self.sentiment.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.autoencoder.encode(x)
        q = self.clustering(z)                         # soft cluster probs
        s = torch.softmax(self.sentiment(z), dim=1)    # sentiment probs
        return q, s

    def extract_feature(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            z = self.autoencoder.encode(xt)
        return z

    # -------------------------
    # Weights I/O
    # -------------------------
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

    # -------------------------
    # Autoencoder pretrain
    # -------------------------

    def pretrain_autoencoder(
        self,
        dataset: Iterable,
        batch_size: int = 256,
        epochs: int = 200,
        lr: float = 1e-3,
        save_dir: str = "./results/ae",
        weights_name: str = "pretrained_ae.weights.pth",
    ) -> str:
        """
        Pretrain autoencoder (MSE reconstruction) dan simpan bobot AE saja.
        Aman kalau save_dir kosong/None (fallback ke './results/ae').
        Returns: path file bobot yang tersimpan (str).
        """
        print("=" * 60)
        print("Pretraining Autoencoder")
        print("=" * 60)

        # --- siapkan folder simpan ---
        save_path = Path(save_dir or "./results/ae")
        save_path.mkdir(parents=True, exist_ok=True)
        weights_path = save_path / weights_name

        # --- kumpulkan embeddings ---
        embs: List[torch.Tensor] = []
        for i in range(len(dataset)):
            item = dataset[i]
            emb = item[0] if (isinstance(item, tuple) and len(item) >= 1) else item
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            else:
                emb = emb.detach()
            embs.append(emb.cpu())

        X = torch.stack(embs)             # (N, d)
        n, d = X.shape

        # --- sanity check dimensi ---
        if d != self.dims[0]:
            raise ValueError(
                f"Input dim mismatch: embeddings dim={d}, tapi dims[0]={self.dims[0]}. "
                f"Bangun model dengan dims=[{d}, ...] terlebih dulu."
            )

        # --- data loader ---
        dev = next(self.parameters()).device
        X = X.to(dev)
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

        # --- training ---
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

        # --- simpan hanya bobot AE (bukan seluruh model) ---
        torch.save(
            {"autoencoder_state_dict": self.autoencoder.state_dict(), "dims": self.dims},
            str(weights_path),
        )
        print(f"✓ Autoencoder pretraining complete. Saved to: {weights_path}")
        return str(weights_path)


    # -------------------------
    # DEC target distribution
    # -------------------------
    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        weight = (q ** 2) / torch.clamp(torch.sum(q, dim=0), min=1e-12)
        p = (weight.t() / torch.clamp(torch.sum(weight, dim=1), min=1e-12)).t()
        return p

    # -------------------------
    # Class weights (for CE)
    # -------------------------
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

    # -------------------------
    # Predictions
    # -------------------------
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
        """
        Predict clusters & sentiment. If inputs are texts, encodes with BERT [CLS].
        """
        self.eval()
        dev = next(self.parameters()).device

        # Text path
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
            X = out.last_hidden_state[:, 0, :]  # [CLS]
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
    # 🆕 INTERPRETABILITY METRICS (with Paper References)
    # -------------------------
    
    def extract_tfidf_keywords(
        self,
        texts: List[str],
        cluster_assignments: np.ndarray,
        top_n: int = 10,
        max_features: int = 5000,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract cluster keywords using TF-IDF.
        
        Reference: 
        - Ramos, J. (2003). "Using TF-IDF to determine word relevance in document queries"
        - Blei et al. (2003). "Latent Dirichlet Allocation" (comparison baseline)
        
        Args:
            texts: List of documents
            cluster_assignments: Cluster labels for each document
            top_n: Number of top keywords per cluster
            max_features: Max vocabulary size for TF-IDF
        
        Returns:
            Dict mapping cluster_id to list of (keyword, tfidf_score) tuples
        """
        # Group texts by cluster
        clusters: Dict[int, List[str]] = {}
        for i, cid in enumerate(cluster_assignments):
            clusters.setdefault(int(cid), []).append(texts[i])
        
        cluster_keywords = {}
        
        for cid, cluster_texts in clusters.items():
            if not cluster_texts:
                cluster_keywords[cid] = []
                continue
            
            # Combine all texts in cluster into one document
            cluster_doc = " ".join(cluster_texts)
            
            # TF-IDF on cluster vs all other clusters
            all_cluster_docs = [" ".join(txts) for txts in clusters.values()]
            
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words=list(self.stop_words) if self.stop_words else None,
                min_df=1,
                ngram_range=(1, 2),  # unigrams and bigrams
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(all_cluster_docs)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get TF-IDF scores for this cluster
                cluster_idx = list(clusters.keys()).index(cid)
                scores = tfidf_matrix[cluster_idx].toarray()[0]
                
                # Get top keywords
                top_indices = scores.argsort()[-top_n:][::-1]
                keywords = [(feature_names[i], float(scores[i])) for i in top_indices]
                cluster_keywords[cid] = keywords
                
            except Exception as e:
                print(f"Warning: TF-IDF failed for cluster {cid}: {e}")
                # Fallback to simple word count
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
        """
        Compute comprehensive clustering evaluation metrics.
        
        References:
        - NMI: Strehl & Ghosh (2002). "Cluster ensembles"
        - ARI: Hubert & Arabie (1985). "Comparing partitions"
        - V-measure: Rosenberg & Hirschberg (2007). "V-Measure: A conditional entropy-based external cluster evaluation measure"
        - Topic Coverage: Entropy-based cluster balance metric
        
        Returns:
            Dict with metrics: ACC, NMI, ARI, Homogeneity, Completeness, V-measure, Topic_Coverage
        """
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            return {
                'ACC': 0.0, 'NMI': 0.0, 'ARI': 0.0,
                'Homogeneity': 0.0, 'Completeness': 0.0, 'V-measure': 0.0,
                'Topic_Coverage': 0.0,
            }
        
        # Calculate topic coverage (cluster entropy balance)
        unique_clusters, cluster_counts = np.unique(y_pred, return_counts=True)
        cluster_probs = cluster_counts / len(y_pred)
        # Use negative entropy to get coverage score (0-1)
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
        """
        Compute variance explained by PCA for dimensionality reduction methods.
        Returns cumulative variance explained by top n_components.
        
        Reference: Jolliffe & Cadima (2016). "Principal component analysis: a review and recent developments"
        """
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
        """
        Automatically select best visualization method based on variance analysis.
        
        References:
        - PCA: Jolliffe (2002). "Principal Component Analysis"
        - t-SNE: van der Maaten & Hinton (2008). "Visualizing Data using t-SNE"
        - UMAP: McInnes et al. (2018). "UMAP: Uniform Manifold Approximation and Projection"
        
        Args:
            embeddings: High-dimensional embeddings
            methods: List of methods to consider ['pca', 'tsne', 'umap']
            variance_threshold: If PCA explains >= this variance, use PCA (faster)
        
        Returns:
            (best_method, metrics_dict)
        """
        if isinstance(embeddings, torch.Tensor):
            emb = embeddings.detach().cpu().numpy()
        else:
            emb = embeddings
        
        n = emb.shape[0]
        
        # Check PCA variance first
        variance_info = self.compute_variance_explained(emb, n_components=2)
        var_explained = variance_info['variance_explained']
        
        metrics = {
            'pca_variance_2d': var_explained,
            'n_samples': n,
        }
        
        # If PCA explains enough variance, use it (faster & interpretable)
        if var_explained >= variance_threshold:
            print(f"✓ PCA explains {var_explained:.2%} variance → using PCA for visualization")
            return 'pca', metrics
        
        # Otherwise prefer t-SNE for medium datasets, UMAP for large
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
        """
        Compute silhouette score for cluster quality assessment.
        
        Reference: Rousseeuw (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis"
        
        Score ranges from -1 to 1, higher is better.
        """
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
        """
        Compute topic coherence for each cluster using PMI-based coherence.
        
        Reference: 
        - Mimno et al. (2011). "Optimizing Semantic Coherence in Topic Models"
        - Röder et al. (2015). "Exploring the Space of Topic Coherence Measures"
        
        Higher values indicate more coherent topics.
        
        Returns:
            Dict mapping cluster_id to coherence score
        """
        from itertools import combinations
        
        # Group texts by cluster
        clusters: Dict[int, List[str]] = {}
        for i, cid in enumerate(cluster_assignments):
            clusters.setdefault(int(cid), []).append(texts[i])
        
        coherence_scores = {}
        
        for cid, cluster_texts in clusters.items():
            # Get top words for this cluster
            words = " ".join(cluster_texts).lower().split()
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            word_counts = Counter(words)
            top_words = [w for w, _ in word_counts.most_common(top_n)]
            
            if len(top_words) < 2:
                coherence_scores[cid] = 0.0
                continue
            
            # Compute PMI-based coherence (NPMI variant)
            doc_freq = {}
            for word in top_words:
                doc_freq[word] = sum(1 for text in cluster_texts if word in text.lower())
            
            npmi_scores = []
            for w1, w2 in combinations(top_words, 2):
                # Co-occurrence
                co_occur = sum(1 for text in cluster_texts if w1 in text.lower() and w2 in text.lower())
                
                if co_occur > 0 and doc_freq[w1] > 0 and doc_freq[w2] > 0:
                    p_w1_w2 = co_occur / len(cluster_texts)
                    p_w1 = doc_freq[w1] / len(cluster_texts)
                    p_w2 = doc_freq[w2] / len(cluster_texts)
                    
                    # NPMI: normalized PMI to [-1, 1] range
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
        """
        Compute topic diversity across all clusters.
        Measures how distinct topics are from each other.
        
        Reference: 
        - Dieng et al. (2020). "Topic Modeling in Embedding Spaces"
        - Bianchi et al. (2021). "Cross-lingual Contextualized Topic Models with Zero-shot Learning"
        
        Returns:
            Diversity score (0-1), higher means more diverse topics
        """
        # Group texts by cluster
        clusters: Dict[int, List[str]] = {}
        for i, cid in enumerate(cluster_assignments):
            clusters.setdefault(int(cid), []).append(texts[i])
        
        # Get top words for each cluster
        all_top_words = []
        for cid, cluster_texts in clusters.items():
            words = " ".join(cluster_texts).lower().split()
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            word_counts = Counter(words)
            top_words = set([w for w, _ in word_counts.most_common(top_n)])
            all_top_words.append(top_words)
        
        if len(all_top_words) < 2:
            return 0.0
        
        # Compute pairwise uniqueness
        unique_words = set()
        total_words = 0
        for word_set in all_top_words:
            unique_words.update(word_set)
            total_words += len(word_set)
        
        # Diversity = unique words / total words (accounting for overlap)
        diversity = len(unique_words) / total_words if total_words > 0 else 0.0
        return float(diversity)
    
    def compute_topic_coverage(
        self,
        texts: List[str],
        cluster_assignments: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute how well topics cover the dataset.
        
        Reference: Evaluation metrics from clustering literature
        
        Returns:
            Dict with coverage metrics:
            - cluster_balance: How evenly distributed samples are across clusters (0-1)
            - min_cluster_size: Smallest cluster size (%)
            - max_cluster_size: Largest cluster size (%)
        """
        cluster_counts = Counter(cluster_assignments)
        total = len(cluster_assignments)
        
        sizes = [count / total for count in cluster_counts.values()]
        
        # Entropy-based balance (higher = more balanced)
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
        """
        Compute all interpretability metrics in one go.
        
        Returns comprehensive metrics dict with:
        - Clustering metrics (if true_labels provided): ACC, NMI, ARI, Topic_Coverage, etc.
        - Visualization method selection
        - Silhouette score
        - Topic coherence (per cluster + mean)
        - Topic diversity
        - Topic coverage
        - TF-IDF keywords per cluster
        """
        print("\n" + "=" * 60)
        print("Computing Comprehensive Interpretability Metrics")
        print("=" * 60)
        
        # Clustering quality metrics (if ground truth available)
        clustering_metrics = {}
        if true_labels is not None:
            clustering_metrics = self.compute_clustering_metrics(true_labels, cluster_assignments)
        
        # Select best visualization method
        best_method, viz_metrics = self.select_best_visualization_method(embeddings)
        
        # Clustering quality (unsupervised)
        silhouette = self.compute_silhouette_score(embeddings, cluster_assignments)
        
        # Topic interpretability metrics
        coherence = self.compute_topic_coherence(texts, cluster_assignments, top_n=top_n_words)
        diversity = self.compute_topic_diversity(texts, cluster_assignments, top_n=top_n_words)
        coverage = self.compute_topic_coverage(texts, cluster_assignments)
        
        # TF-IDF keywords
        tfidf_keywords = self.extract_tfidf_keywords(texts, cluster_assignments, top_n=top_n_words)
        
        metrics = {
            'clustering_supervised': clustering_metrics,  # ACC, NMI, ARI, Topic_Coverage, etc.
            'visualization': {
                'best_method': best_method,
                **viz_metrics,
            },
            'clustering_unsupervised': {
                'silhouette_score': silhouette,
            },
            'topic_coherence': coherence,
            'topic_coherence_mean': float(np.mean(list(coherence.values()))) if coherence else 0.0,
            'topic_diversity': diversity,
            'topic_coverage': coverage,
            'tfidf_keywords': tfidf_keywords,
        }
        
        # Print summary
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
        print(f"   Topic_Coverage:{clustering_metrics['Topic_Coverage']:.4f}")
        
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
        plot_method: str = "tsne",  # 🆕 choose visualization method: 'tsne', 'pca', 'umap'
        compute_metrics: bool = True,  # 🆕 compute comprehensive metrics
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
        """
        Joint training (DEC + Sentiment + Reconstruction). 
        
        Loss = alpha * L_reconstruction + gamma * L_clustering + eta * L_sentiment
        
        Args:
            alpha: weight for reconstruction loss (MSE) - default 0.1
            gamma: weight for clustering loss (KL divergence) - default 1.0
            eta: weight for sentiment loss (Cross Entropy) - default 0.1
            plot_method: visualization method for plots - 'tsne' (default), 'pca', or 'umap'
            compute_metrics: whether to compute comprehensive interpretability metrics
            
        Returns:
            If has_labels: (cluster_pred, sentiment_probs, metrics_dict)
            Else: cluster_pred
        """
        print("=" * 60)
        print("Joint Training: Clustering + Sentiment + Reconstruction")
        print(f"Loss weights - Alpha (recon): {alpha}, Gamma (cluster): {gamma}, Eta (sentiment): {eta}")
        print(f"Update interval: {update_interval}")
        print("=" * 60)
        
        dev = next(self.parameters()).device
        maxiter = int(maxiter)
        plot_interval = int(plot_interval) if plot_interval is not None else update_interval

        os.makedirs(save_dir, exist_ok=True)
        plot_dir = None
        if plot_evolution:
            plot_dir = os.path.join(save_dir, "evolution_plots")
            os.makedirs(plot_dir, exist_ok=True)

        # Collect embeddings (+ optional labels) + texts for metrics
        embs: List[torch.Tensor] = []
        lbls: List[torch.Tensor] = []
        texts_list: List[str] = []  # 🆕 for interpretability metrics
        
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, tuple):
                if len(item) == 2:
                    embs.append(item[0].detach().cpu())
                    lbls.append(item[1].detach().cpu())
                elif len(item) == 3:  # (embedding, label, text)
                    embs.append(item[0].detach().cpu())
                    lbls.append(item[1].detach().cpu())
                    texts_list.append(item[2])
            else:
                t = item.detach() if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.float32)
                embs.append(t.cpu())

        X = torch.stack(embs).to(dev)
        N, D = X.shape
        if D != self.dims[0]:
            raise ValueError(
                f"Input dim = {D}, but dims[0] = {self.dims[0]}. Samakan dims[0] dengan dimensi embedding."
            )
        if N < self.n_clusters:
            raise ValueError(
                f"n_samples ({N}) < n_clusters ({self.n_clusters}). Kurangi n_clusters atau tambah data."
            )

        has_labels = len(lbls) > 0
        Y = torch.stack(lbls).to(dev) if has_labels else None
        has_texts = len(texts_list) > 0

        # class weights (if labels exist; accept one-hot)
        class_w_t: Optional[torch.Tensor] = None
        if has_labels:
            y_np = Y.detach().cpu().numpy()
            if y_np.ndim == 2 and y_np.shape[1] > 1:
                y_np = y_np.argmax(axis=1)
            cw = self.compute_class_weights(y_np)
            class_w_t = torch.tensor([cw.get(i, 1.0) for i in range(2)], dtype=torch.float32, device=dev)

        # optimizer
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

        # Loss functions
        kld_loss = nn.KLDivLoss(reduction="batchmean")
        ce_loss = nn.CrossEntropyLoss(weight=class_w_t) if class_w_t is not None else nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        # Initialize clusters
        print("Initializing cluster centers with k-means.")
        y_pred_last = self._init_clusters_with_kmeans(X)
        
        # Select visualization method based on parameter or auto-select
        valid_methods = ['tsne', 'pca', 'umap']
        if plot_method not in valid_methods:
            print(f"Warning: plot_method '{plot_method}' not in {valid_methods}. Using 'tsne'.")
            plot_method = 'tsne'
        best_viz_method = plot_method
        # Only auto-select if user chose 'tsne' (default) and has texts
        if plot_method == 'tsne' and compute_metrics and has_texts:
            feats_initial = self.extract_feature(X).cpu().numpy()
            best_viz_method, _ = self.select_best_visualization_method(feats_initial)

        # Initial plot with best method (both versions)
        if plot_evolution and plot_dir:
            try:
                feats0 = self.extract_feature(X).cpu().numpy()
                self.plot_cluster_evolution(
                    feats0, y_pred_last, 0, 
                    texts=texts_list if has_texts else None,  # 🆕 pass texts
                    save_dir=plot_dir, 
                    method=best_viz_method,
                    show_plot=False,
                    plot_tfidf_version=has_texts,  # 🆕 only if texts available
                )
            except Exception as e:
                print(f"Warning: initial plot failed: {e}")

        # Logging with comprehensive metrics
        log_path = os.path.join(save_dir, "idec_sentiment_log.csv")
        
        # Extended fieldnames for comprehensive metrics
        log_fieldnames = [
            "iter", "acc_sentiment", "L", "Lr", "Lc", "Ls",
            # Clustering metrics (supervised - if ground truth available)
            "ACC", "NMI", "ARI", "Homogeneity", "Completeness", "V-measure",
            # Clustering metrics (unsupervised)
            "Silhouette",
            # Topic interpretability metrics
            "Topic_Coherence", "Topic_Diversity", "Cluster_Balance",
            "Min_Cluster_Size", "Max_Cluster_Size",
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
                # refresh target distribution
                if ite % update_interval == 0:
                    self.eval()
                    with torch.no_grad():
                        q_list, s_list = [], []
                        for i in range(0, N, batch_size):
                            qb, sb = self(X[i : i + batch_size])
                            q_list.append(qb)
                            s_list.append(sb)
                        q_all = torch.cat(q_list, dim=0)
                        s_all = torch.cat(s_list, dim=0)

                        p_all = self.target_distribution(q_all)

                        y_pred = q_all.argmax(dim=1).cpu().numpy()
                        delta = float((y_pred != y_pred_last).sum() / len(y_pred))
                        y_pred_last = y_pred.copy()

                        # evolution plot with best method (both versions)
                        if plot_evolution and plot_dir and ite > 0 and (ite % plot_interval == 0):
                            try:
                                feats = self.extract_feature(X).cpu().numpy()
                                self.plot_cluster_evolution(
                                    feats, y_pred, ite, 
                                    texts=texts_list if has_texts else None,  # 🆕 pass texts
                                    save_dir=plot_dir,
                                    method=best_viz_method,
                                    show_plot=False,
                                    plot_tfidf_version=has_texts,  # 🆕 create TF-IDF version
                                )
                            except Exception as e:
                                print(f"Warning: plot at iter {ite} failed: {e}")

                        # sentiment accuracy (only during training for monitoring)
                        acc_s = 0.0
                        if has_labels:
                            s_lab = s_all.argmax(dim=1).cpu().numpy()
                            y_true = Y.detach().cpu().numpy()
                            if y_true.ndim == 2 and y_true.shape[1] > 1:
                                y_true = y_true.argmax(axis=1)
                            acc_s = float((s_lab == y_true).mean())
                        
                        # 🆕 Compute comprehensive metrics for logging
                        feats_current = self.extract_feature(X).cpu().numpy()
                        
                        # Clustering metrics (supervised - only if has_labels)
                        clustering_sup = {}
                        if has_labels:
                            y_true_cluster = Y.detach().cpu().numpy()
                            if y_true_cluster.ndim == 2 and y_true_cluster.shape[1] > 1:
                                y_true_cluster = y_true_cluster.argmax(axis=1)
                            clustering_sup = self.compute_clustering_metrics(y_true_cluster, y_pred)
                        
                        # Silhouette score (unsupervised)
                        silhouette_current = self.compute_silhouette_score(feats_current, y_pred)
                        
                        # Topic metrics (only if has_texts)
                        coherence_current = 0.0
                        diversity_current = 0.0
                        coverage_current = {'cluster_balance': 0.0, 'min_cluster_size': 0.0, 'max_cluster_size': 0.0}
                        
                        if has_texts:
                            coherence_dict = self.compute_topic_coherence(texts_list, y_pred, top_n=10)
                            coherence_current = float(np.mean(list(coherence_dict.values()))) if coherence_dict else 0.0
                            diversity_current = self.compute_topic_diversity(texts_list, y_pred, top_n=10)
                            coverage_current = self.compute_topic_coverage(texts_list, y_pred)

                    # log averages with comprehensive metrics
                    avg_L = tot_L / update_interval if iter_count > 0 else 0.0
                    avg_Lr = Lr / update_interval if iter_count > 0 else 0.0
                    avg_Lc = Lc / update_interval if iter_count > 0 else 0.0
                    avg_Ls = Ls / update_interval if iter_count > 0 else 0.0
                    
                    log_row = {
                        "iter": ite,
                        "acc_sentiment": round(acc_s, 5),
                        "L": round(avg_L, 5),
                        "Lr": round(avg_Lr, 5),
                        "Lc": round(avg_Lc, 5),
                        "Ls": round(avg_Ls, 5),
                        # Supervised clustering metrics
                        "ACC": round(clustering_sup.get('ACC', 0.0), 5),
                        "NMI": round(clustering_sup.get('NMI', 0.0), 5),
                        "ARI": round(clustering_sup.get('ARI', 0.0), 5),
                        "Homogeneity": round(clustering_sup.get('Homogeneity', 0.0), 5),
                        "Completeness": round(clustering_sup.get('Completeness', 0.0), 5),
                        "V-measure": round(clustering_sup.get('V-measure', 0.0), 5),
                        # Unsupervised clustering
                        "Silhouette": round(silhouette_current, 5),
                        # Topic metrics
                        "Topic_Coherence": round(coherence_current, 5),
                        "Topic_Diversity": round(diversity_current, 5),
                        "Cluster_Balance": round(coverage_current['cluster_balance'], 5),
                        "Min_Cluster_Size": round(coverage_current['min_cluster_size'], 5),
                        "Max_Cluster_Size": round(coverage_current['max_cluster_size'], 5),
                    }
                    
                    writer.writerow(log_row)
                    
                    # Console output with key metrics
                    print(f"Iter {ite}: Lr={avg_Lr:.5f}, Lc={avg_Lc:.5f}, Ls={avg_Ls:.5f}, Acc={acc_s:.5f}; L={avg_L:.5f}")
                    if clustering_sup:
                        print(f"  Clustering: ACC={clustering_sup['ACC']:.4f}, NMI={clustering_sup['NMI']:.4f}, ARI={clustering_sup['ARI']:.4f}")
                    print(f"  Topic: Coherence={coherence_current:.4f}, Diversity={diversity_current:.4f}, Silhouette={silhouette_current:.4f}")

                    # reset counters
                    tot_L = Lr = Lc = Ls = 0.0
                    iter_count = 0

                    # early stop by cluster stability
                    if ite > 0 and delta < tol:
                        print(f"delta_label {delta:.6f} < tol {tol}. Stop.")
                        break

                    # refresh loader with new p (and labels if any)
                    if has_labels:
                        train_loader = DataLoader(TensorDataset(X, p_all, Y), batch_size=batch_size, shuffle=True)
                    else:
                        train_loader = DataLoader(TensorDataset(X, p_all), batch_size=batch_size, shuffle=True)

                    self.train()

                # train step(s)
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

                    # Forward pass with reconstruction
                    z = self.autoencoder.encode(xb)
                    x_recon = self.autoencoder.decode(z)
                    
                    q = self.clustering(z)
                    s = torch.softmax(self.sentiment(z), dim=1)

                    # Compute all losses
                    recon_loss = mse_loss(x_recon, xb)
                    c_loss = kld_loss((q + 1e-8).log(), pb)
                    s_loss = torch.tensor(0.0, device=dev)
                    if yb is not None:
                        s_loss = ce_loss(s, yb)

                    # Combined loss with reconstruction
                    loss = alpha * recon_loss + gamma * c_loss + eta * s_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track all losses
                    tot_L += float(loss.item())
                    Lr += float(recon_loss.item())
                    Lc += float(c_loss.item())
                    Ls += float(s_loss.item())
                    iter_count += 1

                if ite % save_interval == 0 and ite > 0:
                    self.save_weights(os.path.join(save_dir, f"SEMTGPU_{ite}.weights.pth"))

        # final plot & save with best method (both versions)
        if plot_evolution and plot_dir:
            try:
                feats_f = self.extract_feature(X).cpu().numpy()
                y_final = self.get_cluster_assignments(X)
                self.plot_cluster_evolution(
                    feats_f, y_final, ite, 
                    texts=texts_list if has_texts else None,  # 🆕 pass texts
                    save_dir=plot_dir,
                    method=best_viz_method,
                    show_plot=False,
                    plot_tfidf_version=has_texts,  # 🆕 create TF-IDF version
                )
            except Exception as e:
                print(f"Warning: final plot failed: {e}")

        self.save_weights(os.path.join(save_dir, "SEMTGPU_final.weights.pth"))

        # 🆕 Final evaluation with comprehensive metrics
        self.eval()
        with torch.no_grad():
            q_all, s_all = self(X)
            y_pred_cluster = q_all.argmax(dim=1).cpu().numpy()
            y_pred_sentiment = s_all.argmax(dim=1).cpu().numpy()
            
            metrics = {}
            
            # Sentiment metrics (if labels available)
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
            
            # 🆕 Comprehensive interpretability metrics
            if compute_metrics and has_texts:
                feats_final = self.extract_feature(X).cpu().numpy()
                
                # Pass true labels for clustering evaluation (if available)
                true_cluster_labels = None
                if has_labels:
                    true_cluster_labels = Y.detach().cpu().numpy()
                    if true_cluster_labels.ndim == 2 and true_cluster_labels.shape[1] > 1:
                        true_cluster_labels = true_cluster_labels.argmax(axis=1)
                
                interp_metrics = self.compute_comprehensive_metrics(
                    feats_final,
                    texts_list,
                    y_pred_cluster,
                    true_labels=true_cluster_labels,
                    top_n_words=10,
                )
                metrics['interpretability'] = interp_metrics
                
                # Save metrics to file
                metrics_path = os.path.join(save_dir, "interpretability_metrics.json")
                import json
                with open(metrics_path, 'w') as f:
                    # Convert to JSON-serializable format
                    json_metrics = {}
                    for k, v in interp_metrics.items():
                        if isinstance(v, dict):
                            json_metrics[k] = {str(kk): vv for kk, vv in v.items()}
                        else:
                            json_metrics[k] = v
                    json.dump(json_metrics, f, indent=2)
                print(f"✓ Interpretability metrics saved to: {metrics_path}")
            
            if has_labels:
                return y_pred_cluster, s_all.cpu().numpy(), metrics
            
            return y_pred_cluster

    # -------------------------
    # Helpers
    # -------------------------
    def _init_clusters_with_kmeans(
        self, all_embeddings: torch.Tensor, n_init: int = 20, random_state: int = 42
    ) -> np.ndarray:
        """
        Run KMeans on encoded features and set cluster centers.
        """
        dev = next(self.parameters()).device
        N = all_embeddings.size(0)
        if N < self.n_clusters:
            raise ValueError(
                f"n_samples ({N}) < n_clusters ({self.n_clusters}). Tambah data atau kurangi n_clusters."
            )
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

    # -------------------------
    # Topic & Text Utilities
    # -------------------------
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
        texts: Optional[List[str]] = None,  # 🆕 for TF-IDF annotation
        save_dir: str = "./results/fnnjst",
        method: str = "tsne",
        figsize: Tuple[int, int] = (6, 6),
        point_size: int = 20,
        alpha: float = 0.7,
        save_plot: bool = True,
        show_plot: bool = False,
        plot_tfidf_version: bool = True,  # 🆕 create annotated version
        max_keywords_per_cluster: int = 3,  # 🆕 max keywords to show
        keyword_min_score: float = 0.3,  # 🆕 min TF-IDF score threshold
    ):
        """
        Plot cluster evolution with automatic or manual method selection.
        
        Creates TWO plots per epoch:
        1. Regular scatter plot (clean visualization)
        2. TF-IDF annotated plot (keywords at cluster centroids)
        
        Args:
            embeddings: High-dimensional embeddings
            cluster_assignments: Cluster labels
            epoch: Current epoch number
            texts: Optional list of texts for TF-IDF keyword extraction
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            plot_tfidf_version: If True and texts provided, create annotated plot
            max_keywords_per_cluster: Max keywords to display per cluster
            keyword_min_score: Minimum TF-IDF score to display (filters common words)
        """
        if isinstance(embeddings, torch.Tensor):
            emb = embeddings.detach().cpu().numpy()
        else:
            emb = embeddings

        n = emb.shape[0]
        if n < 3:
            print(f"Skip plot at epoch {epoch}: n_samples={n} < 3")
            return None

        # Dimensionality reduction
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

        # Color setup
        uniq = np.unique(cluster_assignments)
        k = len(uniq)
        if k <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
        elif k <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, k))

        # ====================================================================
        # PLOT 1: Regular scatter plot (clean)
        # ====================================================================
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

        # ====================================================================
        # PLOT 2: TF-IDF Annotated plot (with keywords)
        # ====================================================================
        if plot_tfidf_version and texts is not None and len(texts) == len(cluster_assignments):
            fig2, ax2 = plt.subplots(figsize=figsize)
            
            # Draw scatter points first
            for i, cid in enumerate(uniq):
                mask = cluster_assignments == cid
                pts = emb_2d[mask]
                ax2.scatter(pts[:, 0], pts[:, 1], c=[colors[i]], marker="x", s=point_size, alpha=alpha, label=f"Cluster {cid}")
            
            # Extract TF-IDF keywords
            try:
                tfidf_keywords = self.extract_tfidf_keywords(
                    texts, 
                    cluster_assignments, 
                    top_n=max_keywords_per_cluster * 3,  # Get more, filter later
                    max_features=5000
                )
                
                # Calculate cluster centroids and annotate
                for cid in uniq:
                    mask = cluster_assignments == cid
                    centroid = emb_2d[mask].mean(axis=0)
                    
                    # Filter keywords by score threshold
                    if cid in tfidf_keywords:
                        keywords = tfidf_keywords[cid]
                        
                        # 🆕 Smart filtering logic:
                        # 1. Remove keywords with low TF-IDF score (too common)
                        # 2. Calculate cluster-specific threshold if needed
                        
                        # Get score distribution for this cluster
                        scores = [score for _, score in keywords]
                        if len(scores) > 0:
                            # Adaptive threshold: use provided min or 75th percentile
                            adaptive_threshold = max(
                                keyword_min_score,
                                np.percentile(scores, 75) if len(scores) >= 4 else 0
                            )
                            
                            # Filter keywords
                            filtered_keywords = [
                                (word, score) for word, score in keywords 
                                if score >= adaptive_threshold
                            ][:max_keywords_per_cluster]
                            
                            # Check if keywords are too generic (appear in too many clusters)
                            # Build vocabulary across all clusters
                            all_words_count = {}
                            for other_cid, other_keywords in tfidf_keywords.items():
                                for word, _ in other_keywords[:10]:
                                    all_words_count[word] = all_words_count.get(word, 0) + 1
                            
                            # Remove words that appear in > 50% of clusters (too generic)
                            max_cluster_frequency = max(2, k * 0.5)
                            final_keywords = [
                                (word, score) for word, score in filtered_keywords
                                if all_words_count.get(word, 0) <= max_cluster_frequency
                            ]
                            
                            # If after filtering we have no keywords, show top 1 anyway
                            if not final_keywords and filtered_keywords:
                                final_keywords = [filtered_keywords[0]]
                            
                            # Create annotation text
                            if final_keywords:
                                # Format: "word1\nword2\nword3"
                                keyword_text = "\n".join([
                                    f"{word}" for word, _ in final_keywords
                                ])
                                
                                # Annotate at centroid
                                ax2.annotate(
                                    keyword_text,
                                    xy=centroid,
                                    xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=8,
                                    fontweight='bold',
                                    color=colors[list(uniq).index(cid)],
                                    bbox=dict(
                                        boxstyle='round,pad=0.5',
                                        facecolor='white',
                                        edgecolor=colors[list(uniq).index(cid)],
                                        alpha=0.8,
                                        linewidth=1.5
                                    ),
                                    ha='left',
                                    va='bottom',
                                    zorder=1000
                                )
                                
                                # Mark centroid
                                ax2.scatter(
                                    [centroid[0]], [centroid[1]], 
                                    c=[colors[list(uniq).index(cid)]], 
                                    marker='*', 
                                    s=200, 
                                    edgecolors='black',
                                    linewidths=1,
                                    zorder=999
                                )
                
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
        plot_type: str = "both"  # 🆕 "regular", "tfidf", or "both"
    ):
        """
        Create grid of evolution plots.
        
        Args:
            save_dir: Directory with evolution plots
            epochs_to_show: Specific epochs to display (None = all)
            grid_cols: Number of columns in grid
            figsize: Figure size
            plot_type: "regular" (clean plots), "tfidf" (annotated), or "both" (side-by-side)
        """
        import matplotlib.image as mpimg

        if plot_type == "both":
            # Create two separate grids
            fig_regular = self._create_single_grid(save_dir, epochs_to_show, grid_cols, figsize, suffix="")
            fig_tfidf = self._create_single_grid(save_dir, epochs_to_show, grid_cols, figsize, suffix="_tfidf")
            return fig_regular, fig_tfidf
        elif plot_type == "tfidf":
            return self._create_single_grid(save_dir, epochs_to_show, grid_cols, figsize, suffix="_tfidf")
        else:  # regular
            return self._create_single_grid(save_dir, epochs_to_show, grid_cols, figsize, suffix="")
    
    def _create_single_grid(
        self,
        save_dir: str,
        epochs_to_show: Optional[List[int]],
        grid_cols: int,
        figsize: Tuple[int, int],
        suffix: str = ""  # "" for regular, "_tfidf" for annotated
    ):
        """Helper to create a single grid of plots."""
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