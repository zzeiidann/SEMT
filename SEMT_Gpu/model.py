"""
SEMT-GPU: Feedforward Neural Network (PyTorch) for
Joint Sentiment Analysis and Topic Clustering

- Autoencoder for representation learning
- Student-t Clustering layer (DEC-style) for soft assignments
- Sentiment head (binary) with class-weighted CE
- Robust training loop (fit) dengan target distribution
- Visualizations: t-SNE / UMAP (auto-perplexity / auto-n_neighbors)
"""

from __future__ import annotations

import os
import csv
import glob
import re
from collections import Counter
from typing import Iterable, List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
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
    ind = linear_assignment(w.max() - w)
    return float(sum(w[i, j] for i, j in ind) / y_pred.size)


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
    """

    def __init__(self, dims: List[int], n_clusters: int = 10, alpha: float = 1.0) -> None:
        super().__init__()

        assert len(dims) >= 2, "dims must be [input_dim, ..., latent_dim]"
        self.dims = dims
        self.n_clusters = int(n_clusters)
        self.alpha = float(alpha)

        # Core
        self.autoencoder = Autoencoder(dims)
        self.clustering = ClusteringLayer(n_clusters=n_clusters, input_dim=dims[-1], alpha=alpha)

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
        print(f"✓ Saved weights to {path}")

    # -------------------------
    # Autoencoder pretrain
    # -------------------------
    def pretrain_autoencoder(
        self,
        dataset: Iterable,
        batch_size: int = 256,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> None:
        """
        Pretrain autoencoder with MSE reconstruction.
        Dataset items can be: embedding tensor OR (embedding, label)
        """
        print("=" * 60)
        print("Pretraining Autoencoder")
        print("=" * 60)

        # Collect embeddings
        embs: List[torch.Tensor] = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, tuple) and len(item) >= 1:
                embs.append(item[0].detach().cpu())
            else:
                t = item.detach() if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.float32)
                embs.append(t.cpu())

        X = torch.stack(embs)
        n, d = X.shape
        if d != self.dims[0]:
            raise ValueError(
                f"Input dimension mismatch: embeddings dim = {d}, but dims[0] = {self.dims[0]}. "
                f"Set dims[0] to {d}."
            )

        X = X.to(next(self.parameters()).device)
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

        opt = optim.Adam(self.autoencoder.parameters(), lr=lr)
        crit = nn.MSELoss()
        self.autoencoder.train()

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

        self.save_weights("pretrained_ae.weights.pth")
        print("✓ Autoencoder pretraining complete")

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
    # Training (fit)
    # -------------------------
    def fit(
        self,
        dataset: Iterable,
        gamma: float = 0.7,
        eta: float = 1.0,
        optimizer_type: str = "sgd",
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        tol: float = 1e-3,
        update_interval: int = 140,
        batch_size: int = 128,
        maxiter: int = int(2e4),
        save_dir: str = "./results/fnnjst",
        plot_evolution: bool = True,
        plot_interval: Optional[int] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Joint training (DEC + Sentiment). Uses target distribution refresh every update_interval iters.
        """
        print("Update interval", update_interval)
        dev = next(self.parameters()).device
        maxiter = int(maxiter)
        plot_interval = int(plot_interval) if plot_interval is not None else update_interval

        os.makedirs(save_dir, exist_ok=True)
        plot_dir = None
        if plot_evolution:
            plot_dir = os.path.join(save_dir, "evolution_plots")
            os.makedirs(plot_dir, exist_ok=True)

        # Collect embeddings (+ optional labels)
        embs: List[torch.Tensor] = []
        lbls: List[torch.Tensor] = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, tuple) and len(item) == 2:
                embs.append(item[0].detach().cpu())
                lbls.append(item[1].detach().cpu())
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

        kld_loss = nn.KLDivLoss(reduction="batchmean")
        ce_loss = nn.CrossEntropyLoss(weight=class_w_t) if class_w_t is not None else nn.CrossEntropyLoss()

        # Initialize clusters
        print("Initializing cluster centers with k-means.")
        y_pred_last = self._init_clusters_with_kmeans(X)

        # Initial plot (robust)
        if plot_evolution and plot_dir:
            try:
                feats0 = self.extract_feature(X).cpu().numpy()
                self.plot_cluster_evolution(feats0, y_pred_last, 0, save_dir=plot_dir, show_plot=False)
            except Exception as e:
                print(f"Warning: initial plot failed: {e}")

        # Logging
        log_path = os.path.join(save_dir, "idec_sentiment_log.csv")
        with open(log_path, "w", newline="") as logfile:
            writer = csv.DictWriter(
                logfile, fieldnames=["iter", "acc_cluster", "nmi", "ari", "acc_sentiment", "L", "Lc", "Ls"]
            )
            writer.writeheader()

            save_interval = max(1, (max(1, N // batch_size)) * 5)
            train_loader: Optional[DataLoader] = None
            self.train()
            iter_count = 0
            tot_L = Lc = Ls = 0.0

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

                        # evolution plot
                        if plot_evolution and plot_dir and ite > 0 and (ite % plot_interval == 0):
                            try:
                                feats = self.extract_feature(X).cpu().numpy()
                                self.plot_cluster_evolution(feats, y_pred, ite, save_dir=plot_dir, show_plot=False)
                            except Exception as e:
                                print(f"Warning: plot at iter {ite} failed: {e}")

                        # sentiment accuracy
                        acc_s = 0.0
                        if has_labels:
                            s_lab = s_all.argmax(dim=1).cpu().numpy()
                            y_true = Y.detach().cpu().numpy()
                            if y_true.ndim == 2 and y_true.shape[1] > 1:
                                y_true = y_true.argmax(axis=1)
                            acc_s = float((s_lab == y_true).mean())

                    # log averages
                    avg_L = tot_L / update_interval if iter_count > 0 else 0.0
                    avg_Lc = Lc / update_interval if iter_count > 0 else 0.0
                    avg_Ls = Ls / update_interval if iter_count > 0 else 0.0
                    writer.writerow(
                        {
                            "iter": ite,
                            "acc_cluster": 0,
                            "nmi": 0,
                            "ari": 0,
                            "acc_sentiment": round(acc_s, 5),
                            "L": round(avg_L, 5),
                            "Lc": round(avg_Lc, 5),
                            "Ls": round(avg_Ls, 5),
                        }
                    )
                    print(f"Iter {ite}: Lc={avg_Lc:.5f}, Ls={avg_Ls:.5f}, Acc={acc_s:.5f}; L={avg_L:.5f}")

                    # reset counters
                    tot_L = Lc = Ls = 0.0
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

                    q, s = self(xb)
                    c_loss = kld_loss((q + 1e-8).log(), pb)
                    s_loss = torch.tensor(0.0, device=dev)
                    if yb is not None:
                        s_loss = ce_loss(s, yb)

                    loss = gamma * c_loss + eta * s_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tot_L += float(loss.item())
                    Lc += float(c_loss.item())
                    Ls += float(s_loss.item())
                    iter_count += 1

                if ite % save_interval == 0 and ite > 0:
                    self.save_weights(os.path.join(save_dir, f"SEMTGPU_{ite}.weights.pth"))

        # final plot & save
        if plot_evolution and plot_dir:
            try:
                feats_f = self.extract_feature(X).cpu().numpy()
                y_final = self.get_cluster_assignments(X)
                self.plot_cluster_evolution(feats_f, y_final, ite, save_dir=plot_dir, show_plot=False)
            except Exception as e:
                print(f"Warning: final plot failed: {e}")

        self.save_weights(os.path.join(save_dir, "SEMTGPU_final.weights.pth"))

        # return predictions
        self.eval()
        with torch.no_grad():
            q_all, s_all = self(X)
            y_pred = q_all.argmax(dim=1).cpu().numpy()
            if has_labels:
                return y_pred, s_all.cpu().numpy()
            return y_pred

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
        save_dir: str = "./results/fnnjst",
        method: str = "tsne",
        figsize: Tuple[int, int] = (6, 6),
        point_size: int = 20,
        alpha: float = 0.7,
        save_plot: bool = True,
        show_plot: bool = False,
    ):
        if isinstance(embeddings, torch.Tensor):
            emb = embeddings.detach().cpu().numpy()
        else:
            emb = embeddings

        n = emb.shape[0]
        if n < 3:
            print(f"Skip plot at epoch {epoch}: n_samples={n} < 3")
            return None

        if method.lower() == "tsne":
            from sklearn.manifold import TSNE

            perplexity = max(2, min(30, n - 1))
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
            emb_2d = reducer.fit_transform(emb)
        elif method.lower() == "umap":
            try:
                import umap

                n_neighbors = max(2, min(15, n - 1))
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
                emb_2d = reducer.fit_transform(emb)
            except Exception:
                from sklearn.manifold import TSNE

                perplexity = max(2, min(30, n - 1))
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
                emb_2d = reducer.fit_transform(emb)
        else:
            raise ValueError("method must be 'tsne' or 'umap'")

        fig, ax = plt.subplots(figsize=figsize)
        uniq = np.unique(cluster_assignments)
        k = len(uniq)
        if k <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
        elif k <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, k))

        for i, cid in enumerate(uniq):
            mask = cluster_assignments == cid
            pts = emb_2d[mask]
            ax.scatter(pts[:, 0], pts[:, 1], c=[colors[i]], marker="x", s=point_size, alpha=alpha, label=f"Cluster {cid}")

        ax.set_title(f"Epoch {epoch}", fontsize=14, fontweight="bold")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_facecolor("white")
        plt.tight_layout()

        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"cluster_evolution_epoch_{epoch}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
            print(f"✓ Saved cluster plot: {out}")
        if show_plot:
            plt.show()
        else:
            plt.close()
        return fig

    def create_evolution_grid(
        self, save_dir: str = "./results/fnnjst", epochs_to_show: Optional[List[int]] = None, grid_cols: int = 3, figsize: Tuple[int, int] = (15, 10)
    ):
        import matplotlib.image as mpimg

        files = glob.glob(os.path.join(save_dir, "cluster_evolution_epoch_*.png"))
        if not files:
            print("No cluster evolution plots found")
            return None

        epoch_files = []
        for f in files:
            m = re.search(r"epoch_(\d+)\.png", f)
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

        for i, (ep, fp) in enumerate(epoch_files):
            img = mpimg.imread(fp)
            axes[i].imshow(img)
            axes[i].set_title(f"Epoch {ep}", fontsize=12, fontweight="bold")
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        out = os.path.join(save_dir, "cluster_evolution_grid.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"✓ Saved evolution grid to {out}")
        return fig
