"""
SEMT-GPU: Joint Sentiment Analysis & Topic Clustering (PyTorch, GPU-ready)

Fitur:
- Autoencoder (fully-connected) utk ekstraksi fitur (pretraining MSE).
- ClusteringLayer (Student's t-distribution) -> soft assignment q.
- Joint training ala DEC: target distribution p, KLDiv utk cluster
  + CrossEntropy utk sentiment (dengan opsi class weight).
- Inisialisasi cluster via K-Means.
- Util: prediksi, analisis cluster, visualisasi evolusi cluster.

Catatan:
- Pastikan dimensi `dims` menurun ke latent, mis. [768, 512, 256, 32].
- Default BERT: "indolem/indobert-base-uncased".
"""

from __future__ import annotations

import os
import csv
import glob
import re
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModel


# ------------------------- Util umum -------------------------

DEFAULT_BERT_MODEL = "indolem/indobert-base-uncased"


def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Clustering accuracy via Hungarian (best label permutation).
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    r_ind, c_ind = linear_sum_assignment(w.max() - w)
    return float(w[r_ind, c_ind].sum() / y_pred.size)


def to_device(x: Union[torch.Tensor, np.ndarray], device: torch.device) -> torch.Tensor:
    """
    Pastikan x menjadi torch.Tensor di device yang benar.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


# ------------------------------ Layers ------------------------------

class ClusteringLayer(nn.Module):
    """
    Student's t-distribution soft assignment (mirip t-SNE / DEC).
    q_ij ∝ (1 + ||z_i - μ_j||^2 / α)^(-(α+1)/2)
    """

    def __init__(self, n_clusters: int, input_dim: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.empty(n_clusters, input_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.clusters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D), clusters: (K, D) -> pairwise dist^2: (N, K)
        x_norm = (x ** 2).sum(dim=1, keepdim=True)                 # (N, 1)
        c_norm = (self.clusters ** 2).sum(dim=1, keepdim=True).T   # (1, K)
        dist2 = x_norm + c_norm - 2.0 * (x @ self.clusters.T)      # (N, K)

        q = 1.0 / (1.0 + dist2 / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q


class Autoencoder(nn.Module):
    """
    Fully-connected AE simetris. Pastikan dims menurun ke latent.
    Contoh dims: [768, 512, 256, 32]
    """

    def __init__(self, dims: Sequence[int], act: str = "relu") -> None:
        super().__init__()
        assert len(dims) >= 2, "dims minimal [input_dim, latent_dim]"
        self.dims = list(dims)

        act_map = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        self.activation = act_map.get(act.lower(), nn.ReLU())

        # Encoder: (d0->d1 act) ... (d_{L-2}->d_{L-1} no-act)
        enc: List[nn.Module] = []
        for i in range(len(self.dims) - 2):
            enc += [nn.Linear(self.dims[i], self.dims[i + 1]), self.activation]
        enc += [nn.Linear(self.dims[-2], self.dims[-1])]
        self.encoder = nn.Sequential(*enc)

        # Decoder simetris: (d_{L-1}->d_{L-2} act) ... (d1->d0 no-act)
        dec: List[nn.Module] = []
        for i in range(len(self.dims) - 1, 1, -1):
            dec += [nn.Linear(self.dims[i], self.dims[i - 1]), self.activation]
        dec += [nn.Linear(self.dims[1], self.dims[0])]
        self.decoder = nn.Sequential(*dec)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


# ------------------------------- Main model -------------------------------

class SEMTGPU(nn.Module):
    """
    Joint Sentiment + Topic Clustering (DEC + classifier).
    """

    def __init__(self, dims: Sequence[int], n_clusters: int = 10, alpha: float = 1.0) -> None:
        super().__init__()
        self.dims = list(dims)
        self.n_clusters = n_clusters
        self.alpha = alpha

        self.autoencoder = Autoencoder(self.dims)
        self.clustering = ClusteringLayer(n_clusters, self.dims[-1], alpha)

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(self.dims[-1], 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2),  # 0=negative, 1=positive
        )

        self.class_labels: Dict[int, str] = {0: "negative", 1: "positive"}
        self.topic_mapping: Dict[int, str] = {}
        self.stop_words: set[str] = set()

        # init classifier weights
        for m in self.sentiment_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # --------------------------- Core forward/predict ---------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.autoencoder.encode(x)
        q = self.clustering(z)
        s_logits = self.sentiment_classifier(z)
        s = torch.softmax(s_logits, dim=1)
        return q, s

    @torch.no_grad()
    def extract_feature(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        device = next(self.parameters()).device
        x = to_device(x, device)
        self.eval()
        return self.autoencoder.encode(x)

    @torch.no_grad()
    def predict_clusters(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        device = next(self.parameters()).device
        x = to_device(x, device)
        self.eval()
        q, _ = self(x)
        return q.argmax(dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_sentiment(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        device = next(self.parameters()).device
        x = to_device(x, device)
        self.eval()
        _, s = self(x)
        return s.argmax(dim=1).cpu().numpy()

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[str, List[str], torch.Tensor, np.ndarray],
        bert_model: Optional[Union[str, AutoModel]] = None,
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Prediksi sentiment & topic dari teks (via BERT) atau embedding (tensor).
        """
        device = next(self.parameters()).device
        self.eval()

        # handle text
        if isinstance(inputs, str):
            inputs = [inputs]

        if isinstance(inputs, list) and inputs and isinstance(inputs[0], str):
            model_name = bert_model if isinstance(bert_model, str) else DEFAULT_BERT_MODEL
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokens = tokenizer(
                inputs, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(device)

            if isinstance(bert_model, AutoModel):
                bert = bert_model.to(device)
            else:
                bert = AutoModel.from_pretrained(model_name).to(device)

            outputs = bert(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS
        else:
            embeddings = to_device(inputs, device)

        q, s = self(embeddings)

        y_topic = q.argmax(dim=1).cpu().numpy()
        y_sent = s.argmax(dim=1).cpu().numpy()

        results: List[Dict[str, Union[str, int]]] = []
        for i in range(len(y_sent)):
            topic_id = int(y_topic[i])
            results.append(
                {
                    "sentiment": self.class_labels[int(y_sent[i])],
                    "topic": self.topic_mapping.get(topic_id, topic_id),
                }
            )
        return results

    # ----------------------------- Weights I/O -----------------------------

    def save_weights(self, path: str) -> None:
        torch.save({"model_state_dict": self.state_dict()}, path)
        print(f"✓ Saved weights to {path}")

    def load_weights(self, path: str, map_location: Optional[torch.device] = None) -> None:
        ckpt = torch.load(path, map_location=map_location or torch.device("cpu"))
        self.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Loaded weights from {path}")

    # ------------------------- Pretraining Autoencoder -------------------------

    def pretrain_autoencoder(
        self,
        dataset: Sequence[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
        batch_size: int = 256,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> None:
        """
        Pretrain AE dengan MSE reconstruction.
        `dataset`: list/sequence berisi tensor embedding atau (embedding, label).
        """
        device = next(self.parameters()).device
        print("=" * 60)
        print("Pretraining Autoencoder")
        print("=" * 60)

        embs: List[torch.Tensor] = []
        for item in dataset:
            if isinstance(item, tuple) and len(item) == 2:
                embs.append(item[0].detach().cpu())
            else:
                embs.append((item.detach() if isinstance(item, torch.Tensor) else torch.tensor(item)).cpu())
        X = torch.stack(embs).to(device)

        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.autoencoder.to(device).train()
        for epoch in range(epochs):
            total = 0.0
            for (xb,) in tqdm(dl, desc=f"AE Epoch {epoch+1}/{epochs}", leave=False):
                xb = xb.to(device)
                _, xhat = self.autoencoder(xb)
                loss = criterion(xhat, xb)

                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"[AE] epoch {epoch+1:03d} | loss: {total/len(dl):.6f}")

        self.save_weights("pretrained_ae.weights.pth")
        print("✓ Autoencoder pretraining completed")

    # --------------------------- Clustering utilities ---------------------------

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary target distribution p dari soft assignment q.
        """
        weight = (q ** 2) / q.sum(dim=0, keepdim=True)
        p = (weight.t() / weight.sum(dim=1, keepdim=True)).t()
        return p

    @staticmethod
    def compute_class_weights(y: Union[np.ndarray, torch.Tensor]) -> Dict[int, float]:
        """
        Balanced class weights utk imbalanced sentiment.
        Terima label indeks (bukan one-hot).
        """
        if isinstance(y, torch.Tensor):
            y_idx = y.detach().cpu().numpy()
        else:
            y_idx = np.asarray(y)

        # kalau one-hot, konversi dulu
        if y_idx.ndim == 2 and y_idx.shape[1] > 1:
            y_idx = y_idx.argmax(axis=1)

        classes, counts = np.unique(y_idx, return_counts=True)
        total = len(y_idx)
        ncls = len(classes)
        weights = {int(c): float(total / (ncls * cnt)) for c, cnt in zip(classes, counts)}
        return weights

    def _init_clusters_with_kmeans(
        self, all_embeddings: torch.Tensor, n_init: int = 20, random_state: int = 42
    ) -> np.ndarray:
        """
        Inisialisasi centroid clustering layer via KMeans pada fitur AE.
        """
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            feats = self.extract_feature(all_embeddings).cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=n_init, random_state=random_state)
        y_pred = kmeans.fit_predict(feats)

        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        self.clustering.clusters.data = centers
        return y_pred

    # --------------------------------- Training ---------------------------------

    def fit(
        self,
        dataset: Sequence[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
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
        High-level training loop (DEC + Sentiment). Gunakan ini alih-alih .train().
        """
        device = next(self.parameters()).device
        print("Update interval", update_interval)

        os.makedirs(save_dir, exist_ok=True)
        plot_dir = None
        if plot_evolution:
            plot_dir = os.path.join(save_dir, "evolution_plots")
            os.makedirs(plot_dir, exist_ok=True)
            if plot_interval is None:
                plot_interval = update_interval

        # kumpulkan embeddings & (opsional) labels
        embs: List[torch.Tensor] = []
        lbls: List[torch.Tensor] = []
        for item in dataset:
            if isinstance(item, tuple) and len(item) == 2:
                embs.append(item[0].detach().cpu())
                lbls.append(item[1].detach().cpu())
            else:
                t = item.detach() if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.float32)
                embs.append(t.cpu())

        X = torch.stack(embs).to(device)
        has_labels = len(lbls) > 0
        Y = torch.stack(lbls).to(device) if has_labels else None

        # class weights (kalau ada label)
        if has_labels:
            y_np = Y.detach().cpu().numpy()
            cw = self.compute_class_weights(y_np)
            class_w = torch.tensor([cw.get(i, 1.0) for i in range(2)], dtype=torch.float32, device=device)
        else:
            class_w = None

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
            raise ValueError(f"Unsupported optimizer '{optimizer_type}'.")
        optimizer = opt_map[optimizer_type.lower()]()

        kld_loss = nn.KLDivLoss(reduction="batchmean")
        ce_loss = nn.CrossEntropyLoss(weight=class_w) if class_w is not None else nn.CrossEntropyLoss()

        # init cluster
        print("Initializing cluster centers with k-means.")
        y_pred_last = self._init_clusters_with_kmeans(X)

        # plot awal
        if plot_evolution and plot_dir:
            try:
                feats0 = self.extract_feature(X).cpu().numpy()
                self.plot_cluster_evolution(feats0, y_pred_last, 0, save_dir=plot_dir, show_plot=False)
            except Exception as e:
                print(f"Warning: initial plot failed: {e}")

        # logging
        log_path = os.path.join(save_dir, "idec_sentiment_log.csv")
        with open(log_path, "w", newline="") as logfile:
            writer = csv.DictWriter(
                logfile,
                fieldnames=["iter", "acc_cluster", "nmi", "ari", "acc_sentiment", "L", "Lc", "Ls"],
            )
            writer.writeheader()

            # loader awal (akan di-update tiap interval)
            train_loader: Optional[DataLoader] = None
            save_interval = max(1, (len(X) // batch_size) * 5)

            self.train()  # <- tetap pakai .train() bawaan utk mode toggle
            iter_count = 0
            tot_loss = cl_loss = se_loss = 0.0

            for ite in range(maxiter):
                # update target distribution
                if ite % update_interval == 0:
                    self.eval()
                    with torch.no_grad():
                        q_list, s_list = [], []
                        for i in range(0, len(X), batch_size):
                            xb = X[i : i + batch_size]
                            q, s = self(xb)
                            q_list.append(q)
                            s_list.append(s)
                        q_all = torch.cat(q_list, dim=0)
                        s_all = torch.cat(s_list, dim=0)

                        p_all = self.target_distribution(q_all)

                        y_pred = q_all.argmax(dim=1).cpu().numpy()
                        delta = float((y_pred != y_pred_last).sum() / len(y_pred))
                        y_pred_last = y_pred.copy()

                        # plot tiap beberapa interval
                        if plot_evolution and plot_dir and ite > 0 and ite % plot_interval == 0:
                            try:
                                feats = self.extract_feature(X).cpu().numpy()
                                self.plot_cluster_evolution(feats, y_pred, ite, save_dir=plot_dir, show_plot=False)
                            except Exception as e:
                                print(f"Warning: plot at iter {ite} failed: {e}")

                        # sentiment acc (kalau ada label)
                        acc_sent = 0.0
                        if has_labels:
                            s_label = s_all.argmax(dim=1).cpu().numpy()
                            y_true = Y.detach().cpu().numpy()
                            if y_true.ndim == 2 and y_true.shape[1] > 1:
                                y_true = y_true.argmax(axis=1)
                            acc_sent = float((s_label == y_true).mean())

                    # log
                    avg_L = tot_loss / update_interval if iter_count > 0 else 0.0
                    avg_Lc = cl_loss / update_interval if iter_count > 0 else 0.0
                    avg_Ls = se_loss / update_interval if iter_count > 0 else 0.0

                    writer.writerow(
                        {
                            "iter": ite,
                            "acc_cluster": 0,
                            "nmi": 0,
                            "ari": 0,
                            "acc_sentiment": round(acc_sent, 5),
                            "L": round(avg_L, 5),
                            "Lc": round(avg_Lc, 5),
                            "Ls": round(avg_Ls, 5),
                        }
                    )
                    print(f"Iter {ite}: Lc={avg_Lc:.5f}, Ls={avg_Ls:.5f}, Acc={acc_sent:.5f}; L={avg_L:.5f}")

                    # reset akumulasi
                    tot_loss = cl_loss = se_loss = 0.0
                    iter_count = 0

                    # early stop
                    if ite > 0 and delta < tol:
                        print(f"delta_label {delta:.6f} < tol {tol}. Stop.")
                        break

                    # update loader dengan p (dan y kalau ada)
                    if has_labels:
                        train_loader = DataLoader(TensorDataset(X, p_all, Y), batch_size=batch_size, shuffle=True)
                    else:
                        train_loader = DataLoader(TensorDataset(X, p_all), batch_size=batch_size, shuffle=True)

                    self.train()

                # -------------- step training --------------
                assert train_loader is not None, "train_loader belum terinisialisasi"
                for batch in tqdm(train_loader, desc=f"Train {ite}", leave=False):
                    if has_labels and len(batch) == 3:
                        xb, pb, yb = batch
                        yb = yb.to(device)
                        # kalau one-hot, ubah ke index
                        if yb.dim() > 1 and yb.shape[1] > 1:
                            yb = yb.argmax(dim=1)
                    else:
                        xb, pb = batch
                        yb = None

                    xb = xb.to(device)
                    pb = pb.to(device)

                    q, s = self(xb)
                    c_loss = kld_loss((q + 1e-8).log(), pb)
                    s_loss = torch.tensor(0.0, device=device)
                    if yb is not None:
                        s_loss = ce_loss(s, yb.long())

                    loss = gamma * c_loss + eta * s_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tot_loss += float(loss.item())
                    cl_loss += float(c_loss.item())
                    se_loss += float(s_loss.item())
                    iter_count += 1

                # simpan berkala
                if ite % save_interval == 0 and ite > 0:
                    self.save_weights(os.path.join(save_dir, f"SEMTGPU_{ite}.weights.pth"))

        # plot final
        if plot_evolution and plot_dir:
            try:
                feats_f = self.extract_feature(X).cpu().numpy()
                y_final = self.get_cluster_assignments(X)
                self.plot_cluster_evolution(feats_f, y_final, ite, save_dir=plot_dir, show_plot=False)
            except Exception as e:
                print(f"Warning: final plot failed: {e}")

        # simpan final
        self.save_weights(os.path.join(save_dir, "SEMTGPU_final.weights.pth"))

        # return prediksi
        self.eval()
        with torch.no_grad():
            q_all, s_all = self(X)
            y_pred = q_all.argmax(dim=1).cpu().numpy()
            if has_labels:
                return y_pred, s_all.cpu().numpy()
            return y_pred

    # Backward-compat (boleh dihapus nanti setelah semua callsite pindah ke .fit)
    clustering_with_sentiment = fit

    # ------------------------------ Analysis utils ------------------------------

    @torch.no_grad()
    def get_cluster_assignments(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        device = next(self.parameters()).device
        x = to_device(x, device)
        self.eval()
        q, _ = self(x)
        return q.argmax(dim=1).cpu().numpy()

    def set_stop_words(self, stop_words: Union[List[str], set, Iterable[str]]) -> "SEMTGPU":
        self.stop_words = set(stop_words)
        return self

    def map_texts_to_clusters(
        self, texts: List[str], cluster_assignments: np.ndarray
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[Tuple[str, int]]]]:
        clusters: Dict[int, List[str]] = {}
        n = min(len(texts), len(cluster_assignments))
        for i in range(n):
            cid = int(cluster_assignments[i])
            clusters.setdefault(cid, []).append(texts[i])

        cluster_common: Dict[int, List[Tuple[str, int]]] = {}
        for cid, tlist in clusters.items():
            words = " ".join(tlist).lower().split()
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            counts: Dict[str, int] = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            common = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
            cluster_common[cid] = common

        return clusters, cluster_common

    def analyze_clusters(self, x: Union[torch.Tensor, np.ndarray], texts: List[str]) -> pd.DataFrame:
        cids = self.get_cluster_assignments(x)
        text_clusters, cluster_words = self.map_texts_to_clusters(texts, cids)
        rows = []
        for cid, words in sorted(cluster_words.items()):
            rows.append(
                {
                    "Cluster": cid,
                    "Common Words": ", ".join([f"{w} ({cnt})" for w, cnt in words[:10]]),
                    "Text Count": len(text_clusters.get(cid, [])),
                }
            )
        return pd.DataFrame(rows)

    # --------------------------- Topic name helpers ---------------------------

    def set_topic(self, cluster_id: int, topic_name: str) -> "SEMTGPU":
        if not (isinstance(cluster_id, int) and 0 <= cluster_id < self.n_clusters):
            raise ValueError(f"cluster_id harus di [0, {self.n_clusters-1}]")
        if not isinstance(topic_name, str):
            raise ValueError("topic_name harus string")
        self.topic_mapping[cluster_id] = topic_name
        print(f"✓ Assigned '{topic_name}' to cluster {cluster_id}")
        return self

    def reset_topics(self) -> "SEMTGPU":
        self.topic_mapping = {}
        print("✓ All topic assignments reset")
        return self

    def get_topic_assignments(self) -> Dict[int, str]:
        return self.topic_mapping.copy()

    # ------------------------------ Visualization ------------------------------

    def plot_sentiment_by_topic(
        self,
        data: pd.DataFrame,
        x: Union[torch.Tensor, np.ndarray],
        figsize: Tuple[int, int] = (15, 6),
        palette: str = "Set1",
        negative_color: Optional[str] = None,
        positive_color: Optional[str] = None,
    ):
        """
        Plot distribusi sentiment per topic (stacked horizontal bars).
        Asumsi kolom 'sentiment' di data: nilai 0/1 atau label yang sudah mapped.
        """
        sns.set_style("whitegrid")

        cluster = self.get_cluster_assignments(x)
        df = data.copy()
        df["cluster"] = cluster

        sentiment_count = df.groupby(["cluster", "sentiment"]).size().unstack(fill_value=0)
        total = sentiment_count.values.sum()
        percent = (sentiment_count / total) * 100

        percent["topic"] = percent.index.map(lambda cid: self.topic_mapping.get(cid, f"Cluster {cid}"))
        melted = percent.reset_index().melt(
            id_vars=["cluster", "topic"], value_vars=percent.columns[:2], var_name="sentiment", value_name="percentage"
        ).sort_values("cluster")

        if negative_color is None or positive_color is None:
            colors = sns.color_palette(palette, 2)
            neg_col = negative_color or colors[0]
            pos_col = positive_color or colors[1]
        else:
            neg_col, pos_col = negative_color, positive_color

        fig, ax = plt.subplots(figsize=figsize)

        sentiment_labels = {0: self.class_labels.get(0, "Negative"), 1: self.class_labels.get(1, "Positive")}
        topics = melted["topic"].drop_duplicates().tolist()
        left_map = {t: 0.0 for t in topics}

        for sval, color, label in zip([0, 1], [neg_col, pos_col], [sentiment_labels[0], sentiment_labels[1]]):
            sub = melted[melted["sentiment"] == sval]
            ys = sub["topic"].tolist()
            ws = sub["percentage"].tolist()
            lefts = [left_map[y] for y in ys]
            ax.barh(ys, ws, left=lefts, label=label, color=color, edgecolor="white", linewidth=0.5)
            for y, w, l in zip(ys, ws, lefts):
                if w > 3:
                    ax.text(l + w / 2, y, f"{w:.1f}%", va="center", ha="center", color="white", fontsize=9, fontweight="bold")
            for y, w in zip(ys, ws):
                left_map[y] += w

        ax.set_xlabel("Percentage of All Reviews", fontsize=11)
        ax.set_ylabel("Cluster Topic", fontsize=11)
        ax.set_title("Sentiment Distribution by Cluster Topic", fontsize=14, fontweight="bold")
        ax.legend(title="Sentiment", title_fontsize=11, frameon=True, fancybox=True, framealpha=0.9, shadow=True, fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        sns.despine()
        plt.tight_layout()
        return fig

    def plot_cluster_evolution(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
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
        """
        Visualisasi 2D cluster (t-SNE / UMAP) di epoch tertentu.
        """
        if isinstance(embeddings, torch.Tensor):
            emb_np = embeddings.detach().cpu().numpy()
        else:
            emb_np = embeddings

        if method.lower() == "tsne":
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
            emb_2d = reducer.fit_transform(emb_np)
        elif method.lower() == "umap":
            try:
                import umap  # type: ignore

                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                emb_2d = reducer.fit_transform(emb_np)
            except Exception:
                from sklearn.manifold import TSNE

                reducer = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
                emb_2d = reducer.fit_transform(emb_np)
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
        ax.set_xticks([])
        ax.set_yticks([])
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
        self,
        save_dir: str = "./results/fnnjst",
        epochs_to_show: Optional[List[int]] = None,
        grid_cols: int = 3,
        figsize: Tuple[int, int] = (15, 10),
    ):
        """
        Buat grid dari plot evolusi cluster yang disimpan.
        """
        plot_files = glob.glob(os.path.join(save_dir, "cluster_evolution_epoch_*.png"))
        if not plot_files:
            print("No cluster evolution plots found")
            return None

        epoch_files = []
        for fp in plot_files:
            m = re.search(r"epoch_(\d+)\.png", fp)
            if m:
                epoch_files.append((int(m.group(1)), fp))
        epoch_files.sort(key=lambda t: t[0])

        if epochs_to_show is not None:
            epoch_files = [t for t in epoch_files if t[0] in epochs_to_show]
        if not epoch_files:
            print("No matching epoch plots found")
            return None

        n = len(epoch_files)
        rows = (n + grid_cols - 1) // grid_cols
        fig, axes = plt.subplots(rows, grid_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1) if rows > 1 else (axes if isinstance(axes, np.ndarray) else np.array([axes]))

        import matplotlib.image as mpimg

        for i, (ep, path) in enumerate(epoch_files):
            if i >= len(axes):
                break
            img = mpimg.imread(path)
            axes[i].imshow(img)
            axes[i].set_title(f"Epoch {ep}", fontsize=12, fontweight="bold")
            axes[i].axis("off")

        for i in range(len(epoch_files), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        out = os.path.join(save_dir, "cluster_evolution_grid.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"✓ Saved evolution grid: {out}")
        return fig
