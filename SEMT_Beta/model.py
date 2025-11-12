"""
SEMT: Feedforward Neural Network with Joint Sentiment Analysis and Clustering
============================================================================
A deep learning model combining autoencoder-based clustering with sentiment classification.
"""

from time import time
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional, Set
import csv
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import Counter

from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.utils import to_categorical
import keras.backend as K

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, AutoModel

from .DEC import cluster_acc, ClusteringLayer, autoencoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class SEMT:
    """
    Feedforward Neural Network for joint sentiment analysis and clustering.
    
    Attributes:
        dims (list): Network architecture dimensions
        n_clusters (int): Number of clustering groups
        batch_size (int): Training batch size
        class_labels (dict): Mapping of class indices to labels
    """
    
    def __init__(
        self,
        dims: List[int],
        n_clusters: int = 10,
        batch_size: int = 256
    ):
        """
        Initialize FNN model.
        
        Args:
            dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., latent_dim]
            n_clusters: Number of clusters for unsupervised learning
            batch_size: Batch size for training
        """
        super(SEMT, self).__init__()
        
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        
        self.autoencoder = autoencoder(self.dims)
        self.encoder = None
        self.model = None
        
        self.class_labels = {0: 'negative', 1: 'positive'}
        self.stop_words: Set[str] = set()
        self.gamma: Optional[float] = None
        self.eta: Optional[float] = None
        
        logger.info(f"SEMT initialized with architecture: {dims}")
        logger.info(f"Clusters: {n_clusters}, Batch size: {batch_size}")

    def initialize_model(
        self,
        ae_weights: Optional[str] = None,
        gamma: float = 0.1,
        eta: float = 1.0,
        optimizer: Optional[SGD] = None
    ) -> None:
        """
        Initialize the complete model with pretrained autoencoder weights.
        
        Args:
            ae_weights: Path to pretrained autoencoder weights
            gamma: Weight for clustering loss
            eta: Weight for sentiment loss
            optimizer: Keras optimizer instance
        """
        if ae_weights is None:
            logger.error("Autoencoder weights must be provided")
            raise ValueError("ae_weights parameter is required. Example: 'weights.h5'")
        
        if not Path(ae_weights).exists():
            logger.error(f"Weights file not found: {ae_weights}")
            raise FileNotFoundError(f"Weights file does not exist: {ae_weights}")
        
        self.autoencoder.load_weights(ae_weights)
        logger.info(f"Loaded pretrained weights from: {ae_weights}")
        
        self.gamma = gamma
        self.eta = eta
        
        # Build encoder
        hidden = self.autoencoder.get_layer(name=f'encoder_{self.n_stacks - 1}').output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)
        
        # Build sentiment classification head
        x = Dense(128)(hidden)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)
        
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)
        
        sentiment_output = Dense(2, activation='softmax', name='sentiment')(x)
        
        # Build clustering layer
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        
        # Complete model
        self.model = Model(
            inputs=self.autoencoder.input,
            outputs=[clustering_layer, sentiment_output]
        )
        
        if optimizer is None:
            optimizer = SGD(learning_rate=0.001, momentum=0.9)
        
        self.model.compile(
            loss={'clustering': 'kld', 'sentiment': 'categorical_crossentropy'},
            optimizer=optimizer
        )
        
        logger.info(f"Model compiled successfully (gamma={gamma}, eta={eta})")

    def load_weights(self, weights_path: str) -> None:
        """Load model weights from file."""
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        self.model.load_weights(weights_path)
        logger.info(f"Loaded model weights from: {weights_path}")

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        """Extract features from encoder."""
        encoder = Model(
            self.model.input,
            self.model.get_layer(f'encoder_{self.n_stacks - 1}').output
        )
        return encoder.predict(x, verbose=0)

    def predict_clusters(self, x: np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    def predict_sentiment(self, x: np.ndarray) -> np.ndarray:
        """Predict sentiment labels."""
        _, s = self.model.predict(x, verbose=0)
        return s.argmax(1)

    def predict(
        self,
        inputs: Union[str, List[str], torch.Tensor],
        bert_model: Optional[Union[str, AutoModel]] = None
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Predict sentiment and cluster for input text(s) or embeddings.
        
        Args:
            inputs: Text string(s) or embeddings tensor
            bert_model: BERT model name or instance for encoding text
            
        Returns:
            List of dictionaries containing sentiment and cluster predictions
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        if isinstance(inputs, list) and isinstance(inputs[0], str):
            model_name = bert_model if isinstance(bert_model, str) else "indolem/indobert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            tokens = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(device)

            with torch.no_grad():
                if not callable(bert_model):
                    bert_model = AutoModel.from_pretrained(model_name).to(device)
                else:
                    bert_model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
                
                outputs = bert_model(**tokens)
            
            embeddings_numpy = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()

        elif isinstance(inputs, torch.Tensor):
            embeddings_numpy = inputs.cpu().detach().numpy()
            if embeddings_numpy.ndim == 1:
                embeddings_numpy = np.expand_dims(embeddings_numpy, axis=0)
        else:
            raise ValueError("Input must be text string(s) or embeddings tensor")
        
        cluster_output, sentiment_output = self.model.predict(embeddings_numpy, verbose=0)
        
        cluster_preds = cluster_output.argmax(1)
        sentiment_preds = sentiment_output.argmax(1)

        results = [
            {
                'sentiment': self.class_labels[sentiment_preds[i]],
                'cluster': int(cluster_preds[i])
            }
            for i in range(len(sentiment_preds))
        ]
        
        return results

    def get_cluster_assignments(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Get cluster assignments for input data."""
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
            
        cluster_output, _ = self.model.predict(x, verbose=0)
        return cluster_output.argmax(1)

    def set_stop_words(self, stop_words: Union[List[str], Set[str]]) -> 'SEMT':
        """
        Set stop words for cluster analysis.
        
        Args:
            stop_words: Collection of words to filter out
            
        Returns:
            Self for method chaining
        """
        try:
            self.stop_words = set(stop_words) if not isinstance(stop_words, set) else stop_words
            logger.info(f"Stop words set: {len(self.stop_words)} words")
        except Exception as e:
            raise ValueError(f"Invalid stop_words format: {e}")
        
        return self

    def map_texts_to_clusters(
        self,
        texts: List[str],
        cluster_assignments: np.ndarray
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[Tuple[str, int]]]]:
        """
        Map texts to their assigned clusters and extract common words.
        
        Args:
            texts: List of text strings
            cluster_assignments: Cluster assignment for each text
            
        Returns:
            Tuple of (cluster_texts_dict, cluster_words_dict)
        """
        clusters = {}
        n = min(len(texts), len(cluster_assignments))
        
        for i in range(n):
            cluster = int(cluster_assignments[i])
            clusters.setdefault(cluster, []).append(texts[i])
        
        cluster_common_words = {}
        for cluster, cluster_texts in clusters.items():
            all_text = " ".join(cluster_texts).lower()
            words = [
                word for word in all_text.split()
                if word not in self.stop_words and len(word) > 2
            ]
            word_counts = Counter(words)
            cluster_common_words[cluster] = word_counts.most_common(20)
        
        return clusters, cluster_common_words

    def analyze_clusters(
        self,
        x: Union[torch.Tensor, np.ndarray],
        texts: List[str]
    ) -> pd.DataFrame:
        """
        Analyze cluster assignments and extract common words.
        
        Args:
            x: Input embeddings
            texts: Corresponding text strings
            
        Returns:
            DataFrame with cluster analysis results
        """
        cluster_assignments = self.get_cluster_assignments(x)
        text_clusters, cluster_words = self.map_texts_to_clusters(texts, cluster_assignments)

        df_clusters = pd.DataFrame([
            {
                "Cluster": cluster,
                "Common Words": ", ".join([f"{word} ({count})" for word, count in words[:10]]),
                "Text Count": len(text_clusters[cluster])
            }
            for cluster, words in cluster_words.items()
        ]).sort_values(by=['Cluster']).reset_index(drop=True)

        logger.info("\n" + "=" * 80)
        logger.info("CLUSTER ANALYSIS RESULTS")
        logger.info("=" * 80)
        logger.info("\n" + df_clusters.to_string(index=False))
        logger.info("=" * 80)

        return df_clusters

    def pretrain_autoencoder(
        self,
        dataset,
        batch_size: int = 256,
        epochs: int = 200,
        optimizer: str = 'adam'
    ) -> np.ndarray:
        """
        Pretrain the autoencoder on unlabeled data.
        
        Args:
            dataset: PyTorch dataset or list of embeddings
            batch_size: Training batch size
            epochs: Number of training epochs
            optimizer: Optimizer name
            
        Returns:
            Trained autoencoder weights
        """
        logger.info("=" * 80)
        logger.info("AUTOENCODER PRETRAINING")
        logger.info("=" * 80)
        
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        embeddings = []
        for item in tqdm(dataset, desc="Loading dataset", ncols=100):
            if isinstance(item, tuple):
                embedding, _ = item
                embeddings.append(embedding.cpu().numpy())
            else:
                embeddings.append(item.cpu().numpy())

        x = np.array(embeddings)
        logger.info(f"Dataset shape: {x.shape}")

        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, verbose=1)
        
        weights_path = 'pretrained_ae.weights.h5'
        self.autoencoder.save_weights(weights_path)
        logger.info(f"Autoencoder weights saved to: {weights_path}")

        hidden = self.autoencoder.get_layer(name=f'encoder_{self.n_stacks - 1}').output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        return self.autoencoder.get_weights()

    @staticmethod
    def target_distribution(q: np.ndarray) -> np.ndarray:
        """
        Compute target distribution for clustering.
        
        Args:
            q: Soft cluster assignments
            
        Returns:
            Target distribution
        """
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute balanced class weights for imbalanced datasets.
        
        Args:
            y: Labels array (one-hot or indices)
            
        Returns:
            Dictionary mapping class indices to weights
        """
        y_indices = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        unique_classes, class_counts = np.unique(y_indices, return_counts=True)
        
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_indices
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        logger.info(f"Balanced weights: {class_weight_dict}")
        
        return class_weight_dict

    def train(
        self,
        dataset,
        tol: float = 1e-3,
        update_interval: int = 140,
        maxiter: int = 20000,
        save_dir: str = './results'
    ) -> None:
        """
        Train the joint clustering and sentiment analysis model.
        
        Args:
            dataset: Training dataset with embeddings and labels
            tol: Tolerance for early stopping based on label changes
            update_interval: Iterations between target distribution updates
            maxiter: Maximum training iterations
            save_dir: Directory for saving model checkpoints and logs
        """
        logger.info("=" * 80)
        logger.info("JOINT TRAINING: CLUSTERING + SENTIMENT ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Update interval: {update_interval}")
        logger.info(f"Tolerance: {tol}")
        logger.info(f"Max iterations: {maxiter}")
        
        # Load data
        embeddings, sentiment_labels = [], []
        for item in tqdm(dataset, desc="Loading training data", ncols=100):
            if isinstance(item, tuple):
                embedding, label = item
                embeddings.append(embedding.cpu().numpy())
                sentiment_labels.append(label.cpu().numpy())
            else:
                embeddings.append(item.cpu().numpy())

        x = np.array(embeddings)

        # Process labels
        if sentiment_labels:
            y_sentiment = np.array(sentiment_labels)
            if len(y_sentiment.shape) == 1:
                y_sentiment = to_categorical(y_sentiment, num_classes=2)
            sentiment_class_weights = self.compute_class_weights(y_sentiment)
        else:
            logger.warning("No labels found. Running in clustering-only mode.")
            y_sentiment = None
            sentiment_class_weights = None

        # Setup saving
        save_interval = int(x.shape[0] / self.batch_size * 5)
        logger.info(f"Save interval: {save_interval} iterations")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Initialize clusters
        logger.info("Initializing cluster centers with K-Means...")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        y_pred = kmeans.fit_predict(self.encoder.predict(x, verbose=0))
        y_pred_last = y_pred.copy()
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        logger.info("Cluster initialization complete")

        # Setup logging
        logfile = open(save_path / 'training_log.csv', 'w', newline='')
        fieldnames = ['iter', 'acc_cluster', 'nmi', 'ari', 'acc_sentiment', 'L', 'Lc', 'Ls']
        logwriter = csv.DictWriter(logfile, fieldnames=fieldnames)
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0

        # Training loop
        logger.info("Starting training loop...")
        pbar = tqdm(range(int(maxiter)), desc="Training", ncols=120)

        for ite in pbar:
            # Update target distribution
            if ite % update_interval == 0:
                q, s_pred = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)

                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred.copy()

                # Compute metrics
                if y_sentiment is not None:
                    s_pred_label = s_pred.argmax(1)
                    sentiment_true_label = y_sentiment.argmax(1) if len(y_sentiment.shape) > 1 else y_sentiment
                    acc_sentiment = np.mean(s_pred_label == sentiment_true_label)

                    # Per-class accuracy - only log to file, not to console
                    if len(np.unique(sentiment_true_label)) > 1:
                        class_accs = {}
                        for cls in np.unique(sentiment_true_label):
                            cls_mask = sentiment_true_label == cls
                            cls_acc = np.mean((s_pred_label == sentiment_true_label) & cls_mask) / np.mean(cls_mask)
                            class_accs[self.class_labels[cls]] = cls_acc
                else:
                    acc_sentiment = 0
                    class_accs = {}

                # Log metrics
                loss_rounded = np.round(loss, 5)
                logdict = {
                    'iter': ite,
                    'acc_cluster': 0,
                    'nmi': 0,
                    'ari': 0,
                    'acc_sentiment': round(acc_sentiment, 5),
                    'L': loss_rounded[0],
                    'Lc': loss_rounded[1],
                    'Ls': loss_rounded[2]
                }
                logwriter.writerow(logdict)

                pbar.set_postfix({
                    'Lc': f'{loss[1]:.4f}',
                    'Ls': f'{loss[2]:.4f}',
                    'Acc': f'{acc_sentiment:.4f}',
                }, refresh=True)

                # Log detailed info only every few updates to avoid clutter
                if ite % (update_interval * 5) == 0 and ite > 0:
                    if class_accs:
                        class_acc_str = ', '.join([f"{k}: {v:.4f}" for k, v in class_accs.items()])
                        logger.info(f"Iter {ite} - Class accuracies: {class_acc_str}")

                # Early stopping
                if ite > 0 and delta_label < tol:
                    logger.info(f"Delta label ({delta_label:.6f}) < tolerance ({tol}). Early stopping.")
                    break

            # Training step
            if y_sentiment is not None:
                end_idx = (index + 1) * self.batch_size
                if end_idx > x.shape[0]:
                    batch_x = x[index * self.batch_size:]
                    batch_p = p[index * self.batch_size:]
                    batch_y = y_sentiment[index * self.batch_size:]
                    index = 0
                else:
                    batch_x = x[index * self.batch_size:end_idx]
                    batch_p = p[index * self.batch_size:end_idx]
                    batch_y = y_sentiment[index * self.batch_size:end_idx]
                    index += 1

                # Compute sample weights
                cluster_weights = np.full(batch_p.shape[0], self.gamma)
                if sentiment_class_weights:
                    sentiment_weights = np.array([
                        self.eta * sentiment_class_weights[label]
                        for label in np.argmax(batch_y, axis=1)
                    ])
                else:
                    sentiment_weights = np.full(batch_y.shape[0], self.eta)

                loss = self.model.train_on_batch(
                    x=batch_x,
                    y=[batch_p, batch_y],
                    sample_weight=[cluster_weights, sentiment_weights]
                )

            # Save checkpoint
            if ite % save_interval == 0 and ite > 0:
                checkpoint_path = save_path / f'FNN_model_{ite}.weights.h5'
                self.model.save_weights(str(checkpoint_path))

        pbar.close()
        logfile.close()

        # Save final model
        final_path = save_path / 'FNN_model_final.weights.h5'
        self.model.save_weights(str(final_path))
        logger.info(f"Training complete. Final model saved to: {final_path}")
