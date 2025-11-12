"""
SEMT-GPU: Feedforward Neural Network with GPU support for 
Joint Sentiment Analysis and Topic Clustering

This module implements a deep learning model that combines:
- Autoencoder for feature extraction
- Clustering layer for topic modeling
- Sentiment classifier for sentiment analysis
"""

import os
import csv
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment as linear_assignment

from transformers import AutoTokenizer, AutoModel

import seaborn as sns
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


 
# Utility Functions
 

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using Hungarian algorithm.
    
    Args:
        y_true: True labels, numpy.array with shape (n_samples,)
        y_pred: Predicted labels, numpy.array with shape (n_samples,)
    
    Returns:
        float: Accuracy in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


 
# Neural Network Components
 

class ClusteringLayer(nn.Module):
    """
    Clustering layer that converts input features to soft cluster assignments
    using Student's t-distribution (similar to t-SNE).
    """
    
    def __init__(self, n_clusters, input_dim, alpha=1.0):
        """
        Args:
            n_clusters: Number of clusters
            input_dim: Dimension of input features
            alpha: Degrees of freedom for Student's t-distribution
        """
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, input_dim))
        self._init_weights()
        
    def _init_weights(self):
        """Initialize cluster centers with Xavier uniform."""
        nn.init.xavier_uniform_(self.clusters)

    def forward(self, x):
        """
        Compute soft cluster assignments using Student's t-distribution.
        
        Args:
            x: Input features, shape (n_samples, n_features)
            
        Returns:
            q: Soft cluster assignments, shape (n_samples, n_clusters)
        """
        # Compute squared Euclidean distances
        q = 1.0 / (1.0 + (torch.sum(
            torch.square(x.unsqueeze(1) - self.clusters.unsqueeze(0)), 
            dim=2
        ) / self.alpha))
        
        # Apply Student's t-distribution
        q = q ** ((self.alpha + 1.0) / 2.0)
        
        # Normalize to probability distribution
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q


class Autoencoder(nn.Module):
    """
    Fully connected symmetric autoencoder for dimensionality reduction.
    """
    
    def __init__(self, dims, act='relu'):
        """
        Args:
            dims: List of layer dimensions [input_dim, ..., encoding_dim]
            act: Activation function ('relu', 'sigmoid', 'tanh')
        """
        super(Autoencoder, self).__init__()
        
        self.dims = dims
        self.n_stacks = len(dims) - 1
        
        # Select activation function
        activation_map = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        self.activation = activation_map.get(act, nn.ReLU())
        
        # Build encoder
        encoder_layers = []
        for i in range(self.n_stacks - 1):
            encoder_layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                self.activation
            ))
        encoder_layers.append(nn.Linear(dims[-2], dims[-1]))
        
        # Build decoder (symmetric)
        decoder_layers = []
        for i in range(self.n_stacks - 1, 0, -1):
            decoder_layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i-1]),
                self.activation
            ))
        decoder_layers.append(nn.Linear(dims[1], dims[0]))
        
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize all linear layers with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """Encode input through encoder layers."""
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        return h
    
    def decode(self, h):
        """Decode hidden representation through decoder layers."""
        for layer in self.decoder_layers:
            h = layer(h)
        return h
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        h = self.encode(x)
        return h, self.decode(h)


 
# Main Model
 

class SEMTGPU(nn.Module):
    """
    SEMT-GPU: Joint Sentiment Analysis and Topic Clustering Model
    
    Combines:
    - Autoencoder for feature extraction
    - Clustering layer for topic discovery
    - Sentiment classifier for sentiment prediction
    """
    
    def __init__(self, dims, n_clusters=10, alpha=1.0):
        """
        Args:
            dims: List of layer dimensions for autoencoder
            n_clusters: Number of topic clusters
            alpha: Degrees of freedom for clustering layer
        """
        super(SEMTGPU, self).__init__()
        
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(dims) - 1
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Core components
        self.autoencoder = Autoencoder(dims)
        self.clustering = ClusteringLayer(n_clusters, dims[-1], alpha)
        
        # Sentiment classifier with batch normalization
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(dims[-1], 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(32, 2)  # Binary: positive/negative
        )
        
        # Metadata
        self.class_labels = {0: 'negative', 1: 'positive'}
        self.topic_mapping = {}
        self.stop_words = set()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize sentiment classifier weights."""
        for m in self.sentiment_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input features
            
        Returns:
            cluster_output: Soft cluster assignments
            sentiment_output: Sentiment probabilities
        """
        encoded = self.autoencoder.encode(x)
        cluster_output = self.clustering(encoded)
        sentiment_output = torch.softmax(self.sentiment_classifier(encoded), dim=1)
        return cluster_output, sentiment_output
    
   
    # Feature Extraction
   
    
    def extract_feature(self, x):
        """Extract encoded features from autoencoder."""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)
            encoded = self.autoencoder.encode(x)
        return encoded
    
   
    # Weight Management
   
    
    def load_weights(self, weights_path):
        """Load model weights from file."""
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded weights from {weights_path}")
        
    def save_weights(self, weights_path):
        """Save model weights to file."""
        torch.save({'model_state_dict': self.state_dict()}, weights_path)
        print(f"✓ Saved weights to {weights_path}")
    
   
    # Autoencoder Pretraining
   
    
    def pretrain_autoencoder(self, dataset, batch_size=256, epochs=200, 
                            learning_rate=0.001):
        """
        Pretrain the autoencoder using reconstruction loss.
        
        Args:
            dataset: PyTorch dataset containing embeddings
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        print('=' * 60)
        print('Pretraining Autoencoder')
        print('=' * 60)
        
        # Extract embeddings from dataset
        embeddings = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, tuple):
                embedding, _ = item
            else:
                embedding = item
            embeddings.append(embedding.cpu())
        
        # Create tensor dataset
        embeddings_tensor = torch.stack(embeddings).to(device)
        embeddings_dataset = TensorDataset(embeddings_tensor)
        
        print(f"Dataset shape: {embeddings_tensor.shape}")
        print(f"Batch size: {batch_size}, Epochs: {epochs}, LR: {learning_rate}")
        
        # Setup training
        data_loader = DataLoader(embeddings_dataset, batch_size=batch_size, 
                               shuffle=True)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        self.autoencoder.to(device)
        self.autoencoder.train()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            with tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for data in pbar:
                    inputs = data[0].to(device)
                    
                    optimizer.zero_grad()
                    _, reconstructed = self.autoencoder(inputs)
                    loss = criterion(reconstructed, inputs)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
        
        # Save pretrained weights
        self.save_weights('pretrained_ae.weights.pth')
        print('✓ Autoencoder pretraining completed')
            
   
    # Clustering Utilities
   
    
    @staticmethod
    def target_distribution(q):
        """
        Compute auxiliary target distribution for clustering.
        
        Args:
            q: Soft cluster assignments
            
        Returns:
            p: Target distribution
        """
        weight = q ** 2 / torch.sum(q, dim=0)
        return (weight.t() / torch.sum(weight, dim=1)).t()
    
    def compute_class_weights(self, y):
        """
        Compute balanced class weights for imbalanced datasets.
        
        Args:
            y: Labels (one-hot or class indices)
            
        Returns:
            dict: Mapping from class to weight
        """
        # Convert to class indices if one-hot encoded
        if len(y.shape) > 1:
            y_indices = (torch.argmax(y, dim=1) if isinstance(y, torch.Tensor) 
                        else np.argmax(y, axis=1))
        else:
            y_indices = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        
        # Calculate class distribution
        unique_classes, class_counts = np.unique(y_indices, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        # Compute balanced weights
        total_samples = len(y_indices)
        n_classes = len(unique_classes)
        class_weights = {
            c: total_samples / (n_classes * class_counts[i])
            for i, c in enumerate(unique_classes)
        }
        
        print(f"Class weights: {class_weights}")
        return class_weights
    
   
    # Prediction Methods
   
    
    def predict_clusters(self, x):
        """Predict cluster assignments."""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)
            cluster_output, _ = self(x)
            return torch.argmax(cluster_output, dim=1).cpu().numpy()
    
    def predict_sentiment(self, x):
        """Predict sentiment labels."""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)
            _, sentiment_output = self(x)
            return torch.argmax(sentiment_output, dim=1).cpu().numpy()
    
    def predict(self, inputs, bert_model=None):
        """
        Predict both clusters and sentiment for text or embeddings.
        
        Args:
            inputs: Text strings, list of strings, or embeddings
            bert_model: Pre-trained BERT model or model name
            
        Returns:
            list: Predictions with 'sentiment' and 'topic' keys
        """
        self.eval()
        
        # Handle text inputs
        if isinstance(inputs, str):
            inputs = [inputs]
        
        if isinstance(inputs, list) and isinstance(inputs[0], str):
            # Load tokenizer and model
            model_name = (bert_model if isinstance(bert_model, str) 
                         else "indolem/indobert-base-uncased")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            tokens = tokenizer(
                inputs, padding=True, truncation=True,
                return_tensors="pt", max_length=512
            ).to(device)
            
            with torch.no_grad():
                if not callable(bert_model):
                    bert_model = AutoModel.from_pretrained(model_name)
                    bert_model.to(device)
                outputs = bert_model(**tokens)
            
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            # Handle embedding inputs
            if not isinstance(inputs, torch.Tensor):
                embeddings = torch.tensor(inputs, dtype=torch.float32)
            else:
                embeddings = inputs
            embeddings = embeddings.to(device)
        
        # Get predictions
        with torch.no_grad():
            cluster_output, sentiment_output = self(embeddings)
        
        cluster_preds = torch.argmax(cluster_output, dim=1).cpu().numpy()
        sentiment_preds = torch.argmax(sentiment_output, dim=1).cpu().numpy()
        
        # Format results
        results = []
        for i in range(len(sentiment_preds)):
            sentiment_label = self.class_labels[sentiment_preds[i]]
            cluster_id = int(cluster_preds[i])
            cluster_or_topic = self.topic_mapping.get(cluster_id, cluster_id)
            
            results.append({
                'sentiment': sentiment_label,
                'topic': cluster_or_topic
            })
        
        return results
    
   
    # Joint Training
   
    
    def clustering_with_sentiment(
        self, dataset, gamma=0.7, eta=1, optimizer_type='sgd',
        learning_rate=0.001, momentum=0.9, tol=1e-3, update_interval=140,
        batch_size=128, maxiter=2e4, save_dir='./results/fnnjst',
        plot_evolution=True, plot_interval=None
    ):
        """
        Train the model with joint clustering and sentiment objectives.
        
        Args:
            dataset: PyTorch dataset
            gamma: Weight for clustering loss
            eta: Weight for sentiment loss
            optimizer_type: Optimizer name
            learning_rate: Learning rate
            momentum: Momentum for SGD
            tol: Convergence tolerance
            update_interval: Steps between target distribution updates
            batch_size: Batch size
            maxiter: Maximum iterations
            save_dir: Directory for saving results
            plot_evolution: Whether to save evolution plots
            plot_interval: Steps between plots (default: update_interval)
        
        Returns:
            Cluster and sentiment predictions
        """
        print('=' * 60)
        print('Joint Clustering and Sentiment Training')
        print('=' * 60)
        print(f"Update interval: {update_interval}")
        print(f"Gamma (clustering): {gamma}, Eta (sentiment): {eta}")
        
        # Setup directories
        os.makedirs(save_dir, exist_ok=True)
        if plot_evolution:
            plot_dir = os.path.join(save_dir, 'evolution_plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_interval = plot_interval or update_interval
        
        # Prepare dataset
        embeddings, labels = self._prepare_dataset(dataset)
        all_embeddings = torch.cat(embeddings, dim=0).to(device)
        
        has_labels = len(labels) > 0
        if has_labels:
            all_labels = torch.cat(labels, dim=0).to(device)
            y_sentiment = all_labels.cpu().numpy()
            class_weights = self.compute_class_weights(y_sentiment)
            class_weight_tensor = torch.tensor(
                [class_weights[i] for i in range(len(self.class_labels))],
                dtype=torch.float32
            ).to(device)
        else:
            all_labels = None
            y_sentiment = None
            class_weight_tensor = None
        
        # Setup optimizer
        optimizer = self._get_optimizer(optimizer_type, learning_rate, momentum)
        
        # Setup loss functions
        kld_loss = nn.KLDivLoss(reduction='batchmean')
        sentiment_loss = (nn.CrossEntropyLoss(weight=class_weight_tensor) 
                         if class_weight_tensor is not None 
                         else nn.CrossEntropyLoss())
        
        # Initialize clusters with k-means
        y_pred_last = self._initialize_clusters(all_embeddings, plot_evolution, 
                                                plot_dir if plot_evolution else None)
        
        # Setup logging
        logfile, logwriter = self._setup_logging(save_dir)
        save_interval = len(DataLoader(
            TensorDataset(all_embeddings), batch_size=batch_size
        )) * 5
        
        # Training loop
        self._train_loop(
            all_embeddings, all_labels, y_sentiment, has_labels,
            gamma, eta, optimizer, kld_loss, sentiment_loss,
            batch_size, maxiter, update_interval, tol, save_interval,
            logwriter, save_dir, plot_evolution, plot_interval,
            plot_dir if plot_evolution else None, y_pred_last
        )
        
        # Cleanup and save
        logfile.close()
        self.save_weights(os.path.join(save_dir, 'FNN_model_final.weights.pth'))
        
        # Return predictions
        self.eval()
        with torch.no_grad():
            q, s_pred = self(all_embeddings)
            y_pred = torch.argmax(q, dim=1).cpu().numpy()
            if has_labels:
                return y_pred, s_pred.cpu().numpy()
            return y_pred
    
    def _prepare_dataset(self, dataset):
        """Extract embeddings and labels from dataset."""
        embeddings, labels = [], []
        
        for i in tqdm(range(len(dataset)), desc="Loading dataset"):
            item = dataset[i]
            if isinstance(item, tuple) and len(item) == 2:
                embedding, label = item
                embeddings.append(embedding.cpu())
                labels.append(label.cpu())
            else:
                emb = (item.cpu() if isinstance(item, torch.Tensor) 
                      else torch.tensor(item, dtype=torch.float32))
                embeddings.append(emb)
        
        return embeddings, labels
    
    def _get_optimizer(self, optimizer_type, learning_rate, momentum):
        """Get optimizer by name."""
        optimizers = {
            'sgd': lambda: optim.SGD(self.parameters(), lr=learning_rate, 
                                    momentum=momentum),
            'adam': lambda: optim.Adam(self.parameters(), lr=learning_rate),
            'adamw': lambda: optim.AdamW(self.parameters(), lr=learning_rate),
            'rmsprop': lambda: optim.RMSprop(self.parameters(), lr=learning_rate, 
                                            alpha=0.99),
            'adagrad': lambda: optim.Adagrad(self.parameters(), lr=learning_rate),
            'adamax': lambda: optim.Adamax(self.parameters(), lr=learning_rate),
            'asgd': lambda: optim.ASGD(self.parameters(), lr=learning_rate),
            'adadelta': lambda: optim.Adadelta(self.parameters(), lr=learning_rate),
            'nadam': lambda: optim.NAdam(self.parameters(), lr=learning_rate),
        }
        
        if optimizer_type.lower() not in optimizers:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_type}. "
                f"Supported: {list(optimizers.keys())}"
            )
        
        return optimizers[optimizer_type.lower()]()
    
    def _initialize_clusters(self, embeddings, plot_evolution, plot_dir):
        """Initialize cluster centers using k-means."""
        print('Initializing cluster centers with k-means...')
        self.eval()
        features = self.extract_feature(embeddings).cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        y_pred = kmeans.fit_predict(features)
        
        cluster_centers = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        ).to(device)
        self.clustering.clusters.data = cluster_centers
        
        # Save initial plot
        if plot_evolution and plot_dir:
            try:
                self.plot_cluster_evolution(
                    features, y_pred, 0, plot_dir, show_plot=False
                )
            except Exception as e:
                print(f"Warning: Could not save initial plot: {e}")
        
        return y_pred
    
    def _setup_logging(self, save_dir):
        """Setup CSV logging."""
        logfile = open(os.path.join(save_dir, 'training_log.csv'), 
                      'w', newline='')
        fieldnames = ['iter', 'acc_cluster', 'nmi', 'ari', 'acc_sentiment', 
                     'L', 'Lc', 'Ls']
        logwriter = csv.DictWriter(logfile, fieldnames=fieldnames)
        logwriter.writeheader()
        return logfile, logwriter
    
    def _train_loop(self, all_embeddings, all_labels, y_sentiment, has_labels,
                   gamma, eta, optimizer, kld_loss, sentiment_loss,
                   batch_size, maxiter, update_interval, tol, save_interval,
                   logwriter, save_dir, plot_evolution, plot_interval, plot_dir,
                   y_pred_last):
        """Main training loop."""
        iter_count = 0
        total_loss = cluster_loss = sent_loss = 0
        
        for ite in range(int(maxiter)):
            # Update target distribution periodically
            if ite % update_interval == 0:
                p, y_pred_last, acc_sentiment, avg_losses = self._update_distribution(
                    all_embeddings, all_labels, y_sentiment, has_labels,
                    y_pred_last, batch_size, iter_count, update_interval,
                    total_loss, cluster_loss, sent_loss,
                    plot_evolution, plot_interval, plot_dir, ite
                )
                
                # Log metrics
                self._log_metrics(logwriter, ite, acc_sentiment, *avg_losses)
                
                # Check convergence
                if ite > 0:
                    current_pred = self.get_cluster_assignments(all_embeddings)
                    delta_label = np.sum(
                        y_pred_last != current_pred
                    ).astype(np.float32) / len(y_pred_last)
                    
                    if delta_label < tol:
                        print(f'Convergence reached (delta={delta_label:.5f} < {tol})')
                        break
                
                # Reset loss counters
                total_loss = cluster_loss = sent_loss = 0
                
                # Update train loader with new target distribution
                if has_labels and all_labels is not None:
                    train_loader = DataLoader(
                        TensorDataset(all_embeddings, p, all_labels),
                        batch_size=batch_size, shuffle=True
                    )
                else:
                    train_loader = DataLoader(
                        TensorDataset(all_embeddings, p),
                        batch_size=batch_size, shuffle=True
                    )
            
            # Training step
            self.train()
            for batch in tqdm(train_loader, desc=f"Training {ite}", leave=False):
                # Parse batch
                x_batch = batch[0].to(device)
                p_batch = batch[1].to(device)
                y_batch = batch[2].to(device) if len(batch) > 2 else None
                
                # Forward pass
                q_batch, s_batch = self(x_batch)
                
                # Compute clustering loss
                c_loss = kld_loss(torch.log(q_batch), p_batch)
                
                # Compute sentiment loss
                s_loss = torch.tensor(0.0).to(device)
                if y_batch is not None:
                    if y_batch.dim() > 1 and y_batch.shape[1] > 1:
                        y_batch = torch.argmax(y_batch, dim=1)
                    y_batch = y_batch.long()
                    s_loss = sentiment_loss(s_batch, y_batch)
                
                # Combined loss
                loss = gamma * c_loss + eta * s_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                cluster_loss += c_loss.item()
                sent_loss += s_loss.item()
                iter_count += 1
            
            # Save checkpoint
            if ite % save_interval == 0 and ite > 0:
                self.save_weights(
                    os.path.join(save_dir, f'FNN_model_{ite}.weights.pth')
                )
    
    def _update_distribution(self, all_embeddings, all_labels, y_sentiment,
                           has_labels, y_pred_last, batch_size, iter_count,
                           update_interval, total_loss, cluster_loss, sent_loss,
                           plot_evolution, plot_interval, plot_dir, ite):
        """Update target distribution and compute metrics."""
        self.eval()
        with torch.no_grad():
            # Compute soft assignments
            q_batch, s_pred_batch = [], []
            loader = DataLoader(all_embeddings, batch_size=batch_size, 
                              shuffle=False)
            
            for batch in tqdm(loader, desc=f"Update {ite}", leave=False):
                batch = batch.to(device)
                q, s = self(batch)
                q_batch.append(q)
                s_pred_batch.append(s)
            
            q = torch.cat(q_batch, dim=0)
            s_pred = torch.cat(s_pred_batch, dim=0)
            
            # Compute target distribution
            p = self.target_distribution(q)
            y_pred = torch.argmax(q, dim=1).cpu().numpy()
            
            # Plot evolution
            if plot_evolution and plot_dir and ite % plot_interval == 0 and ite > 0:
                try:
                    features = self.extract_feature(all_embeddings).cpu().numpy()
                    self.plot_cluster_evolution(
                        features, y_pred, ite, plot_dir, show_plot=False
                    )
                except Exception as e:
                    print(f"Warning: Plot failed at iter {ite}: {e}")
            
            # Compute sentiment accuracy
            acc_sentiment = 0
            if y_sentiment is not None:
                s_pred_label = torch.argmax(s_pred, dim=1).cpu().numpy()
                sentiment_true = (np.argmax(y_sentiment, axis=1) 
                                if len(y_sentiment.shape) > 1 
                                else y_sentiment)
                acc_sentiment = np.mean(s_pred_label == sentiment_true)
                
                # Per-class accuracy
                for cls in np.unique(sentiment_true):
                    mask = sentiment_true == cls
                    cls_acc = np.mean((s_pred_label == sentiment_true) & mask)
                    print(f"  {self.class_labels[cls]}: {cls_acc:.4f}")
        
        # Compute average losses
        avg_loss = total_loss / update_interval if iter_count > 0 else 0
        avg_cluster_loss = cluster_loss / update_interval if iter_count > 0 else 0
        avg_sent_loss = sent_loss / update_interval if iter_count > 0 else 0
        
        return p, y_pred, acc_sentiment, (avg_loss, avg_cluster_loss, avg_sent_loss)
    
    def _log_metrics(self, logwriter, ite, acc_sentiment, avg_loss, 
                    avg_cluster_loss, avg_sent_loss):
        """Log training metrics."""
        logdict = {
            'iter': ite,
            'acc_cluster': 0,
            'nmi': 0,
            'ari': 0,
            'acc_sentiment': round(acc_sentiment, 5),
            'L': round(avg_loss, 5),
            'Lc': round(avg_cluster_loss, 5),
            'Ls': round(avg_sent_loss, 5)
        }
        logwriter.writerow(logdict)
        print(f'Iter {ite}: Lc={avg_cluster_loss:.5f}, '
              f'Ls={avg_sent_loss:.5f}, Acc={acc_sentiment:.5f}')
    
   
    # Cluster Analysis
   
    def get_cluster_assignments(self, x):
        """Get hard cluster assignments."""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)
            cluster_output, _ = self(x)
            return torch.argmax(cluster_output, dim=1).cpu().numpy()
    
    def set_stop_words(self, stop_words):
        """
        Set stopwords for text analysis.
        
        Args:
            stop_words: List, set, or iterable of stopwords
        """
        if isinstance(stop_words, (list, set)):
            self.stop_words = set(stop_words)
        else:
            try:
                self.stop_words = set(stop_words)
            except:
                raise ValueError("stop_words must be a list, set, or iterable")
        return self
    
    def map_texts_to_clusters(self, texts, cluster_assignments):
        """
        Map texts to clusters and extract common words.
        
        Args:
            texts: List of text strings
            cluster_assignments: Cluster assignments for each text
            
        Returns:
            clusters: Dict mapping cluster ID to list of texts
            cluster_words: Dict mapping cluster ID to common words
        """
        clusters = {}
        n = min(len(texts), len(cluster_assignments))
        
        # Group texts by cluster
        for i in range(n):
            cluster = int(cluster_assignments[i])
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(texts[i])
        
        # Extract common words per cluster
        cluster_common_words = {}
        for cluster, cluster_texts in clusters.items():
            all_text = " ".join(cluster_texts).lower()
            words = all_text.split()
            
            # Filter stopwords and short words
            filtered = [w for w in words if w not in self.stop_words and len(w) > 2]
            word_counts = Counter(filtered)
            cluster_common_words[cluster] = word_counts.most_common(20)
        
        return clusters, cluster_common_words
    
    def analyze_clusters(self, x, texts):
        """
        Analyze clusters with text data.
        
        Args:
            x: Input features
            texts: Corresponding text strings
            
        Returns:
            DataFrame with cluster analysis
        """
        cluster_assignments = self.get_cluster_assignments(x)
        text_clusters, cluster_words = self.map_texts_to_clusters(
            texts, cluster_assignments
        )
        
        df_clusters = pd.DataFrame([
            {
                "Cluster": cluster,
                "Common Words": ", ".join([
                    f"{word} ({count})" for word, count in words[:10]
                ]),
                "Text Count": len(text_clusters[cluster])
            }
            for cluster, words in cluster_words.items()
        ]).sort_values(by='Cluster').reset_index(drop=True)
        
        return df_clusters
    
   
    # Topic Management
   
    
    def set_topic(self, cluster_id, topic_name):
        """
        Assign a topic name to a cluster.
        
        Args:
            cluster_id: Cluster ID (0 to n_clusters-1)
            topic_name: Topic name to assign
        """
        if not (isinstance(cluster_id, int) and 0 <= cluster_id < self.n_clusters):
            raise ValueError(
                f"cluster_id must be integer in [0, {self.n_clusters-1}]"
            )
        
        if not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string")
        
        self.topic_mapping[cluster_id] = topic_name
        print(f"✓ Assigned '{topic_name}' to cluster {cluster_id}")
        return self
    
    def reset_topics(self):
        """Reset all topic assignments."""
        self.topic_mapping = {}
        print("✓ All topic assignments reset")
        return self
    
    def get_topic_assignments(self):
        """Get current topic assignments."""
        return self.topic_mapping.copy()
    
   
    # Visualization
   
    
    def plot_sentiment_by_topic(self, data, x, figsize=(15, 6),
                                palette="Set1", negative_color=None,
                                positive_color=None):
        """
        Plot sentiment distribution by cluster topics.
        
        Args:
            data: DataFrame with 'sentiment' column
            x: Feature vectors for cluster assignment
            figsize: Figure size (width, height)
            palette: Seaborn color palette
            negative_color: Specific color for negative sentiment
            positive_color: Specific color for positive sentiment
            
        Returns:
            matplotlib Figure object
        """
        sns.set_style("whitegrid")
        
        # Get cluster assignments
        cluster = self.get_cluster_assignments(x)
        new_data = data.copy()
        new_data['cluster'] = cluster
        
        # Compute sentiment percentages
        sentiment_count = new_data.groupby(
            ['cluster', 'sentiment']
        ).size().unstack(fill_value=0)
        total_reviews = sentiment_count.values.sum()
        sentiment_percent = (sentiment_count / total_reviews) * 100
        
        # Map cluster IDs to topic names
        sentiment_percent['topic'] = sentiment_percent.index.map(
            lambda cid: self.topic_mapping.get(cid, f"Cluster {cid}")
        )
        
        # Prepare data for plotting
        df_melted = sentiment_percent.reset_index().melt(
            id_vars=['cluster', 'topic'],
            value_vars=[0, 1],
            var_name='sentiment',
            value_name='percentage'
        ).sort_values('cluster')
        
        # Setup colors
        if negative_color is None or positive_color is None:
            colors = sns.color_palette(palette, 2)
            neg_color = negative_color or colors[0]
            pos_color = positive_color or colors[1]
        else:
            neg_color = negative_color
            pos_color = positive_color
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sentiment_labels = {
            0: self.class_labels.get(0, 'Negative'),
            1: self.class_labels.get(1, 'Positive')
        }
        
        topics = df_melted['topic'].drop_duplicates().tolist()
        bottoms = {topic: 0 for topic in topics}
        
        # Plot stacked bars
        for sentiment_val, color, label in zip(
            [0, 1], [neg_color, pos_color],
            [sentiment_labels[0], sentiment_labels[1]]
        ):
            data_subset = df_melted[df_melted['sentiment'] == sentiment_val]
            y_vals = data_subset['topic'].tolist()
            x_vals = data_subset['percentage'].tolist()
            left_vals = [bottoms[topic] for topic in y_vals]
            
            ax.barh(y=y_vals, width=x_vals, left=left_vals,
                   label=label, color=color, edgecolor='white',
                   linewidth=0.5)
            
            # Add percentage labels
            for y, x, left in zip(y_vals, x_vals, left_vals):
                if x > 3:
                    ax.text(left + x/2, y, f"{x:.1f}%",
                          va='center', ha='center',
                          color='white', fontsize=9, fontweight='bold')
            
            # Update bottoms
            for topic, x in zip(y_vals, x_vals):
                bottoms[topic] += x
        
        # Styling
        ax.set_xlabel("Percentage of All Reviews", fontsize=11)
        ax.set_ylabel("Cluster Topic", fontsize=11)
        ax.set_title("Sentiment Distribution by Cluster Topic",
                    fontsize=14, fontweight='bold')
        ax.legend(title="Sentiment", title_fontsize=11,
                 frameon=True, fancybox=True, framealpha=0.9,
                 shadow=True, fontsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        sns.despine()
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_evolution(self, embeddings, cluster_assignments, epoch,
                              save_dir='./results/fnnjst', method='tsne',
                              figsize=(6, 6), point_size=20, alpha=0.7,
                              save_plot=True, show_plot=False):
        """
        Plot cluster evolution using dimensionality reduction.
        
        Args:
            embeddings: Feature embeddings to visualize
            cluster_assignments: Cluster assignments for each point
            epoch: Current epoch number
            save_dir: Directory to save plots
            method: Reduction method ('tsne' or 'umap')
            figsize: Figure size (width, height)
            point_size: Size of scatter points
            alpha: Point transparency
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.cpu().detach().numpy()
        else:
            embeddings_np = embeddings
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, random_state=42,
                          init='pca', learning_rate='auto')
            embeddings_2d = reducer.fit_transform(embeddings_np)
        elif method.lower() == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42,
                                  n_neighbors=15, min_dist=0.1)
                embeddings_2d = reducer.fit_transform(embeddings_np)
            except ImportError:
                print("UMAP not available, using t-SNE")
                reducer = TSNE(n_components=2, perplexity=30, random_state=42,
                             init='pca', learning_rate='auto')
                embeddings_2d = reducer.fit_transform(embeddings_np)
        else:
            raise ValueError("Method must be 'tsne' or 'umap'")
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_clusters = np.unique(cluster_assignments)
        n_clusters = len(unique_clusters)
        
        # Select colormap
        if n_clusters <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
        elif n_clusters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, n_clusters))
        
        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_assignments == cluster_id
            cluster_points = embeddings_2d[mask]
            
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=[colors[i]], marker='x', s=point_size, alpha=alpha,
                      label=f'Cluster {cluster_id}')
        
        # Styling
        ax.set_title(f'Epoch {epoch}', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(
                save_dir, f'cluster_evolution_epoch_{epoch}.png'
            )
            plt.savefig(plot_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✓ Saved plot: epoch_{epoch}.png")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def create_evolution_grid(self, save_dir='./results/fnnjst',
                             epochs_to_show=None, grid_cols=3,
                             figsize=(15, 10)):
        """
        Create a grid of cluster evolution plots.
        
        Args:
            save_dir: Directory containing individual plots
            epochs_to_show: List of specific epochs to include
            grid_cols: Number of columns in grid
            figsize: Figure size for the grid
            
        Returns:
            matplotlib Figure object
        """
        import glob
        import re
        import matplotlib.image as mpimg
        
        # Find all evolution plot files
        plot_files = glob.glob(
            os.path.join(save_dir, 'cluster_evolution_epoch_*.png')
        )
        
        if not plot_files:
            print("No cluster evolution plots found")
            return None
        
        # Extract and sort epoch numbers
        epoch_files = []
        for file in plot_files:
            match = re.search(r'epoch_(\d+)\.png', file)
            if match:
                epoch_num = int(match.group(1))
                epoch_files.append((epoch_num, file))
        
        epoch_files.sort(key=lambda x: x[0])
        
        # Filter epochs if specified
        if epochs_to_show is not None:
            epoch_files = [
                (epoch, file) for epoch, file in epoch_files
                if epoch in epochs_to_show
            ]
        
        if not epoch_files:
            print("No matching epoch plots found")
            return None
        
        # Calculate grid dimensions
        n_plots = len(epoch_files)
        grid_rows = (n_plots + grid_cols - 1) // grid_cols
        
        # Create grid
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
        
        if grid_rows == 1:
            axes = [axes] if n_plots == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each epoch
        for i, (epoch, file_path) in enumerate(epoch_files):
            if i < len(axes):
                img = mpimg.imread(file_path)
                axes[i].imshow(img)
                axes[i].set_title(f'Epoch {epoch}',
                                fontsize=12, fontweight='bold')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(epoch_files), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save grid
        grid_path = os.path.join(save_dir, 'cluster_evolution_grid.png')
        plt.savefig(grid_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved evolution grid to {grid_path}")
        
        return fig
    
    def __repr__(self):
        """String representation of the model."""
        return (
            f"SEMTGPU(\n"
            f"  dims={self.dims},\n"
            f"  n_clusters={self.n_clusters},\n"
            f"  alpha={self.alpha},\n"
            f"  device={next(self.parameters()).device}\n"
            f")"
        )
