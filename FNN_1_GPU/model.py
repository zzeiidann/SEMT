import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import pandas as pd
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as linear_assignment

import seaborn as sns
import matplotlib.pyplot as plt
# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class ClusteringLayer(nn.Module):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    """
    def __init__(self, n_clusters, input_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        # Initialize cluster centers as parameters
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, input_dim))
        self._init_weights()
        
    def _init_weights(self):
        # Xavier initialization for cluster centers
        nn.init.xavier_uniform_(self.clusters)

    def forward(self, x):
        """
        student t-distribution, as same as used in t-SNE algorithm.
        q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        
        Arguments:
            x: the variable containing data, shape=(n_samples, n_features)
            
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        # Calculate squared distances
        q = 1.0 / (1.0 + (torch.sum(torch.square(x.unsqueeze(1) - self.clusters.unsqueeze(0)), dim=2) / self.alpha))
        q = q ** ((self.alpha + 1.0) / 2.0)
        # Normalize to make sum = 1
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

class Autoencoder(nn.Module):
    """
    Fully connected auto-encoder model, symmetric.
    """
    def __init__(self, dims, act='relu'):
        super(Autoencoder, self).__init__()
        
        self.dims = dims
        self.n_stacks = len(dims) - 1
        
        # Activation function
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif act == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()  # Default to ReLU
        
        # Create encoder layers
        encoder_layers = []
        
        # Internal layers in encoder
        for i in range(self.n_stacks - 1):
            encoder_layers.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    self.activation
                )
            )
        
        # Hidden layer (no activation)
        encoder_layers.append(nn.Linear(dims[self.n_stacks-1], dims[self.n_stacks]))
        
        # Create decoder layers
        decoder_layers = []
        
        # Internal layers in decoder
        for i in range(self.n_stacks-1, 0, -1):
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(dims[i+1], dims[i]),
                    self.activation
                )
            )
        
        # Output layer (no activation)
        decoder_layers.append(nn.Linear(dims[1], dims[0]))
        
        # Create encoder and decoder
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Xavier initialization for all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """
        Encode the input through the encoder network
        """
        h = x
        for i in range(len(self.encoder_layers)):
            h = self.encoder_layers[i](h)
        return h
    
    def decode(self, h):
        """
        Decode the hidden representation through the decoder network
        """
        for i in range(len(self.decoder_layers)):
            h = self.decoder_layers[i](h)
        return h
    
    def forward(self, x):
        """
        Forward pass through both encoder and decoder
        """
        h = self.encode(x)
        return h, self.decode(h)

class FNNGPU(nn.Module):
    def __init__(self, dims, n_clusters=10, alpha=1.0):
        super(FNNGPU, self).__init__()
        
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Autoencoder component
        self.autoencoder = Autoencoder(dims)
        
        # Clustering layer
        self.clustering = ClusteringLayer(n_clusters, dims[-1], alpha)
        self.topic_mapping = {}
        
        # Sentiment classifier
        sentiment_layers = []
        # First layer
        sentiment_layers.append(nn.Linear(dims[-1], 256))
        sentiment_layers.append(nn.BatchNorm1d(256))
        sentiment_layers.append(nn.GELU())
        sentiment_layers.append(nn.Dropout(0.5))
        
        # Second layer
        sentiment_layers.append(nn.Linear(256, 32))
        sentiment_layers.append(nn.BatchNorm1d(32))
        sentiment_layers.append(nn.GELU())
        sentiment_layers.append(nn.Dropout(0.5))
        
        # Output layer
        sentiment_layers.append(nn.Linear(32, 2))  # 2 classes: positive and negative
        
        self.sentiment_classifier = nn.Sequential(*sentiment_layers)
        
        # Class labels
        self.class_labels = {0: 'negative', 1: 'positive'}
        self.stop_words = set()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize sentiment classifier with Xavier initialization
        for m in self.sentiment_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Get encoded representation from autoencoder
        encoded = self.autoencoder.encode(x)
        
        # Get clustering assignments
        cluster_output = self.clustering(encoded)
        
        # Get sentiment prediction
        sentiment_output = torch.softmax(self.sentiment_classifier(encoded), dim=1)
        
        return cluster_output, sentiment_output
    
    def extract_feature(self, x):
        """Extract bottleneck features from the autoencoder"""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(device) if not isinstance(x, torch.Tensor) else x.to(device)
            encoded = self.autoencoder.encode(x)
        return encoded
    
    def load_weights(self, weights_path):
        """Load model weights from .pth file"""
        checkpoint = torch.load(weights_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights from {weights_path}")
        
    def save_weights(self, weights_path):
        """Save model weights to .pth file"""
        torch.save({
            'model_state_dict': self.state_dict(),
        }, weights_path)
        print(f"Saved weights to {weights_path}")
    
    # Fix for your model.pretrain_autoencoder method
    def pretrain_autoencoder(self, dataset, batch_size=256, epochs=200, learning_rate=0.001):
        """Pretrain the autoencoder using the provided PyTorch dataset"""
        print('Pretraining autoencoder...')
        
        # Extract embeddings from dataset
        embeddings = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, tuple):  # If dataset returns (embedding, label)
                embedding, _ = item
                # Make sure embeddings are on CPU for stacking
                embeddings.append(embedding.cpu())
            else:  # If dataset returns only embedding
                embeddings.append(item.cpu())
        
        # Create tensor dataset of just embeddings
        embeddings_tensor = torch.stack(embeddings)
        # Move the combined tensor to the target device after stacking
        embeddings_tensor = embeddings_tensor.to(device)
        embeddings_dataset = TensorDataset(embeddings_tensor)
        
        print(f"Created tensor dataset with shape: {embeddings_tensor.shape}")
        
        # Create data loader from clean dataset
        data_loader = DataLoader(embeddings_dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Make sure the model is on the same device
        self.autoencoder.to(device)
        
        # Training loop
        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            with tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for data in pbar:
                    # Get inputs (first element of the tuple from DataLoader)
                    inputs = data[0].to(device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward + backward + optimize
                    _, reconstructed = self.autoencoder(inputs)
                    loss = criterion(reconstructed, inputs)
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
        
        # Save weights
        self.save_weights('pretrained_ae.weights.pth')
        print('Autoencoder pretrained and weights saved to pretrained_ae.weights.pth')
            
    @staticmethod
    def target_distribution(q):
        """
        Calculate auxiliary target distribution for clustering
        """
        weight = q ** 2 / torch.sum(q, dim=0)
        return (weight.t() / torch.sum(weight, dim=1)).t()
    
    def compute_class_weights(self, y):
        """
        Compute class weights for imbalanced sentiment classes
        """
        # If y is one-hot encoded, convert to class indices
        if len(y.shape) > 1:
            if isinstance(y, torch.Tensor):
                y_indices = torch.argmax(y, dim=1).cpu().numpy()
            else:
                y_indices = np.argmax(y, axis=1)
        else:
            if isinstance(y, torch.Tensor):
                y_indices = y.cpu().numpy()
            else:
                y_indices = y
        
        # Calculate class distribution
        unique_classes, class_counts = np.unique(y_indices, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        # Compute balanced class weights
        class_weights = {}
        total_samples = len(y_indices)
        n_classes = len(unique_classes)
        
        for i, c in enumerate(unique_classes):
            class_weights[c] = total_samples / (n_classes * class_counts[i])
        
        print(f"Computed class weights: {class_weights}")
        return class_weights
    
    def predict_clusters(self, x):
        """
        Predict cluster assignments for input data
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(device) if not isinstance(x, torch.Tensor) else x.to(device)
            cluster_output, _ = self(x)
            return torch.argmax(cluster_output, dim=1).cpu().numpy()
    
    def predict_sentiment(self, x):
        """
        Predict sentiment for input data
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(device) if not isinstance(x, torch.Tensor) else x.to(device)
            _, sentiment_output = self(x)
            return torch.argmax(sentiment_output, dim=1).cpu().numpy()
    
    
    def predict(self, inputs, bert_model=None):
        """
        Predict clusters and sentiment for text inputs or embeddings
        
        Returns the assigned topic name if available for the predicted cluster
        """
        self.eval()
        if isinstance(inputs, str):
            inputs = [inputs]

        if isinstance(inputs, list) and isinstance(inputs[0], str):
            tokenizer = AutoTokenizer.from_pretrained(bert_model if isinstance(bert_model, str) else "indolem/indobert-base-uncased")
            tokens = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(device)

            with torch.no_grad():
                if not callable(bert_model):
                    bert_model = AutoModel.from_pretrained(bert_model if isinstance(bert_model, str) else "indolem/indobert-base-uncased")
                    bert_model.to(device)
                
                outputs = bert_model(**tokens)
            
            embeddings = outputs.last_hidden_state[:, 0, :]

        elif isinstance(inputs, torch.Tensor):
            embeddings = inputs
        else:
            embeddings = torch.tensor(inputs, dtype=torch.float32).to(device)
        
        # Get predictions
        with torch.no_grad():
            cluster_output, sentiment_output = self(embeddings)
        
        # Get the predicted clusters and sentiments
        cluster_preds = torch.argmax(cluster_output, dim=1).cpu().numpy()
        sentiment_preds = torch.argmax(sentiment_output, dim=1).cpu().numpy()
        
        # Prepare results
        results = []
        for i in range(len(sentiment_preds)):
            sentiment_label = self.class_labels[sentiment_preds[i]]
            cluster_id = int(cluster_preds[i])
            
            # Use the topic name if assigned, otherwise use the cluster ID
            cluster_or_topic = self.topic_mapping.get(cluster_id, cluster_id)
            
            result = {
                'sentiment': sentiment_label,
                'topic': cluster_or_topic
            }
            results.append(result)
        
        return results
    def clustering_with_sentiment(self, dataset, gamma=0.7, eta=1, optimizer_type='sgd', learning_rate=0.001, momentum=0.9,
                        tol=1e-3, update_interval=140, batch_size=128, maxiter=2e4, 
                        save_dir='./results/fnnjst'):
        """
        Train the model with joint clustering and sentiment tasks
        """
        print('Update interval', update_interval)

        # Create directories for saving
        os.makedirs(save_dir, exist_ok=True)
        # Collect all embeddings and labels
        embeddings = []
        labels = []

        # Iterate through the dataset and collect data
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, tuple) and len(item) == 2:  # If dataset returns (embedding, label)
                embedding, label = item
                # Store tensors on CPU initially
                embeddings.append(embedding.cpu())
                labels.append(label.cpu())
            else: 
                # If only embeddings are returned
                embeddings.append(item.cpu() if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.float32))

        # Check if embeddings have consistent shapes
        shapes = [emb.shape for emb in embeddings]
        if len(set(shapes)) > 1:
            print(f"Warning: Inconsistent embedding shapes detected: {set(shapes)}")
            print("Attempting to handle this case...")

        # Stack embeddings
        embeddings_tensor = torch.stack(embeddings)
        # Move model to device first to ensure it's on the right device
        self.to(device)
        # Move data to the same device as the model
        embeddings_tensor = embeddings_tensor.to(device)

        # Create the final dataset
        if labels:
            # Convert labels to tensor if they exist
            labels_tensor = torch.stack(labels).to(device)
            combined_dataset = TensorDataset(embeddings_tensor, labels_tensor)
            print(f"Created dataset with {len(combined_dataset)} samples, embedding shape: {embeddings_tensor.shape}, label shape: {labels_tensor.shape}")
        else:
            combined_dataset = TensorDataset(embeddings_tensor)
            print(f"Created dataset with {len(combined_dataset)} samples, embedding shape: {embeddings_tensor.shape}")

        data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting features"):
                if len(batch) > 1:  # If the batch contains both embeddings and labels
                    embeddings, labels = batch
                    # Ensure embeddings are on the correct device
                    embeddings = embeddings.to(device)
                    if not isinstance(embeddings, torch.Tensor):
                        embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
                    all_embeddings.append(embeddings)
                    # Ensure labels are on the correct device
                    labels = labels.to(device)
                    all_labels.append(labels)
                else:  # If the batch contains only embeddings
                    embeddings = batch[0]  # Get the first (and only) item from the tuple
                    # Ensure embeddings are on the correct device
                    embeddings = embeddings.to(device)
                    try:
                        if not isinstance(embeddings, torch.Tensor):
                            embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
                    except Exception as e:
                        print(f"Error converting to tensor: {e}")
                        continue
                    all_embeddings.append(embeddings)

            
            if all_embeddings:
                all_embeddings = torch.cat(all_embeddings, dim=0).to(device)
            else:
                raise ValueError("No embeddings were extracted from the dataset")
                
            if all_labels:
                # Create a tensor from all_labels if there are items in the list
                all_labels = torch.cat(all_labels, dim=0).to(device)
                has_labels = True
            else:
                all_labels = None
                has_labels = False
        
        # Create a tensor dataset for batch training
        if has_labels and isinstance(all_labels, torch.Tensor) and all_labels.numel() > 0:
            x_dataset = TensorDataset(all_embeddings, all_labels)
            y_sentiment = all_labels.cpu().numpy()
            
            # Compute class weights for handling imbalanced classes
            sentiment_class_weights = self.compute_class_weights(y_sentiment)
            class_weight_tensor = torch.tensor([sentiment_class_weights[i] for i in range(len(self.class_labels))], 
                                            dtype=torch.float32).to(device)
        else:
            x_dataset = TensorDataset(all_embeddings)
            y_sentiment = None
            sentiment_class_weights = None
            class_weight_tensor = None
        
        # Create data loader for batch training
        train_loader = DataLoader(x_dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizers
        if optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, alpha=0.99)
        elif optimizer_type == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'adamax':
            optimizer = optim.Adamax(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'asgd':
            optimizer = optim.ASGD(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'nadam':
            optimizer = optim.NAdam(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}", f"Supported types are: 'sgd', 'adam', 'adamw', 'rmsprop', 'adagrad', 'adamax', 'asgd', 'adadelta', 'nadam'")

        # Loss functions
        kld_loss = nn.KLDivLoss(reduction='batchmean')
        if class_weight_tensor is not None:
            sentiment_loss = nn.CrossEntropyLoss(weight=class_weight_tensor)
        else:
            sentiment_loss = nn.CrossEntropyLoss()
        
        # Initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        # Ensure model is in eval mode and using the proper device
        self.eval()
        features = self.extract_feature(all_embeddings).cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(features)
        y_pred_last = np.copy(y_pred)
        
        # Set cluster centers as initial weights - ensure on correct device
        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
        self.clustering.clusters.data = cluster_centers
        
        # Logging file
        logfile = open(os.path.join(save_dir, 'idec_sentiment_log.csv'), 'w', newline='')
        fieldnames = ['iter', 'acc_cluster', 'nmi', 'ari', 'acc_sentiment', 'L', 'Lc', 'Ls']
        logwriter = csv.DictWriter(logfile, fieldnames=fieldnames)
        logwriter.writeheader()
        
        save_interval = len(train_loader) * 5  # 5 epochs
        print('Save interval', save_interval)
        
        # Training loop
        self.train()
        iter_count = 0
        total_loss = cluster_loss = sent_loss = 0
        
        gamma = gamma  # Weight for clustering loss
        eta = eta    # Weight for sentiment loss
        
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                self.eval()
                with torch.no_grad():
                    q_batch = []
                    s_pred_batch = []
                    
                    for batch in tqdm(DataLoader(all_embeddings, batch_size=batch_size, shuffle=False), 
                                    desc=f"Updating distribution (iter {ite})"):
                       
                        batch = batch.to(device)
                        q, s = self(batch)  
                        q_batch.append(q)
                        s_pred_batch.append(s)
                    
                    q = torch.cat(q_batch, dim=0)
                    s_pred = torch.cat(s_pred_batch, dim=0)
                    
                    p = self.target_distribution(q)
                    
                    y_pred = torch.argmax(q, dim=1).cpu().numpy()
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)

                    pred_docker = []
                    q_docker = []
                    p_docker = []
                    epochs_docker = []

                    if ite % update_interval == 0:
                        for i in range(self.n_clusters):
                            pred_docker.append(y_pred[y_pred == i])
                            q_docker.append(q[y_pred == i])
                            p_docker.append(p[y_pred == i])
                            epochs_docker.append(ite)

                    
                    #make df to plot 
                    plot_cluster_df = pd.DataFrame({
                        'pred': y_pred_last,
                        'q': q.cpu().numpy(),
                        'p': p.cpu().numpy(),
                        'epoch': ite
                    })
                    
                    print(plot_cluster_df)
                    
                    if y_sentiment is not None:
                        s_pred_label = torch.argmax(s_pred, dim=1).cpu().numpy()
                        
                        if len(y_sentiment.shape) > 1:
                            sentiment_true_label = np.argmax(y_sentiment, axis=1)
                        else:
                            sentiment_true_label = y_sentiment
                            
                        acc_sentiment = np.sum(s_pred_label == sentiment_true_label).astype(np.float32) / s_pred_label.shape[0]
                        
                        
                        if len(np.unique(sentiment_true_label)) > 1:
                            for cls in np.unique(sentiment_true_label):
                                cls_mask = sentiment_true_label == cls
                                cls_acc = np.sum((s_pred_label == sentiment_true_label) & cls_mask).astype(np.float32) / np.sum(cls_mask)
                                print(f"Class {self.class_labels[cls]} accuracy: {np.round(cls_acc, 5)}")
                    else:
                        acc_sentiment = 0
                    
                    acc_cluster = 0
                    nmi = 0
                    ari = 0
                
                avg_loss = total_loss / update_interval if iter_count > 0 else 0
                avg_cluster_loss = cluster_loss / update_interval if iter_count > 0 else 0
                avg_sent_loss = sent_loss / update_interval if iter_count > 0 else 0
                
                logdict = {
                    'iter': ite, 
                    'acc_cluster': acc_cluster, 
                    'nmi': nmi, 
                    'ari': ari, 
                    'acc_sentiment': np.round(acc_sentiment, 5),
                    'L': np.round(avg_loss, 5), 
                    'Lc': np.round(avg_cluster_loss, 5), 
                    'Ls': np.round(avg_sent_loss, 5)
                }
                logwriter.writerow(logdict)
                print(f'Iter {ite}: Cluster Loss {avg_cluster_loss:.5f}, Sentiment Loss {avg_sent_loss:.5f}, Acc_sentiment {acc_sentiment:.5f}; loss={avg_loss:.5f}')
                
                # Reset counters
                total_loss = cluster_loss = sent_loss = 0
                
                # Check stop criterion based on cluster stability
                if ite > 0 and delta_label < tol:
                    print(f'delta_label {delta_label} < tol {tol}')
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break
                
                # Update dataset with new target distribution - ensure all on same device
                if has_labels and isinstance(all_labels, torch.Tensor) and all_labels.numel() > 0:
                    train_loader = DataLoader(TensorDataset(all_embeddings, p, all_labels), 
                                        batch_size=batch_size, shuffle=True)
                else:
                    train_loader = DataLoader(TensorDataset(all_embeddings, p), 
                                        batch_size=batch_size, shuffle=True)
            
            # Train on batch
            self.train()
            for batch in tqdm(train_loader, desc=f"Training iter {ite}", leave=False):
                if y_sentiment is not None:
                    if len(batch) == 3:  # With labels
                        x_batch, p_batch, y_batch = batch
                    else:
                        x_batch, p_batch = batch
                        y_batch = None
                else:
                    x_batch, p_batch = batch
                    y_batch = None
                
                # Explicitly ensure all batch elements are on the correct device
                x_batch = x_batch.to(device)
                p_batch = p_batch.to(device)
                if y_batch is not None:
                    y_batch = y_batch.to(device)
                
                # Forward pass
                q_batch, s_batch = self(x_batch)
                
                # Compute loss
                c_loss = kld_loss(torch.log(q_batch), p_batch)
                s_loss = torch.tensor(0.0).to(device)
                
                if y_batch is not None:
                    if y_batch.dim() > 1 and y_batch.shape[1] > 1: 
                        y_batch = torch.argmax(y_batch, dim=1)
                    y_batch = y_batch.long() 
                    s_loss = sentiment_loss(s_batch, y_batch)
                
                # Combined loss
                loss = gamma * c_loss + eta * s_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                cluster_loss += c_loss.item()
                sent_loss += s_loss.item() if y_batch is not None else 0
                
                iter_count += 1
            
            # Save intermediate model
            if ite % save_interval == 0 and ite > 0:
                model_path = os.path.join(save_dir, f'FNN_model_{ite}.weights.pth')
                self.save_weights(model_path)
        
        # Save the trained model
        logfile.close()
        model_path = os.path.join(save_dir, 'FNN_model_final.weights.pth')
        self.save_weights(model_path)
        
        # Return final predictions
        self.eval()
        with torch.no_grad():
            q, s_pred = self(all_embeddings)
            y_pred = torch.argmax(q, dim=1).cpu().numpy()
            if y_sentiment is not None and has_labels:
                s_pred = s_pred.cpu().numpy()
                return y_pred, s_pred
            else:
                return y_pred

    def get_cluster_assignments(self, x):
        """
        Get cluster assignments for a batch of inputs
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                x = x.to(device)
            else:
                x = torch.tensor(x, dtype=torch.float32).to(device)
            
            cluster_output, _ = self(x)
            return torch.argmax(cluster_output, dim=1).cpu().numpy()
    
    def set_stop_words(self, stop_words):
        """
        Set custom stopwords for cluster text analysis
        """
        if isinstance(stop_words, list):
            self.stop_words = set(stop_words)   
        elif isinstance(stop_words, set):
            self.stop_words = stop_words
        else:
            try:
                self.stop_words = set(stop_words)
            except:
                raise ValueError("stop_words must be a list, set, or convertible to a set")
        
        return self
    
    def map_texts_to_clusters(self, texts, cluster_assignments):
        """
        Map texts to their assigned clusters and extract common words
        """
        clusters = {}
        
        n = min(len(texts), len(cluster_assignments))
        
        for i in range(n):
            cluster = int(cluster_assignments[i])
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(texts[i])
        
        cluster_common_words = {}
        for cluster, cluster_texts in clusters.items():
            all_text = " ".join(cluster_texts)
            
            words = all_text.lower().split()
            
            filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
            
            word_counts = Counter(filtered_words)
        
            top_words = word_counts.most_common(20)
            cluster_common_words[cluster] = top_words
        
        return clusters, cluster_common_words
    
    def analyze_clusters(self, x, texts):
        """
        Analyze clusters by getting assignments and mapping texts
        """
        cluster_assignments = self.get_cluster_assignments(x)
        text_clusters, cluster_words = self.map_texts_to_clusters(texts, cluster_assignments)
    
        df_clusters = pd.DataFrame([
            {"Cluster": cluster, "Common Words": ", ".join([f"{word} ({count})" for word, count in words[:10]]),
             "Text Count": len(text_clusters[cluster])}
            for cluster, words in cluster_words.items()
        ]).sort_values(by=['Cluster']).reset_index(drop=True)
        
        return df_clusters
    
    def set_topic(self, cluster_id, topic_name):
        """
        Assign a topic name to a specific cluster
        
        Parameters:
        -----------
        cluster_id : int
            The ID of the cluster to assign a topic name to
        topic_name : str
            The name of the topic to assign to the cluster
        
        Returns:
        --------
        self : FNNGPU
            The instance of the class for method chaining
        """
        if not isinstance(cluster_id, int) or cluster_id < 0 or cluster_id >= self.n_clusters:
            raise ValueError(f"cluster_id must be an integer between 0 and {self.n_clusters-1}")
        
        if not isinstance(topic_name, str):
            raise ValueError("topic_name must be a string")
        
        self.topic_mapping[cluster_id] = topic_name
        print(f"Assigned topic '{topic_name}' to cluster {cluster_id}")
        return self

    def reset_topics(self):
        """
        Reset all topic assignments
        
        Returns:
        --------
        self : FNNGPU
            The instance of the class for method chaining
        """
        self.topic_mapping = {}
        print("All topic assignments have been reset")
        return self

    def get_topic_assignments(self):
        """
        Get the current topic assignments
        
        Returns:
        --------
        dict
            A dictionary mapping cluster IDs to topic names
        """
        return self.topic_mapping
    
    def plot_sentiment_by_topic(self, data, x, figsize=(15, 6), 
                          palette="Set1", negative_color=None, positive_color=None):
        """
        Plot sentiment distribution by cluster topics assigned in the model
        with enhanced Seaborn color aesthetics.
        
        Parameters:
        -----------
        data : pandas DataFrame
            The DataFrame containing sentiment data with a 'sentiment' column
        x : tensor or numpy array
            The feature vectors to use for cluster assignment
        figsize : tuple, optional
            Figure size (width, height) in inches
        palette : str or seaborn palette, optional
            Color palette to use if specific colors are not provided
        negative_color : str, optional
            Specific color for negative sentiment (overrides palette)
        positive_color : str, optional
            Specific color for positive sentiment (overrides palette)
        
        Returns:
        --------
        matplotlib.figure.Figure
            The matplotlib figure object
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Set Seaborn style
        sns.set_style("whitegrid")
        
        # Get cluster assignments
        cluster = self.get_cluster_assignments(x)
        
        # Add cluster column to the data
        new_data = data.copy()
        new_data['cluster'] = cluster
        
        # Group by cluster and sentiment
        sentiment_count = new_data.groupby(['cluster', 'sentiment']).size().unstack(fill_value=0)
        total_reviews = sentiment_count.values.sum()
        sentiment_percent = (sentiment_count / total_reviews) * 100
        
        # Create a mapping from cluster ID to topic name using existing topic_mapping
        cluster_topics = {}
        for cluster_id in sentiment_percent.index:
            # Use the assigned topic if available, otherwise use "Cluster X"
            cluster_topics[cluster_id] = self.topic_mapping.get(cluster_id, f"Cluster {cluster_id}")
        
        # Add topic names to the DataFrame
        sentiment_percent['topic'] = sentiment_percent.index.map(cluster_topics)
        
        # Melt the DataFrame for easier plotting
        df_melted = sentiment_percent.reset_index().melt(
            id_vars=['cluster', 'topic'],
            value_vars=[0, 1],  # Assuming 0=negative, 1=positive
            var_name='sentiment',
            value_name='percentage'
        )
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Get colors from seaborn palette
        if negative_color is None or positive_color is None:
            colors = sns.color_palette(palette, 2)
            neg_color = negative_color if negative_color else colors[0]
            pos_color = positive_color if positive_color else colors[1]
        else:
            neg_color = negative_color
            pos_color = positive_color
        
        # Map sentiment values to class labels if available
        sentiment_labels = {
            0: self.class_labels.get(0, 'Negative'),
            1: self.class_labels.get(1, 'Positive')
        }
        
        # Sort data by cluster ID for consistent ordering
        df_melted = df_melted.sort_values('cluster')
        
        # Get unique topics in cluster order
        topics = df_melted['topic'].drop_duplicates().tolist()
        
        # Initialize bottoms for stacked bars
        bottoms = {topic: 0 for topic in topics}
        
        # Plot each sentiment category
        for sentiment_val, color, label in zip([0, 1], [neg_color, pos_color], 
                                            [sentiment_labels[0], sentiment_labels[1]]):
            data = df_melted[df_melted['sentiment'] == sentiment_val]
            
            # Prepare data for plotting
            y_vals = data['topic'].tolist()
            x_vals = data['percentage'].tolist()
            left_vals = [bottoms[topic] for topic in y_vals]
            
            bar = plt.barh(
                y=y_vals,
                width=x_vals,
                left=left_vals,
                label=label,
                color=color,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Add percentage labels on bars
            for y, x, left in zip(y_vals, x_vals, left_vals):
                if x > 3:  # Only show label if percentage is significant enough
                    plt.text(left + x/2, y, f"{x:.1f}%", va='center', ha='center', 
                            color='white', fontsize=9, fontweight='bold')
            
            # Update bottoms for next stack
            for topic, x in zip(y_vals, x_vals):
                bottoms[topic] += x
        
        # Enhance the plot with Seaborn styling
        plt.xlabel("Percentage of All Reviews", fontsize=11)
        plt.ylabel("Cluster Topic", fontsize=11)
        plt.title("Sentiment Distribution by Cluster Topic", fontsize=14, fontweight='bold')
        
        # Add a legend with enhanced style
        legend = plt.legend(title="Sentiment", title_fontsize=11, 
                frameon=True, fancybox=True, framealpha=0.9, 
                shadow=True, fontsize=10)
        
        # Add subtle grid lines only on x-axis
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Remove top and right spines
        sns.despine()
        
        plt.tight_layout()
        
        return plt.gcf()  # Return the figure
    
    
