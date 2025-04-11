from time import time
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, BatchNormalization, Dropout, Activation
import keras.backend as K

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, AutoModel
from collections import Counter
import pandas as pd 

from .DEC import cluster_acc, ClusteringLayer, autoencoder
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class FNN(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):

        super(FNN, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.autoencoder = autoencoder(self.dims)
        self.class_labels = {0: 'negative', 1: 'positive'}
        self.stop_words = set()

    def initialize_model(self, ae_weights=None, gamma=0.1, eta=1.0, optimizer=SGD(learning_rate=0.001, momentum=0.9)):
        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('Pretrained AE weights are loaded successfully.')
        else:
            print('ae_weights must be given. E.g.')
            print('    python FNN_1/model.py --ae_weights weights.h5')
            exit()

        # Get the encoder part from autoencoder
        hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)
        
        # Define the sentiment classifier
        hidden_size = self.dims[-1]  # Size of the bottleneck encoding
        

        x = Dense(128)(hidden)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)  # Changed from gelu to 'relu'
        x = Dropout(0.4)(x)

        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)  # Changed from gelu to 'relu'
        x = Dropout(0.4)(x)
        
        sentiment_output = Dense(2, activation='softmax', name='sentiment')(x)
        
        # Create clustering layer
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        
        # Create the combined model
        self.model = Model(inputs=self.autoencoder.input,
                          outputs=[clustering_layer, sentiment_output])
        
        # Compile with multiple losses
        self.model.compile(loss={'clustering': 'kld', 'sentiment': 'categorical_crossentropy'},
                          loss_weights=[gamma, eta],  # Balance between clustering and sentiment tasks
                          optimizer=optimizer)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):
        encoder = Model(self.model.input, self.model.get_layer('encoder_%d' % (self.n_stacks - 1)).output)
        return encoder.predict(x)

    def predict_clusters(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)
        
    def predict_sentiment(self, x):
        _, s = self.model.predict(x, verbose=0)
        return s.argmax(1)

    def predict(self, inputs, bert_model=None):
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
                else:
                    bert_model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
                
                outputs = bert_model(**tokens)
            
            embeddings_tensor = outputs.last_hidden_state[:, 0, :]
            embeddings_numpy = np.expand_dims(embeddings_tensor.cpu().detach().numpy(), axis=0)
            embeddings_numpy = embeddings_numpy.squeeze(0)

        elif isinstance(inputs, torch.Tensor):
            embeddings_tensor = inputs
            embeddings_numpy = np.expand_dims(embeddings_tensor.cpu().detach().numpy(), axis=0)
            embeddings_numpy = embeddings_numpy.squeeze(0)
        else:
            raise ValueError("Input must be a list of texts or embeddings tensor")
        
        cluster_output, sentiment_output = self.model.predict(embeddings_numpy, verbose=0)
        
        # Get the predicted clusters and sentiments
        cluster_preds = cluster_output.argmax(1)
        sentiment_preds = sentiment_output.argmax(1)
        # sentiment_probs = np.max(sentiment_output, axis=1)
        
        # Prepare results
        results = []
        for i in range(len(sentiment_preds)):
            sentiment_label = self.class_labels[sentiment_preds[i]]
            result = {
                'sentiment': sentiment_label,
                # 'sentiment_confidence': float(sentiment_probs[i]),
                'cluster': int(cluster_preds[i])
            }
            
            # if isinstance(inputs, list) and isinstance(inputs[0], str):
            #     result['text'] = inputs[i]               
            results.append(result)
        
        return results

    def get_cluster_assignments(self, x):
        """
        Get cluster assignments for a batch of inputs
        
        Args:
            x: Input features as numpy array
            
        Returns:
            numpy array of cluster assignments
        """
        x = np.expand_dims(x.cpu().detach().numpy(), axis=0) if isinstance(x, torch.Tensor) else np.expand_dims(x, axis=0)
        x = x.squeeze(0)
        cluster_output, _ = self.model.predict(x, verbose=0)
        return cluster_output.argmax(1)

    def set_stop_words(self, stop_words):
        """
        Set custom stopwords for cluster text analysis
        
        Args:
            stop_words: List or set of stopwords to use when analyzing text clusters
            
        Returns:
            self: For method chaining
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
        
        Args:
            texts: List of text strings
            cluster_assignments: Numpy array of cluster assignments
            
        Returns:
            tuple: (clusters dict mapping cluster IDs to texts, 
                common_words dict mapping cluster IDs to word frequencies)
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
        
        Args:
            x: Input features as numpy array
            texts: List of corresponding text strings
            
        Returns:
            DataFrame with cluster analysis
        """
        cluster_assignments = self.get_cluster_assignments(x)
        text_clusters, cluster_words = self.map_texts_to_clusters(texts, cluster_assignments)
    
        df_clusters = pd.DataFrame([
            {"Cluster": cluster, "Common Words": ", ".join([f"{word} ({count})" for word, count in words[:10]]),
             "Text Count": len(text_clusters[cluster])}
            for cluster, words in cluster_words.items()
        ]).sort_values(by=['Cluster']).reset_index(drop=True)
        
        print("\n============== CLUSTER ANALYSIS ==============")
        print(df_clusters)
        
        return df_clusters
    
    def pretrain_autoencoder(self, dataset, batch_size=256, epochs=200, optimizer='adam'):
        """
        Pretrain the autoencoder using the provided PyTorch dataset
        """
        print('Pretraining autoencoder...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        
        # Convert PyTorch dataset to numpy arrays
        embeddings = []
        
        # Extract embeddings from the dataset
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, tuple):  # If dataset returns (embedding, label)
                embedding, _ = item
                embeddings.append(embedding.cpu().numpy())
            else:  # If dataset returns only embedding
                embeddings.append(item.cpu().numpy())
        
        # Convert to numpy array
        x = np.array(embeddings)
        
        print(f"Converted dataset to numpy array with shape: {x.shape}")
        
        # Train the autoencoder
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)
        
        # Save the weights
        self.autoencoder.save_weights('pretrained_ae.weights.h5')
        print('Autoencoder pretrained and weights saved to pretrained_ae.weights.h5')
        
        # Initialize encoder from the trained autoencoder
        hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)
        
        return self.autoencoder.get_weights()
    
    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compute_class_weights(self, y):
        """
        Compute class weights for imbalanced sentiment classes
        
        Args:
            y: Sentiment labels (can be one-hot encoded or class indices)
            
        Returns:
            Dictionary of class weights
        """
        # If y is one-hot encoded, convert to class indices
        if len(y.shape) > 1:
            y_indices = np.argmax(y, axis=1)
        else:
            y_indices = y
            
        # Calculate class distribution
        unique_classes, class_counts = np.unique(y_indices, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        # Compute balanced class weights
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_indices)
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        print(f"Computed class weights: {class_weight_dict}")
        return class_weight_dict

    def clustering_with_sentiment(self, dataset, tol=1e-3, update_interval=140, maxiter=2e4, 
                                 save_dir='./results/fnnjst'):
        """
        dataset: CachedBERTDataset instance containing texts and labels
        """
        print('Update interval', update_interval)
        
        # Convert PyTorch dataset to NumPy arrays for Keras
        embeddings = []
        sentiment_labels = []
        
        # Extract embeddings and labels from dataset
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, tuple):  # If dataset returns (embedding, label)
                embedding, label = item
                embeddings.append(embedding.cpu().numpy())
                sentiment_labels.append(label.cpu().numpy())
            else:  # If dataset returns only embedding
                embeddings.append(item.cpu().numpy())
                
        x = np.array(embeddings)
        
        if sentiment_labels:
            y_sentiment = np.array(sentiment_labels)
            # One-hot encoding for categorical crossentropy
            from keras.utils import to_categorical
            if len(y_sentiment.shape) == 1:
                y_sentiment = to_categorical(y_sentiment, num_classes=2)
            
            # Compute class weights for handling imbalanced classes
            sentiment_class_weights = self.compute_class_weights(y_sentiment)
        else:
            print("Warning: No labels found in dataset. Clustering only.")
            y_sentiment = None
            sentiment_class_weights = None
            
        save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
        print('Save interval', save_interval)

        # Initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = y_pred
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/idec_sentiment_log.csv', 'w')
        fieldnames = ['iter', 'acc_cluster', 'nmi', 'ari', 'acc_sentiment', 'L', 'Lc', 'Ls']
        logwriter = csv.DictWriter(logfile, fieldnames=fieldnames)
        logwriter.writeheader()

        loss = [0, 0, 0]  # Total loss, clustering loss, sentiment loss
        index = 0
        
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, s_pred = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # Update auxiliary target distribution
                
                # Evaluate clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                
                # Compute sentiment prediction accuracy if labels available
                if y_sentiment is not None:
                    s_pred_label = s_pred.argmax(1)
                    sentiment_true_label = y_sentiment.argmax(1) if len(y_sentiment.shape) > 1 else y_sentiment
                    acc_sentiment = np.sum(s_pred_label == sentiment_true_label).astype(np.float32) / s_pred_label.shape[0]
                    
                    # Compute per-class accuracy to monitor imbalance effects
                    if len(np.unique(sentiment_true_label)) > 1:
                        for cls in np.unique(sentiment_true_label):
                            cls_mask = sentiment_true_label == cls
                            cls_acc = np.sum((s_pred_label == sentiment_true_label) & cls_mask).astype(np.float32) / np.sum(cls_mask)
                            print(f"Class {self.class_labels[cls]} accuracy: {np.round(cls_acc, 5)}")
                else:
                    acc_sentiment = 0
                
                # For now, we don't have ground truth cluster labels
                acc_cluster = nmi = ari = 0
                
                loss = np.round(loss, 5)
                logdict = dict(iter=ite, acc_cluster=acc_cluster, nmi=nmi, ari=ari, 
                              acc_sentiment=np.round(acc_sentiment, 5),
                              L=loss[0], Lc=loss[1], Ls=loss[2])
                logwriter.writerow(logdict)
                print('Iter', ite,': Cluster Loss', loss[1], ', Sentiment Loss', loss[2] , ', Acc_sentiment', np.round(acc_sentiment, 5), '; loss=', loss)
        

                # Check stop criterion based on cluster stability
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break
            
            # Train on batch with class weights for sentiment
            if y_sentiment is not None:
                if (index + 1) * self.batch_size > x.shape[0]:
                    batch_x = x[index * self.batch_size::]
                    batch_p = p[index * self.batch_size::]
                    batch_y_sentiment = y_sentiment[index * self.batch_size::]
                    
                    # Apply class weights manually to sentiment loss
                    if sentiment_class_weights:
                        # Get sample weights based on class labels
                        sample_weights = np.ones(batch_y_sentiment.shape[0])
                        for i, label in enumerate(np.argmax(batch_y_sentiment, axis=1)):
                            sample_weights[i] = sentiment_class_weights[label]
                            
                        # Pass sample weights as a list - None for clustering, weights for sentiment
                        loss = self.model.train_on_batch(
                            x=batch_x,
                            y=[batch_p, batch_y_sentiment],
                            sample_weight=[None, sample_weights]
                        )
                    else:
                        loss = self.model.train_on_batch(
                            x=batch_x,
                            y=[batch_p, batch_y_sentiment]
                        )
                    index = 0
                else:
                    batch_x = x[index * self.batch_size:(index + 1) * self.batch_size]
                    batch_p = p[index * self.batch_size:(index + 1) * self.batch_size]
                    batch_y_sentiment = y_sentiment[index * self.batch_size:(index + 1) * self.batch_size]
                    
                    # Apply class weights manually to sentiment loss
                    if sentiment_class_weights:
                        # Get sample weights based on class labels
                        sample_weights = np.ones(batch_y_sentiment.shape[0])
                        for i, label in enumerate(np.argmax(batch_y_sentiment, axis=1)):
                            sample_weights[i] = sentiment_class_weights[label]
                            
                        # Pass sample weights as a list - None for clustering, weights for sentiment
                        loss = self.model.train_on_batch(
                            x=batch_x,
                            y=[batch_p, batch_y_sentiment],
                            sample_weight=[None, sample_weights]
                        )
                    else:
                        loss = self.model.train_on_batch(
                            x=batch_x,
                            y=[batch_p, batch_y_sentiment]
                        )
                    index += 1
            
            # Save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/FNN_model_' + str(ite) + '.weights' + '.h5')
                self.model.save_weights(save_dir + '/FNN_model_' + str(ite) + '.weights' + '.h5')
        
        # Save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/FNN_model_final.weights.h5')
        self.model.save_weights(save_dir + '/FNN_model_final.weights.h5')
        
        return y_pred, s_pred if y_sentiment is not None else y_pred

    def evaluate_sentiment_performance(self, x, y_true):
        """
        Evaluate sentiment classification performance with metrics for imbalanced data
        
        Args:
            x: Input features 
            y_true: True sentiment labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        _, y_pred_probs = self.model.predict(x, verbose=0)
        y_pred = y_pred_probs.argmax(1)
        
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        cm = confusion_matrix(y_true, y_pred)
        
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        # Print evaluation results
        print("\n============ SENTIMENT EVALUATION ============")
        print(f"Confusion Matrix:\n{cm}")
        print("\nPer-class metrics:")
        for i, class_name in self.class_labels.items():
            print(f"Class {class_name}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Accuracy={per_class_acc[i]:.3f}")
        
        # Calculate macro and weighted averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        print(f"\nMacro Avg: Precision={macro_precision:.3f}, Recall={macro_recall:.3f}, F1={macro_f1:.3f}")
        print(f"Weighted Avg: Precision={weighted_precision:.3f}, Recall={weighted_recall:.3f}, F1={weighted_f1:.3f}")
        
        return {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_accuracy': per_class_acc,
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1
            }
        }