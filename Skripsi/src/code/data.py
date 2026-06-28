import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

class CachedBERTDataset(Dataset):
    def __init__(
        self, 
        texts, 
        labels=None, 
        bert_model="bert-base-uncased", 
        max_length=128, 
        cuda=True, 
        testing_mode=False,
        return_texts=True  #  CRITICAL: Enable for TF-IDF plots
    ):
        """
        Dataset that caches BERT embeddings for text data
        
        Args:
            texts: List of text strings to encode
            labels: Optional list of labels corresponding to the texts
            bert_model: Pre-trained BERT model name to use
            max_length: Maximum sequence length for BERT tokenizer
            cuda: Whether to use GPU acceleration
            testing_mode: If True, only use a small subset of data
            return_texts: If True, return (embedding, label, text) tuple for TF-IDF plots
        """
        self.texts = texts
        self.labels = labels
        self.cuda = cuda
        self.testing_mode = testing_mode
        self.return_texts = return_texts  # 
        self._cache = {}
        
        print(f"Loading BERT model: {bert_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model)
        
        if cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.max_length = max_length
        self.model.eval()
        
        print(f"Dataset initialized: {len(self)} samples")
        print(f"Return format: {'(embedding, label, text)' if return_texts and labels is not None else '(embedding, label)' if labels is not None else 'embedding'}")
    
    def _get_bert_embedding(self, text):
        """Generate BERT embedding for a single text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        if self.cuda and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings.squeeze(0).cpu()  #  Move to CPU for caching
    
    def __getitem__(self, index: int):
        """
        Get embedding and (optional) label and text for index
        
        Returns:
            - If labels=None: embedding (Tensor)
            - If labels provided & return_texts=False: (embedding, label)
            - If labels provided & return_texts=True: (embedding, label, text)
        """
        #  Handle testing mode wrapping
        if self.testing_mode and index >= 128:
            index = index % 128
        
        # Check cache
        if index not in self._cache:
            text = self.texts[index]
            embedding = self._get_bert_embedding(text)
            
            if self.labels is not None:
                label = self.labels[index]
                
                #  Handle both int labels and one-hot/array labels
                if isinstance(label, (int, float)):
                    label_tensor = torch.tensor(label, dtype=torch.long)
                elif isinstance(label, (list, tuple)):
                    label_tensor = torch.tensor(label, dtype=torch.float)
                elif isinstance(label, torch.Tensor):
                    label_tensor = label.clone().detach()
                else:
                    label_tensor = torch.tensor(label, dtype=torch.long)
                
                #  Return with text for TF-IDF plots
                if self.return_texts:
                    self._cache[index] = (embedding, label_tensor, text)
                else:
                    self._cache[index] = (embedding, label_tensor)
            else:
                # No labels - just embedding
                self._cache[index] = embedding
        
        return self._cache[index]
    
    def __len__(self):
        """Return dataset length, limited in testing mode"""
        return min(128, len(self.texts)) if self.testing_mode else len(self.texts)
    
    def clear_cache(self):
        """Clear cached embeddings to free memory"""
        self._cache.clear()
        print("Cache cleared")
    
    def get_embedding_dim(self):
        """Get the dimensionality of BERT embeddings"""
        sample = self[0]
        if isinstance(sample, tuple):
            return sample[0].shape[0]
        else:
            return sample.shape[0]
