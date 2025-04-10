import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

class CachedBERTDataset(Dataset):
    def __init__(self, texts, labels=None, bert_model="bert-base-uncased", max_length=128, cuda=True, testing_mode=False):
        """
        Dataset that caches BERT embeddings for text data
        
        Args:
            texts: List of text strings to encode
            labels: Optional list of labels corresponding to the texts
            bert_model: Pre-trained BERT model name to use
            max_length: Maximum sequence length for BERT tokenizer
            cuda: Whether to use GPU acceleration
            testing_mode: If True, only use a small subset of data
        """
        self.texts = texts
        self.labels = labels  # Can be None
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = {}
        print(f"Loading BERT model: {bert_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model)
        if cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.max_length = max_length
        self.model.eval()
    
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
            embeddings = outputs.last_hidden_state[:, 0, :]  
        return embeddings.squeeze(0)
    
    def __getitem__(self, index: int):
        """Get embedding and (optional) label for index"""
        if self.testing_mode and index >= 128:
            index = index % 128
    
        if index not in self._cache:
            text = self.texts[index]
            embedding = self._get_bert_embedding(text)
    
            if self.labels is not None:  
                label = self.labels[index]
                label_tensor = torch.tensor(label, dtype=torch.long if isinstance(label, int) else torch.float)
    
                if self.cuda and torch.cuda.is_available():
                    label_tensor = label_tensor.cuda(non_blocking=True)
                
                self._cache[index] = (embedding, label_tensor)
            else:  
                self._cache[index] = embedding  
    
        return self._cache[index]
    
    def __len__(self):
        """Return dataset length, limited in testing mode"""
        return min(128, len(self.texts)) if self.testing_mode else len(self.texts)