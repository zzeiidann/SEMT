# FNN_Beta: Sentiment and Topic Modeling

This project implements a **Fully Neural Network (FNN)** combined with **BERT embeddings** for sentiment analysis and topic clustering.

## Model Architecture

![FNN_Beta Architecture](architecture.jpg)


## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/zzeiidann/FNN_Beta.git
   ```

2. Import required modules:
   ```python
   from FNN_Beta.model import FNN
   from FNN_Beta.dataset import CachedBERTDataset
   from keras.optimizers import SGD, Adam
   ```

3. Prepare the data (example):
   ```python
   texts = ["Makanan enak...", "Saya kecewa...", ...]
   labels = [1, 0, ...]
   ```

4. Build dataset with IndoBERT embeddings:
   ```python
   dataset = CachedBERTDataset(
       texts=texts,
       labels=labels,
       bert_model="indolem/indobert-base-uncased",
       max_length=128,
       cuda=True,
       testing_mode=False
   )
   ```

5. Initialize the model:
   ```python
   model = FNN(
       dims=[768, 2022, 2023, 32],
       n_clusters=3,
       batch_size=12
   )
   ```

6. Pre-train autoencoder:
   ```python
   encoder_weight = model.pretrain_autoencoder(dataset, epochs=100, optimizer="sgd")
   ```

7. Initialize the model with pretrained autoencoder weights:
   ```python
   model.initialize_model(
       ae_weights="/content/pretrained_ae.weights.h5",
       gamma=0.5,
       eta=0.5,
       optimizer=SGD(learning_rate=0.001, momentum=0.9)
   )
   ```

8. Train the model:
   ```python
   model.train(
       dataset=dataset,
       tol=1e-4,
       update_interval=70,
       maxiter=1000,
   )
   ```

9. Run prediction on a sample input:
   ```python
   model.predict("Kamu jelek", bert_model="indolem/indobert-base-uncased")
   ```

## Notes
- The model uses **IndoBERT** (`indolem/indobert-base-uncased`).
- Ensure the pretrained autoencoder weights file (`pretrained_ae.weights.h5`) is available at the correct path.
- Parameters `gamma` and `eta` can be adjusted to balance topic and sentiment loss functions.
