import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import nltk
from nltk.tokenize import sent_tokenize

# -----------------------------
# 1. Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
LR = 2e-5
EPOCHS = 20

print("device:", DEVICE)

nltk.download("punkt")
nltk.download("punkt_tab")

# -----------------------------
# 2. Neural Sentence Scorer
# -----------------------------
class SentenceScorer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, embeddings):
        scores = self.classifier(embeddings)
        return scores.squeeze(-1)

# -----------------------------
# 3. BERT Encoder
# -----------------------------
class BertSentenceEncoder:
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(DEVICE)

    def encode(self, sentences):
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = self.model(**inputs)
        # Use [CLS] token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings

# -----------------------------
# 4. Label Generation
# -----------------------------
def generate_labels(sentences, summary_sentences):
    labels = [1.0 if s in summary_sentences else 0.0 for s in sentences]
    return torch.tensor(labels, dtype=torch.float32).to(DEVICE)

# -----------------------------
# 5. Training Function
# -----------------------------
def train_model(encoder, scorer, sentences, labels):
    encoder.model.train()
    scorer.train()

    optimizer = torch.optim.Adam(
        list(encoder.model.parameters()) + list(scorer.parameters()),
        lr=LR
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        embeddings = encoder.encode(sentences)
        scores = scorer(embeddings)
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")

# -----------------------------
# 6. Summarization (Inference)
# -----------------------------
def summarize(encoder, scorer, text, num_sentences):
    encoder.model.eval()
    scorer.eval()

    # Normalize sentences: remove leading/trailing spaces
    sentences = [s.strip() for s in sent_tokenize(text)]

    with torch.no_grad():
        embeddings = encoder.encode(sentences)
        scores = scorer(embeddings)
        top_indices = torch.topk(scores, k=num_sentences).indices
        summary = [sentences[i] for i in sorted(top_indices.tolist())]

    # Join sentences cleanly
    return " ".join(summary)

# -----------------------------
# 7. Example Usage
# -----------------------------
def experimental_train_and_test():
    # Example document + reference summary
    document = """
    Neural networks have been widely adopted in natural language processing.
    BERT introduced bidirectional transformers for language understanding.
    Extractive summarization selects important sentences from text.
    This approach avoids generating new content.
    Neural sentence scoring allows trainable summarization models.
    """
    document2 = """
    This is another document.
    This document is a test to see how a model trained on one model would behave on others.
    This document is only 4 sentences long.
    Let's see how it does.
    """
    reference_summary = [
        "BERT introduced bidirectional transformers for language understanding.",
        "Extractive summarization selects important sentences from text."
    ]
    sentences = sent_tokenize(document)
    # Initialize model
    encoder = BertSentenceEncoder()
    scorer = SentenceScorer().to(DEVICE)
    # Generate training labels
    labels = generate_labels(sentences, reference_summary)
    # Train
    train_model(encoder, scorer, sentences, labels)

    # Inference
    n = 3
    summary = summarize(encoder, scorer, document, num_sentences=n)
    print(f"\nGenerated Summary ({n} sentences):\n", summary)

    # Inference on test document
    n = 2
    summary = summarize(encoder, scorer, document2, num_sentences=n)
    print(f"\nSummary on test document ({n} sentences):\n", summary)



if __name__ == "__main__":

    experimental_train_and_test()