import os

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import nltk
from nltk.tokenize import sent_tokenize
from PyModuli.example_extractor import extract_examples
import tracemalloc

tracemalloc.start()

# # ==============================
# 1. Configuration
# # ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
LR = 2e-5
EPOCHS = 20
torch.set_num_threads(10)
torch.set_num_interop_threads(1)

print("device:", DEVICE)

nltk.download("punkt")
nltk.download("punkt_tab")



# ==============================
# 2. Neural network for scoring
# ==============================
class SentenceScorer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()


        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300,1)
        )




    def forward(self, embeddings):
        scores = self.classifier(embeddings)
        return scores.squeeze(-1)


    def save(self, filepath):
        """

        \n\n\n
        DEPRECATED!
        \n\n\n

        Save the SentenceScorer model to a file.
        :param filepath:
        :return:
        """
        torch.save(self.state_dict(), filepath)
        print(f"SentenceScorer saved to {filepath}")


    @classmethod
    def load(cls, filepath, device=DEVICE):
        """

        \n\n\n
        DEPRECATED!
        \n\n\n

        Load a SentenceScorer model from a file.
        :param filepath:
        :param device:
        :return:
        """
        scorer = cls().to(device)
        scorer.load_state_dict(torch.load(filepath, map_location=device))
        scorer.eval()  # set to evaluation mode
        print(f"SentenceScorer loaded from {filepath}")
        return scorer



# # ==============================
# 3. BERT Encoder
# # ==============================
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

# # ==============================
# 4. Label Generation
# # ==============================
def generate_labels(sentences, summary_sentences):
    labels = [1.0 if s in summary_sentences else 0.0 for s in sentences]
    return torch.tensor(labels, dtype=torch.float32).to(DEVICE)

# # ==============================
# 5. Training Function
# # ==============================
def train_model(encoder, scorer, sentences, labels, save_path = None, epoch_num = EPOCHS, epoch_idx = 0, max_epochs = 0):
    encoder.model.train()
    scorer.train()

    optimizer = torch.optim.Adam(
        list(encoder.model.parameters()) + list(scorer.parameters()),
        lr=LR
    )
    loss_fn = nn.BCEWithLogitsLoss()

    print("Starting training...")
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current: {current / 1024:.2f} KB")
    print(f"Peak: {peak / 1024:.2f} KB")

    for epoch in range(epoch_num):
        optimizer.zero_grad()
        embeddings = encoder.encode(sentences)
        scores = scorer(embeddings)
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()
        print(f"\nEpoch {epoch + 1 + epoch_idx}/{max_epochs} - Loss: {loss.item():.4f}")
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current / 1024:.2f} KB")
        print(f"Peak: {peak / 1024:.2f} KB")

    if save_path is not None:
        save_models(encoder, scorer, save_path)

# -----------------------------
# 6. Summarization (Inference)
# -----------------------------
def summarize(encoder, scorer, text, num_sentences):
    encoder.model.eval()
    scorer.eval()

    sentences = [
        s.strip()
        for s in sent_tokenize(text)
        if len(s.strip()) > 10 and s.strip() not in {".", ","}
    ]

    if not sentences:
        return ""

    with torch.no_grad():
        embeddings = encoder.encode(sentences)
        scores = scorer(embeddings)

        k = min(num_sentences, len(sentences))
        top_indices = torch.topk(scores, k=k).indices
        summary = [sentences[i] for i in sorted(top_indices.tolist())]

    return " ".join(summary)

def summarize_old(encoder, scorer, text, num_sentences):
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



def save_models(encoder, scorer, path):
    """
    Save encoder and scorer to path.
    :param encoder:
    :param scorer:
    :param path:
    :return:
    """
    torch.save(
        {
            "encoder": encoder.model.state_dict(),
            "scorer": scorer.state_dict(),
        },
        path,
    )
    print(f"Models saved to {path}")



def load_models(path, device=DEVICE):
    """
    Load encoder and scorer from path.
    :param path:
    :param device:
    :return:
    """
    checkpoint = torch.load(path, map_location=device)

    encoder = BertSentenceEncoder()
    scorer = SentenceScorer().to(device)

    encoder.model.load_state_dict(checkpoint["encoder"])
    scorer.load_state_dict(checkpoint["scorer"])

    encoder.model.eval()
    scorer.eval()

    print(f"Models loaded from {path}")
    return encoder, scorer




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

def train_N_single_example(N=1, test_examples_num = 1, dataset_filepath=None, save_path=None, summary_sentences_num = 5, epoch_num = EPOCHS):
    # Create iterator
    examples_iter = iter(extract_examples(dataset_filepath))

    encoder = BertSentenceEncoder()
    scorer = SentenceScorer().to(DEVICE)

    # EPOCH LOOP
    for epoch in range(epoch_num):


        # Get train examples
        for i in range(N):
            try:
                example = next(examples_iter)
            except StopIteration:
                break

            # Get one example
            print("\nLoading example", i)
            # Data from example
            article = example.get("article")
            labels = example.get("labels")
            print(article[:100], "...")
            print(labels)
            # Compute sentence tokens
            # Normalize sentences: remove leading/trailing spaces
            sentences = [s.strip() for s in sent_tokenize(article)]

            current, peak = tracemalloc.get_traced_memory()
            print(f"Current: {current / 1024:.2f} KB")
            print(f"Peak: {peak / 1024:.2f} KB")

            # Train one example
            train_labels_tensor = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
            print(train_labels_tensor)
            train_model(encoder, scorer, sentences, train_labels_tensor, save_path=save_path, epoch_num=1, epoch_idx=epoch, max_epochs=epoch_num)

    # Get test examples (resume iterator)
    test_examples = []
    for i in range(test_examples_num):
        try:
            example = next(examples_iter)
        except StopIteration:
            break

        print("\nLoading test example", i)
        # Data from example
        article = example.get("article")
        print(article[:100], "...")
        # Compute sentence tokens
        # Append
        test_examples.append(example)

    # Test model on test examples.
    for i, e in enumerate(test_examples):
        summary = summarize(encoder, scorer, e["article"], summary_sentences_num)
        #summary_old = summarize_old(encoder, scorer, e["article"], summary_sentences_num)

        print("\n==========================")
        print(f"TEST {i + 1}")
        print("==========================\n")
        print("ARTICLE:")
        print(e["article"])
        print("SUMMARY:")
        print(summary)

        #print("SUMMARY_OLD:")
        #print(summary_old)

        #print("SENTENCES")
        #sentences = e["sentences"]
        #for j, sent in enumerate(sentences):
            #print(f"sent {j}: {sent}")


def train_N_all_examples(N=1, test_examples_num = 1, dataset_filepath=None, save_path=None, summary_sentences_num = 5, epoch_num = EPOCHS, example_by_example=True):
    # Create iterator
    examples_iter = iter(extract_examples(dataset_filepath))

    # Get train examples
    train_sentences = []
    train_labels = []
    for i in range(N):
        try:
            example = next(examples_iter)
        except StopIteration:
            break

        print("\nLoading example", i)
        # Data from example
        article = example.get("article")
        labels = example.get("labels")
        print(article[:100], "...")
        print(labels)
        # Compute sentence tokens
        # Normalize sentences: remove leading/trailing spaces
        sentences = [s.strip() for s in sent_tokenize(article)]
        # Add
        train_sentences = train_sentences + sentences
        train_labels = train_labels + labels
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current / 1024:.2f} KB")
        print(f"Peak: {peak / 1024:.2f} KB")

    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).to(DEVICE)
    print(train_labels_tensor)

    # Get test examples (resume iterator)
    test_examples = []
    for i in range(test_examples_num):
        try:
            example = next(examples_iter)
        except StopIteration:
            break

        print("\nLoading test example", i)
        # Data from example
        article = example.get("article")
        print(article[:100], "...")
        # Compute sentence tokens
        # Append
        test_examples.append(example)

    encoder = BertSentenceEncoder()
    scorer = SentenceScorer().to(DEVICE)

    train_model(encoder, scorer, train_sentences, train_labels_tensor, save_path=save_path, epoch_num=epoch_num)

    for i, e in enumerate(test_examples):
        summary = summarize(encoder, scorer, e["article"], summary_sentences_num)
        #summary_old = summarize_old(encoder, scorer, e["article"], summary_sentences_num)

        print("\n==========================")
        print(f"TEST {i + 1}")
        print("==========================\n")
        print("ARTICLE:")
        print(e["article"])
        print("SUMMARY:")
        print(summary)

        #print("SUMMARY_OLD:")
        #print(summary_old)

        #print("SENTENCES")
        #sentences = e["sentences"]
        #for j, sent in enumerate(sentences):
            #print(f"sent {j}: {sent}")




def train_on_N_examples(N=1, test_examples_num = 1, dataset_filepath=None, save_path=None, summary_sentences_num = 5, epoch_num = EPOCHS, example_by_example=True):

    if example_by_example:
        train_N_single_example(N, test_examples_num, dataset_filepath, save_path, summary_sentences_num, epoch_num)
    else:
        train_N_all_examples(N, test_examples_num, dataset_filepath, save_path, summary_sentences_num, epoch_num)




def load_and_test(load_path, dataset_filepath, test_example_start_idx = 0, test_example_num = 1, summary_sentences_num = 5):
    # Initialize encoder and scorer
    encoder, scorer = load_models(load_path)

    # Set iterator
    examples_iter = iter(extract_examples(dataset_filepath, test_example_start_idx))

    # Get test examples (resume iterator)
    test_examples = []
    for i in range(test_example_num):
        try:
            example = next(examples_iter)
        except StopIteration:
            break

        print("\nLoading test example", i)
        # Data from example
        article = example.get("article")
        print(article[:100], "...")
        # Compute sentence tokens
        # Append
        test_examples.append(example)

    # Evaluate test examples
    for i, e in enumerate(test_examples):
        summary = summarize(encoder, scorer, e["article"], summary_sentences_num)
        #summary_old = summarize_old(encoder, scorer, e["article"], summary_sentences_num)

        print("\n==========================")
        print(f"TEST {i + 1}")
        print("==========================\n")
        print("ARTICLE:")
        print(e["article"])
        print("SUMMARY:")
        print(summary)


if __name__ == "__main__":
    #experimental_train_and_test()
    train_on_N_examples(10, 2,"../Podatci/SkupPodataka.csv", save_path = "../Modeli/model-test.pt",
                        epoch_num=100, example_by_example=True)
    #load_and_test(load_path="../Modeli/model-test.pt",
    #              dataset_filepath= r"C:\Docs\01 - Filip\02 - FER\070 - G4S1\03-NM\Projekt\SkupPodataka-last.csv",
    #              test_example_start_idx= 3,
    #              test_example_num= 2,
    #              summary_sentences_num= 5)