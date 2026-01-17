# PyModuli/SimpleTestV2.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertModel

from PyModuli.example_extractor import extract_examples
from PyModuli.eval_metrics import evaluate_summary, aggregate_metrics

nltk.download("punkt", quiet=True)

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
LR = 2e-5
EPOCHS = 20

# =========================================================
# MODELS
# =========================================================
class SentenceScorer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 1)
        )

    def forward(self, x):
        return self.classifier(x).squeeze(-1)


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
        return outputs.last_hidden_state[:, 0, :]  # CLS


# =========================================================
# CORE
# =========================================================
def summarize(encoder, scorer, text, k):
    encoder.model.eval()
    scorer.eval()

    sentences = [
        s.strip()
        for s in sent_tokenize(text or "")
        if len(s.strip()) > 10 and s.strip() not in {".", ","}
    ]

    if not sentences:
        return ""

    with torch.no_grad():
        emb = encoder.encode(sentences)
        scores = scorer(emb)
        k = min(k, len(sentences))
        idx = torch.topk(scores, k=k).indices.tolist()
        idx.sort()

    return " ".join(sentences[i] for i in idx)


def save_models(encoder, scorer, path):
    torch.save(
        {
            "encoder": encoder.model.state_dict(),
            "scorer": scorer.state_dict(),
        },
        path,
    )


def load_models(path):
    ckpt = torch.load(path, map_location=DEVICE)

    encoder = BertSentenceEncoder()
    scorer = SentenceScorer().to(DEVICE)

    encoder.model.load_state_dict(ckpt["encoder"])
    scorer.load_state_dict(ckpt["scorer"])

    encoder.model.eval()
    scorer.eval()
    return encoder, scorer


# =========================================================
# TRAINING API (koristi main.py)
# =========================================================
def train_on_N_examples(
    N=1,
    test_examples_num=0,
    dataset_filepath=None,
    save_path=None,
    epoch_num=EPOCHS,
    example_by_example=True,
):
    examples = list(extract_examples(dataset_filepath))[:N]

    encoder = BertSentenceEncoder()
    scorer = SentenceScorer().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.model.parameters()) + list(scorer.parameters()),
        lr=LR
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(epoch_num):
        for ex in examples:
            sentences = ex["sentences"] or sent_tokenize(ex["article"])
            labels = ex["labels"]

            m = min(len(sentences), len(labels))
            sentences = sentences[:m]
            labels = labels[:m]

            y = torch.tensor(labels, dtype=torch.float32, device=DEVICE)

            encoder.model.train()
            scorer.train()

            optimizer.zero_grad()
            emb = encoder.encode(sentences)
            scores = scorer(emb)
            loss = loss_fn(scores, y)
            loss.backward()
            optimizer.step()

        print(f"[TRAIN] epoch {ep+1}/{epoch_num} | loss={loss.item():.4f}")

        if save_path:
            save_models(encoder, scorer, save_path)


# =========================================================
# TEST API (koristi main.py)
# =========================================================
def load_and_test(
    load_path,
    dataset_filepath,
    test_example_start_idx=0,
    test_example_num=1,
    summary_sentences_num=6,
):
    encoder, scorer = load_models(load_path)
    examples = iter(extract_examples(dataset_filepath, start_index=test_example_start_idx))

    for i in range(test_example_num):
        ex = next(examples)
        summary = summarize(
            encoder,
            scorer,
            ex["article"],
            summary_sentences_num,
        )

        print("\n==========================")
        print(f"TEST {i+1}")
        print("==========================")
        print("ARTICLE:")
        print(ex["article"])
        print("SUMMARY:")
        print(summary)


# =========================================================
# EVAL API (koristi main.py)
# =========================================================
def evaluate_model_on_N_examples(
    load_path,
    dataset_filepath,
    start_idx=0,
    num_examples=100,
    min_fragment_len=3,
    novel_ngram_n=2,
    redundancy_ngram_n=1,
    match_abstract=True,
    fixed_ratio=0.15,
):
    encoder, scorer = load_models(load_path)
    examples = iter(extract_examples(dataset_filepath, start_index=start_idx))

    metrics = []

    for i in range(num_examples):
        ex = next(examples)

        article = ex["article"]
        abstract = ex.get("abstract")

        k = len(sent_tokenize(abstract)) if (match_abstract and abstract) else max(
            1, round(fixed_ratio * len(sent_tokenize(article)))
        )

        gen = summarize(encoder, scorer, article, k)

        m = evaluate_summary(
            article=article,
            generated_summary=gen,
            reference_summary=abstract,
            novel_ngram_n=novel_ngram_n,
            redundancy_ngram_n=redundancy_ngram_n,
            min_fragment_len=min_fragment_len,
        )

        metrics.append(m)

        print(
            f"[EVAL] {i+1}/{num_examples} | "
            f"R1F1={m.rouge1_f1:.3f} "
            f"D={m.density:.3f} "
            f"RR={m.redundancy_rate:.3f}"
        )

    avg = aggregate_metrics(metrics)

    print("\n=== AVERAGE METRICS ===")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

    return avg, metrics
