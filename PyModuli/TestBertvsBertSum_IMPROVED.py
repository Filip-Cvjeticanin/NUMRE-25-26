import os
import re
import tracemalloc
from time import perf_counter

import pandas as pd
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from rouge_score import rouge_scorer

tracemalloc.start()

# ==============================
# 1) Config
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"

MAX_LEN = 128
LR = 2e-5
EPOCHS = 3

# BERTSum-like params
NHEAD = 8
FF_DIM = 2048
DROPOUT = 0.1
NUM_LAYERS = 2

# CPU-friendly:
# True  -> train only heads (baseline + bertsum), BERT encoder frozen (much faster)
# False -> also fine-tune BERT (much slower on CPU)
FREEZE_BERT = True

# Dataset (2 columns: article, abstract)
DATASET_PATH = r"C:\Users\lovro\OneDrive\Dokumenti\Lovro\fer\Diplomski\1.godina\ZimskiSem\NeuMre\Projekt\NUMRE-25-26\Podatci\SkupPodataka.csv"

# Train/test split
TEST_RATIO = 0.2
SEED = 42

# Summary settings
MAX_SENTS_PER_DOC = 60
K_SUMMARY = 5

# Silver labels: how many sentences marked as 1 during training
MAX_K_SILVER = 5

# Batch training (documents per batch)
BATCH_DOCS = 8

# Output paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "Modeli")
SAVE_PATH = os.path.join(MODEL_DIR, "bert_vs_bertsum_trained.pt")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "rezultati.txt")

print("device:", DEVICE)
nltk.download("punkt")

# ROUGE scorer (library)
ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


# ==============================
# 2) Text utils
# ==============================
def clean_sentences(text: str):
    sents = [s.strip() for s in sent_tokenize(text or "") if s and len(s.strip()) > 3]
    sents = [s for s in sents if s not in {".", ",", ";", ":"}]
    return sents


def normalize_for_tfidf(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z0-9]+", text or ""))


def _norm_sent(s: str) -> str:
    # used only to forbid identical sentence repeats
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def topk_unique(sentences, scores: torch.Tensor, k: int):
    """
    Pick top-k by score, but forbid identical sentences from appearing twice.
    This only removes exact duplicates (after simple normalization).
    """
    n = len(sentences)
    if n == 0:
        return []

    k = min(k, n)

    # take more candidates, then filter duplicates
    cand_k = min(n, max(k * 3, k))
    cand_idx = torch.topk(scores, k=cand_k).indices.tolist()

    chosen = []
    seen = set()

    for i in cand_idx:
        key = _norm_sent(sentences[i])
        if key in seen:
            continue
        seen.add(key)
        chosen.append(i)
        if len(chosen) >= k:
            break

    # if still short, fill with remaining indices
    if len(chosen) < k:
        for i in range(n):
            if i not in chosen:
                chosen.append(i)
                if len(chosen) >= k:
                    break

    return chosen


# ==============================
# 3) Load dataset + split
# ==============================
def load_article_abstract_csv(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["article", "abstract"])
    df["article"] = df["article"].astype(str)
    df["abstract"] = df["abstract"].astype(str)
    return df


def split_train_test(df, test_ratio=0.2, seed=42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = int(len(df) * test_ratio)
    df_test = df.iloc[:n_test].copy()
    df_train = df.iloc[n_test:].copy()
    return df_train, df_test


# ==============================
# 4) Silver labels from abstract (TF-IDF)
# ==============================
def make_silver_labels(article: str, abstract: str, max_k: int):
    art_sents = clean_sentences(article)[:MAX_SENTS_PER_DOC]
    abs_sents = clean_sentences(abstract)

    if not art_sents:
        return [], []

    # dynamic k based on abstract sentence count, capped
    k = len(abs_sents)
    if k <= 0:
        k = 1
    k = min(max_k, max(1, k))
    k = min(k, len(art_sents))

    # faster: fit on article sentences, transform abstract
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    art_X = vec.fit_transform([normalize_for_tfidf(s) for s in art_sents])
    abs_X = vec.transform([normalize_for_tfidf(abstract)])

    sims = cosine_similarity(art_X, abs_X).reshape(-1)

    top_idx = sims.argsort()[::-1][:k]
    labels = [0] * len(art_sents)
    for idx in top_idx:
        labels[int(idx)] = 1

    return art_sents, labels


def build_training_items(df_train):
    items = []
    for _, row in tqdm(
        df_train.iterrows(), total=len(df_train), desc="building train items"
    ):
        article = row["article"]
        abstract = row["abstract"]
        sents, labels = make_silver_labels(article, abstract, MAX_K_SILVER)
        if not sents or not labels:
            continue
        items.append((sents, labels))
    return items


# ==============================
# 5) Models
# ==============================
class BertSentenceEncoder:
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(DEVICE)

        if FREEZE_BERT:
            for p in self.model.parameters():
                p.requires_grad = False

    def encode(self, sentences):
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # (n_sents, 768)

    def encode_batch(self, batch_sent_lists):
        """
        batch_sent_lists: list[list[str]]  (B docs, each list of sentences)
        returns:
          emb: (B, S, 768)
          sent_mask: (B, S) True where valid sentence, False where PAD
        """
        B = len(batch_sent_lists)
        lengths = [len(x) for x in batch_sent_lists]
        S = max(lengths) if lengths else 0
        if S == 0:
            return None, None

        flat = []
        for sents in batch_sent_lists:
            flat.extend(sents)

        inputs = self.tokenizer(
            flat,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)
        outputs = self.model(**inputs)
        flat_emb = outputs.last_hidden_state[:, 0, :]  # (sumS, 768)

        emb = torch.zeros(B, S, flat_emb.size(-1), device=DEVICE, dtype=flat_emb.dtype)
        sent_mask = torch.zeros(B, S, device=DEVICE, dtype=torch.bool)

        idx = 0
        for b in range(B):
            L = lengths[b]
            if L > 0:
                emb[b, :L, :] = flat_emb[idx : idx + L, :]
                sent_mask[b, :L] = True
                idx += L

        return emb, sent_mask


class SentenceScorer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, embeddings):
        # embeddings: (N, 768) OR (B*S, 768)
        return self.classifier(embeddings).squeeze(-1)


class BertSumScorer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=NHEAD,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=NUM_LAYERS)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, sent_embeddings, src_key_padding_mask=None):
        # sent_embeddings: (B, S, 768)
        x = self.encoder(sent_embeddings, src_key_padding_mask=src_key_padding_mask)
        return self.classifier(x).squeeze(-1)  # (B, S)


def save_checkpoint(encoder, baseline, bertsum, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "encoder": encoder.model.state_dict(),
            "baseline": baseline.state_dict(),
            "bertsum": bertsum.state_dict(),
        },
        path,
    )


# ==============================
# 6) Train both models (baseline + bertsum-like) with document batches
# ==============================
def train_two_models(df_train):
    encoder = BertSentenceEncoder()
    baseline = SentenceScorer().to(DEVICE)
    bertsum = BertSumScorer().to(DEVICE)

    params = []
    if not FREEZE_BERT:
        params += list(encoder.model.parameters())
    params += list(baseline.parameters()) + list(bertsum.parameters())

    optimizer = torch.optim.AdamW(params, lr=LR)

    # elementwise loss so we can mask PAD sentences
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    train_items = build_training_items(df_train)
    print("Training docs (usable):", len(train_items))

    for epoch in range(EPOCHS):
        encoder.model.train()
        baseline.train()
        bertsum.train()

        total_a = 0.0
        total_b = 0.0
        seen_docs = 0

        g = torch.Generator()
        g.manual_seed(SEED + epoch)
        perm = torch.randperm(len(train_items), generator=g).tolist()

        for start in tqdm(
            range(0, len(perm), BATCH_DOCS),
            desc=f"epoch {epoch+1}/{EPOCHS}",
        ):
            batch_idx = perm[start : start + BATCH_DOCS]
            batch_sents = [train_items[i][0] for i in batch_idx]
            batch_labels = [train_items[i][1] for i in batch_idx]

            emb, sent_mask = encoder.encode_batch(batch_sents)
            if emb is None:
                continue

            B, S, H = emb.shape

            y = torch.zeros(B, S, device=DEVICE, dtype=torch.float32)
            for b in range(B):
                L = len(batch_labels[b])
                y[b, :L] = torch.tensor(
                    batch_labels[b], device=DEVICE, dtype=torch.float32
                )

            optimizer.zero_grad(set_to_none=True)

            # baseline scores: apply to each sentence embedding
            scores_a = baseline(emb.view(B * S, H)).view(B, S)

            # bertsum-like scores: inter-sentence transformer
            # pass padding mask (True where PAD) to transformer
            pad_mask = ~sent_mask  # transformer expects True for PAD
            scores_b = bertsum(emb, src_key_padding_mask=pad_mask)

            # elementwise loss then mask out PAD
            loss_a_all = loss_fn(scores_a, y)
            loss_b_all = loss_fn(scores_b, y)

            mask_f = sent_mask.float()
            loss_a = (loss_a_all * mask_f).sum() / (mask_f.sum() + 1e-8)
            loss_b = (loss_b_all * mask_f).sum() / (mask_f.sum() + 1e-8)

            loss = loss_a + loss_b
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            # accumulate weighted by batch size (docs)
            total_a += float(loss_a.item()) * B
            total_b += float(loss_b.item()) * B
            seen_docs += B

        current, peak = tracemalloc.get_traced_memory()
        avg_a = total_a / max(1, seen_docs)
        avg_b = total_b / max(1, seen_docs)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"avg_loss_baseline={avg_a:.4f} avg_loss_bertsum={avg_b:.4f} | "
            f"docs={seen_docs} | mem={current/1024/1024:.2f}MB peak={peak/1024/1024:.2f}MB"
        )

    save_checkpoint(encoder, baseline, bertsum, SAVE_PATH)
    print("Saved checkpoint to:", SAVE_PATH)
    return encoder, baseline, bertsum


# ==============================
# 7) Summarization (no exact duplicates)
# ==============================
@torch.no_grad()
def summarize_baseline(encoder, baseline, article: str, k: int):
    encoder.model.eval()
    baseline.eval()

    sents = clean_sentences(article)[:MAX_SENTS_PER_DOC]
    if not sents:
        return ""

    emb = encoder.encode(sents)
    scores = baseline(emb)

    idx = topk_unique(sents, scores, k)
    picked = [sents[i] for i in sorted(idx)]
    return " ".join(picked)


@torch.no_grad()
def summarize_bertsum_like(encoder, bertsum, article: str, k: int):
    encoder.model.eval()
    bertsum.eval()

    sents = clean_sentences(article)[:MAX_SENTS_PER_DOC]
    if not sents:
        return ""

    emb = encoder.encode(sents).unsqueeze(0)  # (1, S, 768)
    scores = bertsum(emb).squeeze(0)  # (S,)

    idx = topk_unique(sents, scores, k)
    picked = [sents[i] for i in sorted(idx)]
    return " ".join(picked)


# ==============================
# 8) Evaluate on test (ROUGE via library) + save summaries for comparison
# ==============================
def rouge_all(pred: str, ref: str):
    scores = ROUGE.score(ref, pred)  # rouge-score expects (target, prediction)
    return {
        "rouge1_f1": float(scores["rouge1"].fmeasure),
        "rouge2_f1": float(scores["rouge2"].fmeasure),
        "rougeL_f1": float(scores["rougeL"].fmeasure),
    }


@torch.no_grad()
def evaluate_on_test(df_test, encoder, baseline, bertsum, k: int):
    r_base = []
    r_bsum = []

    used = 0
    skipped = 0

    sum_words_base = 0
    sum_words_bsum = 0
    sum_words_abs = 0

    t_eval_1 = perf_counter()

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("BERT baseline (trained) vs BERTSum-like (trained) - TEST SET\n")
        f.write("Dataset: article + abstract\n")
        f.write("Training: silver labels (TF-IDF) from abstract\n")
        f.write("Metric: ROUGE-1/2/L F1 (vs abstract)\n")
        f.write(f"Summary length: top-k sentences, k={k}\n")
        f.write(f"Max sentences per doc: {MAX_SENTS_PER_DOC}\n")
        f.write(f"FREEZE_BERT={FREEZE_BERT}\n")
        f.write(f"BATCH_DOCS={BATCH_DOCS}\n\n")

        for i, row in tqdm(
            df_test.reset_index(drop=True).iterrows(),
            total=len(df_test),
            desc="evaluating",
        ):
            article = row["article"]
            abstract = row["abstract"]

            if not article or not abstract:
                skipped += 1
                continue

            if len(clean_sentences(article)) == 0:
                skipped += 1
                continue

            used += 1

            pred_a = summarize_baseline(encoder, baseline, article, k)
            pred_b = summarize_bertsum_like(encoder, bertsum, article, k)

            ra = rouge_all(pred_a, abstract)
            rb = rouge_all(pred_b, abstract)

            r_base.append(ra)
            r_bsum.append(rb)

            sum_words_base += word_count(pred_a)
            sum_words_bsum += word_count(pred_b)
            sum_words_abs += word_count(abstract)

            f.write(f"DOC {i+1}\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"BASE    ROUGE1={ra['rouge1_f1']:.3f} ROUGE2={ra['rouge2_f1']:.3f} ROUGEL={ra['rougeL_f1']:.3f}\n"
            )
            f.write(
                f"BSUM    ROUGE1={rb['rouge1_f1']:.3f} ROUGE2={rb['rouge2_f1']:.3f} ROUGEL={rb['rougeL_f1']:.3f}\n"
            )
            f.write(
                f"Words (BASE/BSUM/ABS): {word_count(pred_a)}/{word_count(pred_b)}/{word_count(abstract)}\n\n"
            )

            # Save reference and model summaries for direct comparison
            f.write("GOLD ABSTRACT:\n")
            f.write(abstract.strip() + "\n\n")
            f.write("BASE SUMMARY:\n")
            f.write(pred_a.strip() + "\n\n")
            f.write("BSUM SUMMARY:\n")
            f.write(pred_b.strip() + "\n\n\n")

        def avg(xs, key):
            if not xs:
                return 0.0
            return sum(d[key] for d in xs) / len(xs)

        a_r1 = avg(r_base, "rouge1_f1")
        a_r2 = avg(r_base, "rouge2_f1")
        a_rl = avg(r_base, "rougeL_f1")

        b_r1 = avg(r_bsum, "rouge1_f1")
        b_r2 = avg(r_bsum, "rouge2_f1")
        b_rl = avg(r_bsum, "rougeL_f1")

        denom = used if used > 0 else 1
        avg_w_a = sum_words_base / denom
        avg_w_b = sum_words_bsum / denom
        avg_w_abs = sum_words_abs / denom

        t_eval_2 = perf_counter()
        eval_total = t_eval_2 - t_eval_1
        eval_per_doc = eval_total / denom

        f.write("\n" + "=" * 80 + "\n")
        f.write("AVERAGES (TEST)\n")
        f.write("=" * 80 + "\n")
        f.write(f"BASE    ROUGE1={a_r1:.3f} ROUGE2={a_r2:.3f} ROUGEL={a_rl:.3f}\n")
        f.write(f"BSUM    ROUGE1={b_r1:.3f} ROUGE2={b_r2:.3f} ROUGEL={b_rl:.3f}\n\n")

        f.write("EXTRA METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Used docs: {used}\n")
        f.write(f"Skipped docs: {skipped}\n")
        f.write(f"Avg words (BASE): {avg_w_a:.2f}\n")
        f.write(f"Avg words (BSUM): {avg_w_b:.2f}\n")
        f.write(f"Avg words (abstract): {avg_w_abs:.2f}\n")
        f.write(f"Total eval time (s): {eval_total:.2f}\n")
        f.write(f"Avg eval time per doc (s): {eval_per_doc:.4f}\n")

    print("Saved results to:", RESULTS_PATH)


# ==============================
# 9) Main
# ==============================
if __name__ == "__main__":
    t1 = perf_counter()

    df = load_article_abstract_csv(DATASET_PATH)[0:200]
    df_train, df_test = split_train_test(df, test_ratio=TEST_RATIO, seed=SEED)

    print("Train size:", len(df_train), "| Test size:", len(df_test))
    print(
        f"Using MAX_SENTS_PER_DOC={MAX_SENTS_PER_DOC}, K_SUMMARY={K_SUMMARY}, "
        f"FREEZE_BERT={FREEZE_BERT}, BATCH_DOCS={BATCH_DOCS}"
    )

    encoder, baseline, bertsum = train_two_models(df_train)
    evaluate_on_test(df_test, encoder, baseline, bertsum, k=K_SUMMARY)

    t2 = perf_counter()
    print(f"Ukupno trajanje programa: {t2 - t1:.2f} sekundi")
