# PyModuli/SimpleTest.py
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertModel

from tqdm import tqdm

from PyModuli.example_extractor import extract_examples
from PyModuli.eval_metrics import evaluate_summary, aggregate_metrics

nltk.download("punkt", quiet=True)

# --------------------------------
# konfiguracija
# --------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
LR = 2e-5
EPOCHS = 20


# --------------------------------
# pomocne funkcije za zabranu duplikata recenica
# --------------------------------
def _norm_sentence(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)  # vise razmaka -> jedan razmak
    s = re.sub(r"[^\w\s]", "", s)  # izbaci interpunkciju
    return s


def topk_unique(sentences, scores: torch.Tensor, k: int):
    # vraca indekse top-k recenica bez ponavljanja istih recenica
    n = len(sentences)
    if n == 0:
        return []

    # k ne smije biti veci od broja recenica
    k = min(k, n)

    # uzmi vise kandidata pa filtriraj duplikate
    cand_k = min(n, max(k * 2, k))
    cand_idx = torch.topk(scores, k=cand_k).indices.tolist()

    chosen = []
    seen = set()

    # prvo prolazimo po najboljim kandidatima
    for i in cand_idx:
        key = _norm_sentence(sentences[i])
        if key and key not in seen:
            seen.add(key)
            chosen.append(i)
            if len(chosen) == k:
                return chosen

    # ako nema dovoljno razlicitih, dopuni preostalim recenicama
    for i in range(n):
        key = _norm_sentence(sentences[i])
        if key and key not in seen:
            seen.add(key)
            chosen.append(i)
            if len(chosen) == k:
                break

    return chosen


# --------------------------------
# modeli
# --------------------------------
class SentenceScorer(nn.Module):
    # mreza koja na temelju embeddinga recenice daje logit
    def __init__(self, hidden_size=768):
        super().__init__()

        # vise linearnih slojeva za ucenje nelinearnih odnosa
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
            nn.Linear(300, 1),
        )

    def forward(self, x):
        # x je (broj_recenica, 768), izlaz je (broj_recenica,)
        return self.classifier(x).squeeze(-1)


class BertSentenceEncoder:
    # bert tokenizira recenice i vraca cls embedding po recenici
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(DEVICE)

    def encode(self, sentences):
        # pretvori listu recenica u tenzore tokena
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)

        # bert prolaz naprijed
        outputs = self.model(**inputs)

        # uzmi cls embedding kao reprezentaciju recenice
        return outputs.last_hidden_state[:, 0, :]


# --------------------------------
# sumarizacija
# --------------------------------
def summarize(encoder, scorer, text, k):
    # prebaci modele u eval nacin (bez dropouta, bez gradijenata)
    encoder.model.eval()
    scorer.eval()

    # razbij tekst na recenice i ukloni prekratke / prazne
    sentences = [
        s.strip()
        for s in sent_tokenize(text or "")
        if len(s.strip()) > 10 and s.strip() not in {".", ","}
    ]

    # ako nema recenica, nema ni sazetka
    if not sentences:
        return ""

    with torch.no_grad():
        # dobavi embedding svake recenice
        emb = encoder.encode(sentences)

        # izracunaj logite za svaku recenicu
        scores = scorer(emb)

        # k ne smije biti veci od broja recenica
        k = min(k, len(sentences))

        # odaberi top-k recenica bez identicnih ponavljanja
        idx = topk_unique(sentences, scores, k)

        # sortiraj indekse da se recenice pojave redom kao u originalu
        idx.sort()

    # spoji izabrane recenice u jedan tekst sazetka
    return " ".join(sentences[i] for i in idx)


# --------------------------------
# spremanje i ucitavanje modela
# --------------------------------
def save_models(encoder, scorer, path):
    # spremi state_dict od bert encodera i scorera u jednu datoteku
    torch.save(
        {
            "encoder": encoder.model.state_dict(),
            "scorer": scorer.state_dict(),
        },
        path,
    )


def load_models(path):
    # ucitaj checkpoint (na cpu ili gpu, ovisno o DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)

    # inicijaliziraj nove instance modela
    encoder = BertSentenceEncoder()
    scorer = SentenceScorer().to(DEVICE)

    # ucitaj tezine u modele
    encoder.model.load_state_dict(ckpt["encoder"])
    scorer.load_state_dict(ckpt["scorer"])

    # prebaci u eval nacin (za test/eval)
    encoder.model.eval()
    scorer.eval()
    return encoder, scorer


# --------------------------------
# treniranje (koristi main.py ili direktno)
# --------------------------------
def train_on_N_examples(
    N=1,
    dataset_filepath=None,
    save_path=None,
    epoch_num=EPOCHS,
):
    # ucitaj prvih N primjera iz csv-a preko extract_examples
    examples = list(extract_examples(dataset_filepath))[:N]

    # inicijaliziraj encoder i scorer
    encoder = BertSentenceEncoder()
    scorer = SentenceScorer().to(DEVICE)

    # optimizer optimizira i bert i scorer parametre
    optimizer = torch.optim.Adam(
        list(encoder.model.parameters()) + list(scorer.parameters()), lr=LR
    )

    # binarna klasifikacija recenica (0/1) preko logita
    loss_fn = nn.BCEWithLogitsLoss()

    # epohe treniranja
    for ep in range(epoch_num):

        last_loss = None

        # tqdm progress bar kroz primjere u jednoj epohi
        for ex in tqdm(examples, desc=f"train epoch {ep+1}/{epoch_num}", leave=False):

            # uzmi recenice i labele (ako recenice ne postoje, tokeniziraj iz article)
            sentences = ex["sentences"] or sent_tokenize(ex["article"])
            labels = ex["labels"]

            # poravnaj duljine
            m = min(len(sentences), len(labels))
            if m == 0:
                continue
            sentences = sentences[:m]
            labels = labels[:m]

            y = torch.tensor(labels, dtype=torch.float32, device=DEVICE)

            encoder.model.train()
            scorer.train()

            # ponisti prethodne gradijente
            optimizer.zero_grad()

            # izracunaj bert embedding za svaku recenicu
            emb = encoder.encode(sentences)

            # dobijemo logite za svaku recenicu
            scores = scorer(emb)

            # izracun loss-a za klasifikaciju 0/1 po recenici
            loss = loss_fn(scores, y)

            # propagiranje pogreske unatrag
            loss.backward()

            # azuriranje parametara modela pomocu optimizatora
            optimizer.step()

            last_loss = float(loss.item())

        # ispis po epohi (bez greske ako nije bilo valjanih primjera)
        if last_loss is None:
            print(f"Train epoch {ep+1}/{epoch_num} | no valid training examples")
        else:
            print(f"Train epoch {ep+1}/{epoch_num} | loss={last_loss:.4f}")

        # spremi checkpoint nakon svake epohe (ako je zadan put)
        if save_path:
            save_models(encoder, scorer, save_path)


# --------------------------------
# test
# --------------------------------
def load_and_test(
    load_path,
    dataset_filepath,
    test_example_start_idx=0,
    test_example_num=1,
    summary_sentences_num=6,
):
    encoder, scorer = load_models(load_path)

    # uzmi iterator primjera od zadanog start_index
    examples = iter(
        extract_examples(dataset_filepath, start_index=test_example_start_idx)
    )

    with open("./rezultatiSimpleTest_LoadTest.txt", "w", encoding="utf-8") as f:

        # tqdm progress bar za testiranje vise primjera
        for i in tqdm(range(test_example_num), desc="TESTING", leave=False):
            ex = next(examples)

            summary = summarize(
                encoder,
                scorer,
                ex["article"],
                summary_sentences_num,
            )

            f.write("\n==========================\n")
            f.write(f"TEST {i+1}\n")
            f.write("==========================\n")
            f.write("\n\nARTICLE:\n")
            f.write(ex["article"] + "\n")
            f.write("\n\nSUMMARY:\n")
            f.write(summary + "\n")

            print("\n==========================")
            print(f"TEST {i+1}")
            print("==========================")
            print("\nARTICLE:\n")
            print(ex["article"])
            print("\nSUMMARY:\n")
            print(summary)

            print()


# --------------------------------
# evaluacija
# --------------------------------
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

    with open("./rezultatiSimpleTest_Evaluated.txt", "w", encoding="utf-8") as f:

        # tqdm progress bar kroz evaluacijske primjere
        for i in tqdm(range(num_examples), desc="EVALUATING"):

            ex = next(examples)

            article = ex["article"]
            abstract = ex.get("abstract")

            # odredi k: ili broj recenica u abstraktu ili omjer duljine clanka
            if match_abstract and abstract:
                k = len(sent_tokenize(abstract))
            else:
                k = max(1, round(fixed_ratio * len(sent_tokenize(article))))

            # generiraj ekstraktivni sazetak
            gen = summarize(encoder, scorer, article, k)

            # izracun metrika (rouge, density, redundancy...)
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
                f"EVAL {i+1}/{num_examples} | "
                f"R1F1={m.rouge1_f1:.3f} "
                f"D={m.density:.3f} "
                f"RR={m.redundancy_rate:.3f}"
            )

            f.write(
                f"EVAL {i+1}/{num_examples} | R1F1={m.rouge1_f1:.3f} D={m.density:.3f} RR={m.redundancy_rate:.3f}"
                + "\n"
            )

        # agregiraj metrike preko svih primjera
        avg = aggregate_metrics(metrics)

        print("\n=== AVERAGE METRICS ===")
        f.write("\n\n=== AVERAGE METRICS ===\n")
        for k, v in avg.items():
            print(f"{k}: {v:.4f}")
            f.write(f"{k}: {v:.4f}\n")

    return avg, metrics
