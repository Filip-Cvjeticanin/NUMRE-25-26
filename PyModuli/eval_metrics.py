import math
import re
from collections import Counter
from dataclasses import dataclass
from rouge_score import rouge_scorer

# regex za izdvajanje rijeci iz teksta
#\b praznina, \w+ "barem jedan word znakovi" - a-z, A-Z, 0-9, _  , \b praznina
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE) #re.UNICODE -> radi s Unicode slovima (sa svim jezicima)

# tokenizira tekst u listu rijeci (za n-grame i metrike)
def tokenize_words(text, lowercase=True):
    # sve pretvori u mala slova
    if lowercase:
        text = text.lower()
    # izdvoji tokene pomocu regexa
    return _WORD_RE.findall(text)

# generira listu n-grama iz tokena (n >= 1)
def ngrams(tokens, n):
    # ako nema dovoljno tokena, nema n-grama
    if len(tokens) < n:
        return []

    # npr za n=2, tokens = ["ovo", "je", "tekst"]
    # tokens[i:i + n] -> ["ovo", "je"], ["je", "tekst"] -> tuple = ["ovo", "je"] -> ("ovo", "je") ...
    # tuple jer se kasnije koriste kao kljucevi
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# racuna duljinu najduzeg podudaranja na tocnoj poziciji
                                #gen. sazetak     #article index, summary index
def _max_match_len_at(article_tokens, summary_tokens, ai, si):
    # brojac duljine podudaranja
    k = 0
    # usporeduj tokene dok god se poklapaju
    # ogranicavanje indeksa sazetka i clanka da ne izadju van granica
    while ai + k < len(article_tokens) and si + k < len(summary_tokens):
        # ako se pojavi prvi nepodudarni token, prekida se
        if article_tokens[ai + k] != summary_tokens[si + k]:
            break
        #inace se podudara za +1 token
        k += 1
    return k


# pronalazi sve ekstraktivne fragmente generiranog sazetka u clanku
def extractive_fragments(article, summary, lowercase=True, min_len=1):
    # tokenizacija clanka i generiranog sazetka
    A = tokenize_words(article, lowercase)
    S = tokenize_words(summary, lowercase)

    frags = []      #lista pronadjenih ekstraktivnih fragmenata
    si = 0      #summary indeks

    # greedy prolaz kroz sazetak, dokle god ne dodjemo do kraja sazetka
    while si < len(S):
        best = 0
        # trazi najduzi match u cijelom clanku
        #za svaku rijec/indeks clanka
        for ai in range(len(A)):
            #najduzi pronadjeni kontinuirani match
            best = max(best, _max_match_len_at(A, S, ai, si))

        if best >= min_len:     #ako ima bar jednu rijec
            frags.append(S[si:si + best])       #uzima se fragment iz sazetka
            si += best      #nastavlja se nakon fragmenta
        else:
            si += 1     #inace provjeravamo od sljedece rijeci
    return frags

# racuna density mjeru (razinu ekstraktivnosti sazetka)
#           clanak, generirani sazetak
def density(article, summary, lowercase=True, min_fragment_len=1):
    # tokeni sazetka
    S = tokenize_words(summary, lowercase)
    # ako nema tokena, density je 0
    if not S:
        return 0.0
    # pronadi ekstraktivne fragmente
    frags = extractive_fragments(article, summary, lowercase, min_fragment_len)
    # formula: suma kvadrata duljina fragmenata / broj tokena
    return sum(len(f) ** 2 for f in frags) / float(len(S))

# racuna omjer novih n-grama u generiranom sazetku
def novel_ngrams_ratio(article, summary, n=2, lowercase=True):
    # tokeni clanka i sazetka
    A = tokenize_words(article, lowercase)
    S = tokenize_words(summary, lowercase)

    # n-grami generiranog sazetka
    Sg = ngrams(S, n)
    if not Sg:
        return 0.0

    # n-grami clanka
    Aset = set(ngrams(A, n))

    # broji n-grame koji se ne pojavljuju u clanku
    novel = sum(1 for g in Sg if g not in Aset)

    return novel / float(len(Sg))


# racuna mjeru redundancije temeljenu na entropiji
#                   #gen. sazetak
def redundancy_rate(summary, n=1, lowercase=True, eps=1e-12):
    # tokenizacija sazetka
    tokens = tokenize_words(summary, lowercase)
    # koristi unigrame (vec izracunate tokene) ili dohvati sve n-grame ako > 1
    grams = tokens if n == 1 else ngrams(tokens, n)
    grams = list(grams)

    if not grams:
        return 0.0

    # frekvencije pojavljivanja
    counts = Counter(grams)
    total = sum(counts.values())        #{"ovaj": 3, "model": 1, ... }

    # izracun vjerojatnosti
    probs = [c / total for c in counts.values()]
    # shannonova entropija
    H = -sum(p * math.log(max(p, eps)) for p in probs)

    # normalizacija entropije
    N = len(counts)     #broj jedinstvenih tokena za maksimalnu entropiju
    if N <= 1:
        return 0.0

    return float(H / math.log(N))


# racuna rouge-1 f1 mjeru izmedu referentnog i generiranog sazetka
#           ljudski sazetak, generirani sazetak, spajanje rijeci = True -> predmet = predmetno (smanjuje se osjetljivost na razlicite rijeci s istim "korijenom")
def rouge_1_f1(reference, hypothesis, use_stemmer=True):
    #inicijalizacija objekta za izracun
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=use_stemmer)
                #primjena na sazetke: scorer.score -> vraca preciznost, odziv i F1; ["rouge1"].fmeasure -> vraca F1
    return float(scorer.score(reference, hypothesis)["rouge1"].fmeasure)


# struktura za spremanje svih metrika jednog sazetka
@dataclass(frozen=True)
class SummaryMetrics:
    density: float
    novel_ngrams_ratio: float
    redundancy_rate: float
    rouge1_f1: float | None = None  # ako se nema referentni sazetak


# racuna sve metrike za jedan primjer sazetka
def evaluate_summary(
    article,
    generated_summary,
    reference_summary=None,
    novel_ngram_n=2,
    redundancy_ngram_n=1,
    lowercase=True,
    use_stemmer=True,
    min_fragment_len=1,
):
    # izracun metrika
    d = density(article, generated_summary, lowercase, min_fragment_len)
    nnr = novel_ngrams_ratio(article, generated_summary, novel_ngram_n, lowercase)
    rr = redundancy_rate(generated_summary, redundancy_ngram_n, lowercase)
    r1 = None       # rouge-1 se racuna samo ako postoji referentni sazetak
    if reference_summary is not None:
        r1 = rouge_1_f1(reference_summary, generated_summary, use_stemmer)

    return SummaryMetrics(d, nnr, rr, r1)

# agregira metrike preko vise primjera (prosjek)
                    #tipa SummaryMetrics
def aggregate_metrics(metrics):

    # pomocna funkcija za racunanje prosjeka
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # prosjek svake metrike
    return {
        "density": mean([m.density for m in metrics]),
        "novel_ngrams_ratio": mean([m.novel_ngrams_ratio for m in metrics]),
        "redundancy_rate": mean([m.redundancy_rate for m in metrics]),
        "rouge1_f1": mean([m.rouge1_f1 for m in metrics if m.rouge1_f1 is not None]),
    }
