import os
import re
import csv
import hashlib
import unicodedata

import pandas as pd
import numpy as np

import torch
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


#============PUTANJE I IZLAZ==========================

# putanje do izvornih csv splitova
trainPutanja = r"C:\Users\Sami\Documents\SAMI\Faks\FAKS\4.g\1. sem\NM\PROJEKT\Skup podataka\OriginalniSkup\train.csv"
valPutanja   = r"C:\Users\Sami\Documents\SAMI\Faks\FAKS\4.g\1. sem\NM\PROJEKT\Skup podataka\OriginalniSkup\validation.csv"
testPutanja  = r"C:\Users\Sami\Documents\SAMI\Faks\FAKS\4.g\1. sem\NM\PROJEKT\Skup podataka\OriginalniSkup\test.csv"

# bazni direktorij i poddirektorij za izlaz
izlazDir = r"C:\Users\Sami\Documents\SAMI\Faks\FAKS\4.g\1. sem\NM\PROJEKT\Skup podataka\probaFiltracije"
os.makedirs(izlazDir, exist_ok=True)

izlazPoddir = os.path.join(izlazDir, "post_clean")
os.makedirs(izlazPoddir, exist_ok=True)

# broj primjera u uzorku, seed, velicina chunkova
brojPrimjera = 120
seed = 1
velicinaChunka = 5000

# izlazna datoteka
izlazPutanja = os.path.join(izlazPoddir, f"Predobrada{brojPrimjera}.csv")

#===================================================


#============FILTRACIJA ZAPISA=======================

# pragovi duljine u rijecima
minRijeciClanak = 50
minRijeciSazetak = 30
maxRijeciClanak = 5000

# regex: jedan ili vise razmaka ili tabova
# primjer: "a\t   b" -> "a b"
razmakTabRegex = re.compile(r"[ \t]+")

# regex: tri ili vise uzastopna newlinea
# primjer: "a\n\n\n\nb" -> "a\n\nb"
viseNovihRedovaRegex = re.compile(r"\n{3,}")

# regex: ne-whitespace tokeni
# primjer: "hello   world\nx" -> ["hello","world","x"] (n = 3)
rijecRegex = re.compile(r"\S+")     #\S+ jedan ili vise znakova (ne white-space)

def normalizirajTekst(text):
    # ako je none, vrati prazan string
    if text is None:
        return ""

    text = str(text)

    # zamijeni null byte i non-breaking space
    text = text.replace("\x00", " ")
    text = text.replace("\u00A0", " ")

    # nfkc normalizacija
    text = unicodedata.normalize("NFKC", text)

    # standardiziraj newlineove
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # kolapsiraj razmake i tabove
    text = razmakTabRegex.sub(" ", text)

    # trimaj svaku liniju
    text = "\n".join(line.strip() for line in text.split("\n"))

    # ograniči prazne redove
    text = viseNovihRedovaRegex.sub("\n\n", text)

    return text.strip()

def normalizirajSeriju(serija):
    # pretvorba u string dtype
    serija = serija.astype("string")
    return serija.map(normalizirajTekst)

def prebrojiRijeci(serija):
    # broj rijeci = broj ne-whitespace tokena
    return serija.astype(str).map(lambda x: len(rijecRegex.findall(x)))

def filtrirajZapise(df):
    # zadrzi samo potrebne stupce
    df = df[["article", "abstract"]].copy()

    # ukloni zapise s missing vrijednostima
    df = df.dropna(subset=["article", "abstract"])

    # normaliziraj prije brojanja i drugih koraka
    df["article"] = normalizirajSeriju(df["article"])
    df["abstract"] = normalizirajSeriju(df["abstract"])

    # ukloni zapise koji su prazni nakon normalizacije
    df = df[(df["article"].str.len() > 0) & (df["abstract"].str.len() > 0)].copy()

    # izracunaj brojeve rijeci
    df["rijeciClanak"] = prebrojiRijeci(df["article"])
    df["rijeciSazetak"] = prebrojiRijeci(df["abstract"])

    # primijeni pragove duljine
    df = df[
        (df["rijeciClanak"] >= minRijeciClanak) &
        (df["rijeciSazetak"] >= minRijeciSazetak) &
        (df["rijeciClanak"] <= maxRijeciClanak)
    ].copy()

    return df

#===================================================


#============UKLANJANJE POSEBNIH ODJELJAKA============

# header mora poceti u zadnjih 30% teksta
minUdioTeksta = 0.70

zaglavljaOdjeljaka = [
    "references", "reference", "bibliography",
    "literature", "literature cited", "works cited",
    "acknowledgements", "acknowledgments",
    "funding", "financial support",
    "conflict of interest", "conflicts of interest", "competing interests",
    "ethics", "ethical approval",
    "author contributions", "contributions", "disclosure",
    "supplementary material", "supplementary materials", "appendix",
]

zaglavljaAlternacija = "|".join(re.escape(h) for h in zaglavljaOdjeljaka)

# regex za rezanje od standalone header linije do kraja dokumenta
# - (?im): ignorecase + multiline
    #\s* nula ili vise whitespace znakova, ?: nije bitan sadrzaj, [:\-]? opcionalno : ili - iza riječi, $ kraj retka
# - ^\s*(?:headers)\s*[:\-]?\s*$ : header kao samostalni redak, opcionalno ":" ili "-"
# - .* \Z : sve do kraja stringa
# primjer gdje se reze:
#   "...text...\nReferences\n[1] ...\n" -> reze od "References" do kraja
# primjer gdje se ne reze (header nije standalone):
#   "...in references we show...\n" -> ne reze jer nije samostalni redak
rezanjeRegex = re.compile(rf"(?im)^\s*(?:{zaglavljaAlternacija})\s*[:\-]?\s*$.*\Z", re.DOTALL)

def ukloniOdjeljke(text):
    # ako je prazan tekst, nema rezanja
    if not text:
                    #headerPronadjen, rezanjeObavljeno
        return text, False, False

    # duljina originala radi pravila zadnjih 30%
    duljina = len(text)

    # pronadi pocetak header bloka
    match = rezanjeRegex.search(text)
    if not match:
        return text, False, False

    # indeks gdje pocinje header
    pocetak = match.start()

    # rezi samo ako je header dovoljno kasno u tekstu
    if pocetak < minUdioTeksta * duljina:
        return text, True, False

    # odrezi sve od headera do kraja
    noviTekst = text[:pocetak].strip()
    return noviTekst, True, True

#===================================================


#============DEDUPLIKACIJA===========================

def hashZapis(clanak, sazetak):
    # hash para (article, abstract) za detekciju duplikata
    kljuc = (clanak + "\n<SEP>\n" + sazetak).encode("utf-8", errors="ignore")
    return hashlib.md5(kljuc).hexdigest()

#===================================================


#============OZNACAVANJE RECENICA====================

# odaberi gpu ako postoji
uredaj = "cuda" if torch.cuda.is_available() else "cpu"
sbertModel = "all-MiniLM-L6-v2"

@torch.no_grad()
def kodirajRecenice(model, recenice):
    # encode + normalize => dot product = cos sim
    return model.encode(
        recenice,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

def razdijeliRecenice(text):
    # punkt tokenizacija recenica
    if text is None:
        return []
    return [s.strip() for s in sent_tokenize(str(text)) if s.strip()]

def poravnajRecenice(model, receniceClanka, receniceSazetka):
    # greedy 1-1 uparivanje: svakoj recenici sazetka dodijeli jednu recenicu clanka (bez ponavljanja)
    if len(receniceClanka) == 0 or len(receniceSazetka) == 0:
        return []

    # izracunaj embeddinge
    embClanak = kodirajRecenice(model, receniceClanka)
    embSazetak = kodirajRecenice(model, receniceSazetka)

    # matrica slicnosti [len(abs) x len(art)]
    slicnosti = embSazetak @ embClanak.T

    iskoristeno = set()
    indeksi = []

    for i in range(slicnosti.size(0)):
        # kopija reda slicnosti za i-tu recenicu sazetka
        red = slicnosti[i].clone()

        # maskiraj vec odabrane recenice clanka
        if iskoristeno:
            red[list(iskoristeno)] = -1e9

        # odaberi najvecu slicnost medu neodabranima
        najbolji = int(torch.argmax(red).item())
        iskoristeno.add(najbolji)
        indeksi.append(najbolji)

    return indeksi

#===================================================


#============UZORKOVANJE U CHUNKOVIMA================

def uzorkujZapise(putanje, n, velicina, seed):
    # reservoir sampling daje uniforman uzorak velicine n bez ucitavanja svega
    rng = np.random.default_rng(seed)
    spremnik = []
    vidjeno = 0

    for putanja in putanje:
        for chunk in tqdm(
            pd.read_csv(putanja, usecols=["article", "abstract"], chunksize=velicina),
            desc=f"ucitavanje ({os.path.basename(putanja)})",
            unit="chunk"
        ):
            # prolaz po redovima trenutnog chunka
            for clanak, sazetak in chunk.itertuples(index=False, name=None):
                vidjeno += 1
                zapis = {"article": clanak, "abstract": sazetak}

                # prvo napuni reservoir do n
                if len(spremnik) < n:
                    spremnik.append(zapis)
                else:
                    # slucajni indeks u [0, seen-1]
                    j = rng.integers(0, vidjeno)

                    # s vjerojatnoscu n/seen zamijeni element u reservoiru
                    if j < n:
                        spremnik[j] = zapis

    # pretvori u dataframe i dodatno promijesaj redoslijed unutar uzorka
    df = pd.DataFrame(spremnik, columns=["article", "abstract"])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

#===================================================


#============GLAVNI PIPELINE=========================

def pokreniPredobradu(trainPut, valPut, testPut, izlazPut):
    # osiguraj punkt tokenizer
    nltk.download("punkt", quiet=True)

    # ucitaj sbert model na odabrani uredaj
    model = SentenceTransformer(sbertModel, device=uredaj)

    # uzorkuj max_examples iz svih splitova bez ucitavanja cijelog skupa
    df = uzorkujZapise([trainPut, valPut, testPut], brojPrimjera, velicinaChunka, seed)

    # primijeni filtraciju + normalizaciju
    df = filtrirajZapise(df)

    # ukloni posebne odjeljke iz clanka
    df["article"] = [ukloniOdjeljke(str(t))[0] for t in df["article"].tolist()]

    # izbaci prazne nakon rezanja
    df = df[
        (df["article"].str.len() > 0) &
        (df["abstract"].str.len() > 0)
    ].copy()

    # dedupe + oznacavanje recenica
    vidjeniHash = set()
    izlazniRedci = []

    for _, redak in tqdm(df.iterrows(), total=len(df), desc="predobrada + oznacavanje", unit="row"):
        # procitaj polja kao string
        clanak = str(redak["article"])
        sazetak = str(redak["abstract"])

        # preskoci duplikate
        h = hashZapis(clanak, sazetak)
        if h in vidjeniHash:
            continue
        vidjeniHash.add(h)

        # razbij tekst na recenice
        receniceClanka = razdijeliRecenice(clanak)
        receniceSazetka = razdijeliRecenice(sazetak)

        # greedy uparivanje recenica
        matchedIdx = poravnajRecenice(model, receniceClanka, receniceSazetka)

        # label 1 za odabrane recenice clanka, 0 za ostale
        oznake = np.zeros(len(receniceClanka), dtype=int)
        for idx in matchedIdx:
            if 0 <= idx < len(oznake):
                oznake[idx] = 1

        # spremi tocno 4 stupca u izlaz
        izlazniRedci.append({
            "article": clanak,
            "abstract": sazetak,
            "sentences": str(receniceClanka),
            "sentence_labels": str(oznake.tolist())
        })

        # ogranicenje izlaza na najvise max_examples
        if len(izlazniRedci) >= brojPrimjera:
            break

    # spremi konacni dataframe u csv
    outDf = pd.DataFrame(izlazniRedci, columns=["article", "abstract", "sentences", "sentence_labels"])
    outDf.to_csv(izlazPut, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

#===================================================


if __name__ == "__main__":
    pokreniPredobradu(trainPutanja, valPutanja, testPutanja, izlazPutanja)
