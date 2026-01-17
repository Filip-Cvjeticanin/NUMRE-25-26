import csv
import json
import ast

# cita primjere iz csv datoteke i vraca ih jedan po jedan (generator)
def extract_examples(file_path, start_index=0):
    # otvara csv datoteku u utf-8 encodingu
    with open(file_path, newline="", encoding="utf-8") as f:
        # inicijalizira csv reader
        reader = csv.reader(f)

        # preskace header (prvi red)
        next(reader, None)

        # preskace retke do zadanog pocetnog indeksa
        for _ in range(start_index):
            next(reader, None)

        # prolazi kroz svaki preostali red u csv-u
        for row_idx, row in enumerate(reader, start=start_index):
            # ocekuje tocno 4 stupca po retku
            if len(row) != 4:
                continue

            # raspakirava stupce
            article, abstract, sentences, sentence_labels = row

            # uklanja visak razmaka i None vrijednosti
            article = (article or "").strip()
            abstract = (abstract or "").strip()

            # ako nema clanka ili sazetka, preskace primjer
            if not article or not abstract:
                continue

            # parsira listu recenica iz stringa
            try:                                        
                #iz "['A sentence.', 'Another sentence.']" u ['A sentence.', 'Another sentence.']    
                #iz stringa recenica u listu recenica
                sents = ast.literal_eval(sentences)    
            except Exception:
                sents = []

            # parsira oznake recenica iz json stringa
            try:

                #iz "[0, 1, 0, 1]" u [0, 1, 0, 1] (razlika navodni znakovi)
                labels = json.loads(sentence_labels)
            except Exception:
                labels = []

            # vraca jedan primjer kao rjecnik
            yield {
                "article": article,
                "abstract": abstract,
                "sentences": sents,
                "labels": labels,
            }

