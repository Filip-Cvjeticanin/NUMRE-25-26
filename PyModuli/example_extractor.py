import csv

def extract_examples(file_path):
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for row_idx, row in enumerate(reader):
            if len(row) != 4:
                print(f"[WARNING] Skipping malformed row {row_idx}")
                continue

            article, abstract, sentences, sentence_labels = row
            article = article.strip()
            abstract = abstract.strip()

            if article and abstract:
                example = {
                    "article": article,
                    "abstract": abstract,
                    "sentences": sentences,
                    "sentence_labels": sentence_labels
                }
                yield example



if __name__ == "__main__":
    # Example usage:
    file_path = "C:/Docs/01 - Filip/02 - FER/070 - G4S1/03-NM/Projekt/SkupPodataka-last.csv"
    for i, example in enumerate(extract_examples(file_path)):
        print(example)
        input()


