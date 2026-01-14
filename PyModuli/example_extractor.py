import csv
import json
import ast


def extract_examples(file_path, start_index = 0):
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for i in range(start_index):
            next(reader)

        for row_idx, row in enumerate(reader):
            if len(row) != 4:
                print(f"[WARNING] Skipping malformed row {row_idx}")
                continue

            article, abstract, sentences, sentence_labels = row
            article = article.strip()
            abstract = abstract.strip()

            sentences = ast.literal_eval(sentences)
            sentence_labels = json.loads(sentence_labels)

            if article and abstract:
                example = {
                    "article": article,
                    "abstract": abstract,
                    "sentences": sentences,
                    "labels": sentence_labels
                }
                yield example


def count_examples(file_path):
    count = 0
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for row_idx, row in enumerate(reader):
            if len(row) != 4:
                continue

            article, abstract, *_ = row
            if article.strip() and abstract.strip():
                count += 1

    return count




if __name__ == "__main__":
    # Example usage:
    file_path = "C:/Docs/01 - Filip/02 - FER/070 - G4S1/03-NM/Projekt/SkupPodataka-last.csv"

    total = count_examples(file_path)
    print("Total number of examples:", total)

    for i, example in enumerate(extract_examples(file_path)):
        print(example)
        input("input anything for next example:")


