import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk

# -----------------------------
# 1. Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SBERT_MODEL = "all-MiniLM-L6-v2"

DOWNLOAD_PUNKT = True

def configuration(condition):
    if condition:
        nltk.download("punkt")
        nltk.download("punkt_tab")

# -----------------------------
# 2. SBERT Sentence Encoder
# -----------------------------
class SBERTSentenceEncoder:
    def __init__(self, model_name=SBERT_MODEL):
        self.model = SentenceTransformer(model_name, device=DEVICE)

    @torch.no_grad()
    def encode(self, sentences):
        """
        Returns normalized sentence embeddings
        Shape: [num_sentences, embedding_dim]
        """
        embeddings = self.model.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embeddings

# -----------------------------
# 3. Duplicate-Free Alignment
# -----------------------------
def align_abstract_to_article(article_text, abstract_text, encoder):
    """
    Greedy 1–1 alignment (no replacement).

    Returns:
        matched_sentences: list[str]
        matched_indices: list[int]
        similarity_scores: list[float]
    """

    article_sents = [s.strip() for s in sent_tokenize(article_text)]
    abstract_sents = [s.strip() for s in sent_tokenize(abstract_text)]

    if len(article_sents) == 0 or len(abstract_sents) == 0:
        return [], [], []

    # Encode sentences
    article_embs = encoder.encode(article_sents)
    abstract_embs = encoder.encode(abstract_sents)

    # Cosine similarity matrix: [num_abstract, num_article]
    sim_matrix = torch.matmul(abstract_embs, article_embs.T)

    matched_sentences = []
    matched_indices = []
    similarity_scores = []

    used_article_indices = set()

    for i in range(sim_matrix.size(0)):
        sims = sim_matrix[i].clone()

        # Mask already used article sentences
        for idx in used_article_indices:
            sims[idx] = -1e9

        best_idx = torch.argmax(sims).item()

        used_article_indices.add(best_idx)
        matched_indices.append(best_idx)
        matched_sentences.append(article_sents[best_idx])
        similarity_scores.append(sims[best_idx].item())

    return matched_sentences, matched_indices, similarity_scores

# -----------------------------
# 4. Article sentence labeling
# -----------------------------
def label_article_sentences(article_text, abstract_text, encoder):
    """
    Returns a tensor of shape [num_article_sentences]
    with 1.0 for selected sentences, else 0.0
    """
    article_sents = [s.strip() for s in sent_tokenize(article_text)]
    labels = torch.zeros(len(article_sents), dtype=torch.float32)

    _, matched_indices, _ = align_abstract_to_article(
        article_text, abstract_text, encoder
    )

    for idx in matched_indices:
        labels[idx] = 1.0

    return labels.to(DEVICE)



def example_extraction():
    article = """
    Neural networks have been widely adopted in natural language processing.
    BERT introduced bidirectional transformers for language understanding.
    Extractive summarization selects important sentences from text.
    This approach avoids generating new content.
    Neural sentence scoring allows trainable summarization models.
    """
    abstract = """
    BERT introduced transformers and understanding of language.
    In extractive summarization we select important sentences.
    """
    extracted_summary = text_extraction(article, abstract)
    return extracted_summary



def text_extraction(article: str, abstract: str, override_download_condition: bool | None = None) -> str:
    """
    Converts a pair of (article, abstract) into an extracted summary and returns it.
    :param article:
    :param abstract:
    :param override_download_condition:
    :return:
    """

    # Download punkt if needed.
    download_condition = DOWNLOAD_PUNKT
    if override_download_condition is not None:
        download_condition = override_download_condition
    configuration(download_condition)

    # Initialize encoder.
    encoder = SBERTSentenceEncoder()

    # Match sentences.
    matched, indices, scores = align_abstract_to_article(
        article, abstract, encoder
    )

    # Output and return.
    print("\n\nMatched sentences:")
    for i, (s, idx, sc) in enumerate(zip(matched, indices, scores)):
        print(f"{i + 1}. [Article idx {idx}] (sim={sc:.3f})")
        print(f"   {s}")
    labels = label_article_sentences(article, abstract, encoder)
    print("Labels:")
    print(labels)
    print("Returning text extracted summary.")

    extracted_summary = " ".join(matched)
    return extracted_summary


def label_tensor_extraction(article: str, abstract: str, override_download_condition: bool | None = None) -> torch.Tensor:
    """
    Converts a pair of (article, abstract) into a tensor labels - 0 in that sentence isn't matched, 1 if it is.
    :param article:
    :param abstract:
    :param override_download_condition:
    :return:
    """
    # Download punkt if needed.
    download_condition = DOWNLOAD_PUNKT
    if override_download_condition is not None:
        download_condition = override_download_condition
    configuration(download_condition)

    # Initialize encoder.
    encoder = SBERTSentenceEncoder()

    # Match sentences.
    matched, indices, scores = align_abstract_to_article(
        article, abstract, encoder
    )

    # Output and return.
    print("\n\nMatched sentences:")
    for i, (s, idx, sc) in enumerate(zip(matched, indices, scores)):
        print(f"{i + 1}. [Article idx {idx}] (sim={sc:.3f})")
        print(f"   {s}")
    labels = label_article_sentences(article, abstract, encoder)
    print("Labels:")
    print(labels)
    print("Returning labels tensor")

    return labels



# -----------------------------
# 5. Example / Sanity Check
# -----------------------------
if __name__ == "__main__":
    extracted_summary = example_extraction()
    print(extracted_summary)
