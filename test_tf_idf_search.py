# tf_idf_search.py
import math

def tokenize(text: str) -> list[str]:
    return text.lower().split()

def compute_tf(document: str) -> dict:
    tokens = tokenize(document)
    n = len(tokens)
    if n == 0:
        return {}
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    tf = {}
    for w, c in counts.items():
        tf[w] = c / n
    return tf

def compute_idf(docs: list[str]) -> dict:
    N = len(docs)
    if N == 0:
        return {}
    all_words = set()
    for doc in docs:
        all_words.update(tokenize(doc))
    idf = {}
    for w in all_words:
        df = 0
        for doc in docs:
            if w in set(tokenize(doc)):
                df += 1
        idf[w] = math.log(N / df) if df > 0 else 0.0
    return idf

def compute_tf_idf(document: str, idf: dict) -> dict:
    tf = compute_tf(document)
    tf_idf = {}
    for w, tfv in tf.items():
        if w in idf:
            tf_idf[w] = tfv * idf[w]
    return tf_idf

def cosine_similarity(vec1: dict, vec2: dict) -> float:
    dot = 0
    for word in vec1:
        dot += vec1[word] * vec2.get(word, 0)

    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0

    return dot / (mag1 * mag2)

def tf_idf_search(query: str, documents: list[str]) -> str:
    idf = compute_idf(do_
