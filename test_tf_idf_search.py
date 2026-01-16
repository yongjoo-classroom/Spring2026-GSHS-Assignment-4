from __future__ import annotations

import math
from typing import Dict, List


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def compute_tf(document: str) -> Dict[str, float]:
    tokens = tokenize(document)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {}

    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    tf: Dict[str, float] = {}
    for word, count in counts.items():
        tf[word] = count / total_tokens
    return tf


def compute_idf(docs: List[str]) -> Dict[str, float]:
    num_docs = len(docs)
    if num_docs == 0:
        return {}

    doc_freq: Dict[str, int] = {}
    for doc in docs:
        seen_in_doc = set(tokenize(doc))
        for token in seen_in_doc:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    idf: Dict[str, float] = {}
    for token, df in doc_freq.items():
        idf[token] = math.log(num_docs / df)
    return idf


def compute_tf_idf(document: str, idf: Dict[str, float]) -> Dict[str, float]:
    tf_idf: Dict[str, float] = {}
    tf = compute_tf(document)
    for token, tf_value in tf.items():
        idf_value = idf.get(token)
        if idf_value is not None:
            tf_idf[token] = tf_value * idf_value
    return tf_idf


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1

    dot = 0.0
    for word, v1 in vec1.items():
        dot += v1 * vec2.get(word, 0.0)

    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


def tf_idf_search(query: str, documents: List[str]) -> str:
    if not documents:
        raise ValueError("documents list must not be empty")

    idf = compute_idf(documents)
    query_vec = compute_tf_idf(query, idf)

    best_doc = documents[0]
    best_score = -1.0
    for doc in documents:
        doc_vec = compute_tf_idf(doc, idf)
        score = cosine_similarity(query_vec, doc_vec)
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc
