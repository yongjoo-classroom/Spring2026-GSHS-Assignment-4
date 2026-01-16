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
    tokenized_docs = []
    for doc in docs:
        toks = tokenize(doc)
        tokenized_docs.append(toks)
        all_words.update(toks)

    idf = {}
    for w in all_words:
        df = 0
        for toks in tokenized_docs:
            if w in set(toks):
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
    dot = 0.0
    for word, v1 in vec1.items():
        dot += v1 * vec2.get(word, 0.0)

    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)

def tf_idf_search(query: str, documents: list[str]) -> str:
    idf = compute_idf(documents)
    query_vec = compute_tf_idf(query, idf)
    scores = []

    for doc in documents:
        doc_vec = compute_tf_idf(doc, idf)
        score = cosine_similarity(query_vec, doc_vec)
        scores.append((doc, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]
