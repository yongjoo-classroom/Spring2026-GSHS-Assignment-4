import math

def tokenize(text: str) -> list[str]:
    return text.lower().split()

def compute_tf(document: str) -> dict:
    tf = {}
    tokens = tokenize(document)
    N = len(tokens)

    for word in tokens:
        tf[word] = tf.get(word, 0) + 1
    
    for word in tf:
        tf[word] /= N

    return tf

def compute_idf(docs: list[str]) -> dict:
    idf = {}
    N = len(docs)
    all_words = set()

    for doc in docs:
        all_words.update(tokenize(doc))

    for word in all_words:
        cnt = 0
        for doc in docs:
            if word in tokenize(doc):
                cnt += 1
        idf[word] = math.log(N / cnt)

    return idf

def compute_tf_idf(document: str, idf: dict) -> dict:
    tf_idf = {}
    tf = compute_tf(document)

    for word, x in tf.items():
        tf_idf[word] = x * idf.get(word, 0)

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
    idf = compute_idf(documents)
    query_vec = compute_tf_idf(query, idf)
    scores = []

    for doc in documents:
        doc_vec = compute_tf_idf(doc, idf)
        score = cosine_similarity(query_vec, doc_vec)
        scores.append((doc, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]