import math

def tokenize(text: str) -> list[str]:
    '''
    Tokenize the input text into words.
    '''
    return text.lower().split()

def compute_tf(document: str) -> dict:
    '''
    Given a document, compute the term frequency (TF) for each word.
    Parameters:
        - document: The input document string.

    Returns:
        - A dictionary representing the term frequency (TF) of each word in the document.
          e.g, tf = {'cats': 0.5, 'are': 0.25, 'small': 0.25}
    '''
    tf = {}
    tokens = tokenize(document)

    # Compute term frequency (TF)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return tf
    
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    
    for word in tf:
        tf[word] /= total_tokens

    return tf

def compute_idf(docs: list[str]) -> dict:
    '''
    Given a list of documents, compute the inverse document frequency (IDF) for each word.
    Parameters:
        - docs: A list of document strings.

    Returns:
        - A dictionary representing the inverse document frequency (IDF) of each word.
          e.g, idf = {'cats': 1.0, 'are': 0.5, 'small': 0.5}
    '''
    idf = {}
    N = len(docs)
    all_words = set()

    for doc in docs:
        all_words.update(tokenize(doc))

    # Compute inverse document frequency (IDF), given entire vocabulary \
    # from all the documents (all_words)
    for word in all_words:
        doc_count = 0
        for doc in docs:
            if word in tokenize(doc):
                doc_count += 1
        idf[word] = math.log(N / doc_count) if doc_count > 0 else 0

    return idf

def compute_tf_idf(document: str, idf: dict) -> dict:
    '''
    Given a document and the IDF values, compute the TF-IDF for each word.
    Parameters:
        - document: The input document string.
        - idf: A dictionary representing the inverse document frequency (IDF) of all words.

    Returns:
        - A dictionary representing the TF-IDF of each word in the document.
          e.g, tf_idf = {'cats': 0.5, 'are': 0.25, 'small': 0.25}
    '''
    tf_idf = {}
    tf = compute_tf(document)

    # Compute TF-IDF
    for word in tf:
        tf_idf[word] = tf[word] * idf.get(word, 0)

    return tf_idf


def cosine_similarity(vec1: dict, vec2: dict) -> float:
    '''
    Compute the cosine similarity between two vectors.
    Parameters:
    
        - vec1: The first vector (dictionary).
        - vec2: The second vector (dictionary).

    Returns:
        - The cosine similarity score (float).
    '''
    dot = 0
    for word in vec1:
        dot += vec1[word] * vec2.get(word, 0)

    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0

    return dot / (mag1 * mag2)

def tf_idf_search(query: str, documents: list[str]) -> str:
    '''
    Given a query and a list of documents, return the most relevant document.
    Parameters:
        - query: The search query string.
        - documents: A list of document strings to search within.

    Returns:
       - The most relevant document string.
    '''
    idf = compute_idf(documents)
    query_vec = compute_tf_idf(query, idf)
    scores = []

    # For each document, compute the cosine similarity with the query vector
    for doc in documents:
        doc_vec = compute_tf_idf(doc, idf)
        score = cosine_similarity(query_vec, doc_vec)
        scores.append((doc, score))

    # Sort the documents by their score
    scores.sort(key=lambda x: x[1], reverse=True)
    # print(scores)
    return scores[0][0]