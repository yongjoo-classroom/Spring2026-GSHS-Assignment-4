from tf_idf_search import tf_idf_search

documents = [
    "cats are small animals",
    "dogs are loyal animals",
    "cats and dogs can be pets",
    "cars and bikes are vehicles",
    "trucks and cars move goods"
]

def test_tf_idf_search_1():
    '''
    Test search function with query: Are cats pets?
    '''
    query = "Are cats pets"
    expected_doc = "cats and dogs can be pets"
    pred_doc = tf_idf_search(query, documents)
    assert pred_doc == expected_doc, f"Expected: {expected_doc}, but got: {pred_doc}"

def test_tf_idf_search_2():
    '''
    Test search function with a query: Are cars vehicles?
    '''
    query = "Are cars vehicles"
    expected_doc = "cars and bikes are vehicles"
    pred_doc = tf_idf_search(query, documents)
    assert pred_doc == expected_doc, f"Expected: {expected_doc}, but got: {pred_doc}"
