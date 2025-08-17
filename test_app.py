import pytest
from app import load_data, recommend, compute_similarity

def test_load_data():
    movies, tfidf_matrix = load_data()
    assert movies is not None
    assert not movies.empty
    assert tfidf_matrix.shape[0] == len(movies)

def test_reccomend():
    result = recommend('The Dark Knight')
    assert isinstance(result, list)
    assert len(result) > 0
    assert "The Dark Knight" not in result

def test_compute_similarity():
    movies, tfidf = load_data()
    sim = compute_similarity(tfidf)
    assert sim.shape == (len(movies), len(movies))