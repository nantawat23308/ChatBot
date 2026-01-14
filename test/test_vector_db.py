import unittest
from src.vector_db.searching import (
    search_similar_documents,
    search_with_score,
    vector_store_as_retriever,
)


class MyTestCase(unittest.TestCase):
    def test_search_similar_documents(self):
        query = "Sample query for testing"
        results = search_similar_documents(
            query, collection_name="test_collection", top_k=3
        )
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)

    def test_search_with_score(self):
        query = "Another sample query for testing"
        results_with_scores = search_with_score(
            query, collection_name="test_collection", top_k=3
        )
        self.assertIsInstance(results_with_scores, list)
        self.assertLessEqual(len(results_with_scores), 3)
        for doc, score in results_with_scores:
            self.assertIsInstance(score, float)

    def test_vector_store_as_retriever(self):
        retriever = vector_store_as_retriever(
            collection_name="test_collection", top_k=3
        )
        self.assertIsNotNone(retriever)
        sample_query = "Test query for retriever"
        results = retriever.invoke(sample_query)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
