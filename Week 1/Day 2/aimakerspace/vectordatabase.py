import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = []
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, chunk: dict) -> None:
        self.vectors.append(chunk)

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
    ) -> List[dict]:
        scored_chunks = [
            {**chunk, 'score': distance_measure(query_vector, chunk['embedding'])}
            for chunk in self.vectors
        ]
        return sorted(scored_chunks, key=lambda x: x['score'], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
    ) -> List[dict]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return [chunk for chunk in self.vectors if chunk['text'] == key][0]['embedding']

    async def abuild_from_list(self, list_of_text: List[str] | List[dict]) -> "VectorDatabase":
        if type(list_of_text[0]) == str:
            embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
            list_of_chunks = [{'text': txt, 'embedding': embedding} for txt, embedding in zip(list_of_text, embeddings)]
        elif type(list_of_text[0]) == dict:
            assert 'text' in list_of_text[0].keys(), "chunk object must contain 'text' attribute"
            texts = [chunk['text'] for chunk in list_of_text]
            embeddings = await self.embedding_model.async_get_embeddings(texts)
            list_of_chunks = [{**chunk, 'embedding': embedding} for chunk, embedding in zip(list_of_text, embeddings)]
        
        self.vectors = list_of_chunks
        
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
