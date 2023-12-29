import pathlib
import chromadb
from tqdm import tqdm
from chromadb.utils import embedding_functions
from more_itertools import batched
from chroma_llm import data_preparation
import random

def build_chroma_collection(
    chroma_path: pathlib.Path | str,
    collection_name: str,
    embedding_func_name: str,
    ids: list[int],
    documents: list[str],
    metadatas: list[dict],
    distance_func_name: str = "cosine",
):
    """Create a ChromaDB collection"""

    chroma_client = chromadb.PersistentClient(chroma_path)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_func_name
    )

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": distance_func_name},
    )

    document_indices = list(range(len(documents)))
    print(f'Number of documents to scan: {len(document_indices)}')
    pbar = tqdm(total=len(document_indices), desc="Processing batches")
    for batch in batched(document_indices, 500):
        start_idx = batch[0]
        end_idx = batch[-1]
        collection.add(
            ids=ids[start_idx:end_idx],
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
        )

        pbar.update(len(batch))
    pbar.close()
    print("***********FINISHED PROCESSING ***************")

def get_car_reviews(data_path: pathlib.Path | str) -> dict[str, list]:
    car_reviews = data_preparation.get_car_reviews(data_path)
    car_reviews_as_dicts = [car_review.model_dump(by_alias=True) for car_review in car_reviews]
    random.shuffle(car_reviews_as_dicts)
    car_review_ids = [str(car_review.get('id')) for car_review in car_reviews_as_dicts]
    car_review_texts = [car_review.get('reviewText') for car_review in car_reviews_as_dicts]
    car_review_metadata = [
        {k: v for k, v in car_review.items() if k not in ['reviewText', 'id', 'reviewDate']}
        for car_review in car_reviews_as_dicts
    ]
    print(f'Number of review ids: {len(car_review_ids)}')
    print(f'Number of review texts: {len(car_review_texts)}')
    print(f'Number of review metadata: {len(car_review_metadata)}')

    return {
        'ids': car_review_ids,
        'documents': car_review_texts,
        'metadatas': car_review_metadata
    }

if __name__ == "__main__":
    DATA_PATH = "D:/udemy/transformers_for_nlp/data/car_reviews"
    CHROMA_PATH = "D:/udemy/transformers_for_nlp/database/car_review_embeddings"
    EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
    COLLECTION_NAME = "car_reviews"
    car_reviews = get_car_reviews(DATA_PATH)
    try:
        build_chroma_collection(
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            embedding_func_name=EMBEDDING_FUNC_NAME,
            ids=car_reviews['ids'],
            documents=car_reviews['documents'],
            metadatas=car_reviews['metadatas']
        )
    except Exception as ex:
        print(f'Connection exists: {str(ex)}.')