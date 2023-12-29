import chromadb
from chromadb.utils import embedding_functions
DATA_PATH = "D:/udemy/transformers_for_nlp/data/car_reviews_test"
CHROMA_PATH = "D:/udemy/transformers_for_nlp/database/car_review_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "car_reviews"

client = chromadb.PersistentClient(CHROMA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_FUNC_NAME
    )
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

great_reviews = collection.query(
    query_texts=["Find me some positive reviews with rating above 4 that discuss the car's performance"],
    n_results=5,
    include=["documents", "distances", "metadatas"]
)
#
print(great_reviews)