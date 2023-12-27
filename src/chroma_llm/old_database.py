import chromadb
from chromadb.utils import embedding_functions
from old_documents import DOCUMENTS, GENRES

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

print(client)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

collection = None
try:
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )
except Exception as ex:
    collection = client.get_collection(name=COLLECTION_NAME)

print(collection)
collection.add(documents=DOCUMENTS,
               ids=[f"id{i}" for i in range(len(DOCUMENTS))],
               metadatas=[{"genre": genre} for genre in GENRES])

print(collection)
query_results = collection.query(
    query_texts=["Find me some delicious food!"],
    n_results=1,
)

print(query_results)
print(type(query_results))

query_results = collection.query(
    query_texts=["Teach me about music history"],
    where={"genre": {"$eq": "music"}},
    n_results=1,
    include=["documents", "distances"]
)

print(query_results)