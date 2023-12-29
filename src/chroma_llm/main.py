import os
import openai
import json
import chromadb
from chromadb.utils import embedding_functions
from pprint import pprint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DATA_PATH = "D:/udemy/transformers_for_nlp/data/car_reviews_test"
CHROMA_PATH = "D:/udemy/transformers_for_nlp/database/car_review_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "car_reviews"

with open('D:/udemy/transformers_for_nlp/files/openai-key.json') as f:
    openai_api_key = json.load(f).get('openai_key')

openai.api_key = openai_api_key

client = chromadb.PersistentClient(CHROMA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_FUNC_NAME
)

collection = client.get_collection(
    name=COLLECTION_NAME, embedding_function=embedding_func
)

context = """
You are a customer success employee at a large
 car dealership. Use the following car reviews
 to answer questions: {}
"""

question = """
What's the key to great customer satisfaction
 based on detailed positive reviews? List 5 essential points.
"""

good_reviews = collection.query(
    query_texts=[question],
    n_results=5,
    include=["documents"],
    where={"Rating": {"$gte": 3}},
)


reviews_str = ",".join(good_reviews["documents"][-1])
good_review_summaries = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": context.format(reviews_str)},
        {"role": "user", "content": question},
    ],
    temperature=0,
    n=1,
)

pprint(good_review_summaries["choices"][0]["message"]["content"],
       indent=2)