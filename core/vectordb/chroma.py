#!/usr/bin/env python
# -*- coding:utf-8 _*-
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from ..qa import *


class MyEmbeddingFunction(EmbeddingFunction):
    embedding: Embedding

    def __init__(self, embedding: Embedding):
        self.embedding = embedding

    def __call__(self, texts: Documents) -> Embeddings:
        # embed the documents somehow
        embeddings=[]
        for text in texts:
            embeddings.append(self.embedding.encode(text).tolist())
        return embeddings


class ChromaDB(VectorDB):

    def __init__(self, embedding: Embedding, llm:LLM):
        persist_directory = "chromadb"
        collection_name = "primeton_qa"
        self.llm = llm
        self.client = chromadb.PersistentClient(path = persist_directory)
        self.collection = self.client.get_or_create_collection(name = collection_name
                , metadata = {"hnsw:space": "cosine"})

    def add(self, datas: List[Dict]):
        ids = []
        documents = []
        embeddings = []
        for data in datas:
            if data.has_key("text"):
                query = data["text"]
                text = data["text"]
            else:
                query = data["q"]
                text = data["a"]
            ids.append(data["id"])
            documents.append(text)
            embeddings.append(self.embedding.encode(text).tolist())
        self.collection.add(ids = ids, embeddings = embeddings, documents = documents)
    
    def query(self, query: str, topn: int = 2) -> List[str]:
        result = self.collection.query(query_embeddings=[self.embedding.encode(query).tolist()], n_results = topn)
        return sum(result["documents"], [])