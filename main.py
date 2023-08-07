from core.qa import *
from core.llm.chatglm2 import ChatGLM2
from core.embedding.text2vec_base_chinese import Text2vecEmbedding
from core.vectordb.chroma import ChromaDB

llm = ChatGLM2("http://localhost:8000")
embedding = Text2vecEmbedding()
vectordb = ChromaDB(llm = llm, embedding = embedding)

print(isinstance(llm, LLM))
print(isinstance(embedding, Embedding))
print(isinstance(vectordb, VectorDB))

print(dir(llm))