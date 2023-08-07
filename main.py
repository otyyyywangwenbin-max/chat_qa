from core.qa import *
from core.llm.chatglm2 import ChatGLM2
from core.embedding.text2vec_base_chinese import Text2vecEmbedding
from core.vectordb.chroma import ChromaDB

llm = ChatGLM2("http://localhost:8000")
embedding = Text2vecEmbedding()


print(isinstance(llm, LLM))
print(dir(llm))