from core.qa import *
from core.llm.chatglm2 import ChatGLM2
from core.embedding.text2vec_base_chinese import Text2vecEmbedding
from core.vectordb.chroma import ChromaDB
from core.loader.simple_text import SimpleTextLoader

llm = ChatGLM2("http://localhost:8000")
embedding = Text2vecEmbedding()
vectordb = ChromaDB(llm = llm, embedding = embedding)
loader = SimpleTextLoader()

qa = QA(llm = llm, db = vectordb, loader = loader)

qa.load("./test.txt")
