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

qa.load("./data/doc/simple.txt")

questions = [
        "酒后驾驶会坐牢吗",
        "高速上的最高速度是多少？",
        "残障人士的机动轮椅车可以进入机动车道行驶吗",
        "流程参与者报错: cann't find participant",
    ]
for question in questions:
    print("=====")
    print(qa.ask(question))
    print("\n\n")

