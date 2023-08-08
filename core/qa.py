#!/usr/bin/env python
# -*- coding:utf-8 _*-
from typing import List, Dict, Tuple, Union, Optional
import numpy as np


class Embedding():
    def encode(self, text: str) -> List[float]:
        ...


class VectorDB():
    def add(self, datas: List[Dict]):
        ...

    def query(self, query: str, topn: int = 2) -> List[str]:
        ...


class LLM():
    def generate_ask_prompt(self, question: str, texts: List[str]) -> str:
        ...

    def request(self, prompt: str, history: List[str] = []) -> str:
        ...


class Loader():
    def load(self, path: str) -> List[Dict]:
        ...


class QA:
    """
    其实不用定义那么多class, 直接写在一个文件更简单, 更清楚, 
    最后肯定也只会用一种 "Embedding + VectorDB + LLM" 的组合
    """ 
    db: VectorDB
    llm: LLM
    loader: Loader

    def __init__(self, db: VectorDB, llm: LLM, loader: Loader):
        self.db = db
        self.llm = llm
        self.loader = loader

    def load(self, path: str):
        """
        加载指定路径的知识文档
        """ 
        datas = self.loader.load(path)
        self.db.add(datas)

    def ask(self, question: str) -> str:
        chunks = self.db.query(question, 2)
        prompt = self.llm.generate_ask_prompt(question, chunks)
        return self.llm.request(prompt)



