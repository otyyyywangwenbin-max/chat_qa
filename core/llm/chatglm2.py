#!/usr/bin/env python
# -*- coding:utf-8 _*-
from typing import List, Dict, Tuple, Union, Optional
import requests
import json
from ..qa import *


class ChatGLM2(LLM):
    api_url: str

    def __init__(self, api_url: str):
        self.api_url = api_url

    def generate_ask_prompt(self, question: str, texts: List[str]) -> str:
        prompt = f'根据文档内容来回答问题，问题是"{question}"，文档内容如下：\n'
        for text in texts:
            prompt += text + "\n"
        return prompt

    def request(self, prompt: str, histories: List[str] = []) -> str:
        resp = requests.post(self.api_url, json={
            'prompt': prompt,
            'history': histories
        })
        return resp.json()['response']


