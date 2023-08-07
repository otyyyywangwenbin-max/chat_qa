#!/usr/bin/env python
# -*- coding:utf-8 _*-
import numpy as np
import requests
import torch
from transformers import BertTokenizer, BertModel
from ..qa import *

class Text2vecEmbedding(Embedding):
    
    def __init__(self):
        pretrained_model = "shibing624/text2vec-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertModel.from_pretrained(pretrained_model)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, text: str) -> np.ndarray:
        input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**input)
        embeddings = self.mean_pooling(model_output, input['attention_mask'])
        return embeddings.numpy()[0]


