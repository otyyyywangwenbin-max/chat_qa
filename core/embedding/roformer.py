#!/usr/bin/env python
# -*- coding:utf-8 _*-
import numpy as np
import requests
import torch
from roformer import RoFormerTokenizer, RoFormerForCausalLM
from ..qa import *

class RoFormerEmbedding(Embedding):
    
    def __init__(self):
        pretrained_model = "junnyu/roformer_chinese_sim_char_small"
        self.tokenizer = RoFormerTokenizer.from_pretrained(pretrained_model)
        self.model = RoFormerForCausalLM.from_pretrained(pretrained_model)

    def encode(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=384)
        with torch.no_grad():
            outputs = self.model.forward(**inputs)
            ## 是不是可以换成非cpu?
            embedding = outputs.pooler_output.cpu().numpy()[0]
        return embedding


