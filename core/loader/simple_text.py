#!/usr/bin/env python
# -*- coding:utf-8 _*-
from typing import List, Dict, Tuple, Union, Optional
from ..qa import *


class SimpleTextLoader(Loader):

    def load(self, path: str) -> List[Dict]:
        datas=[]
        i = 0
        with open(path) as file:
            for line in file:
                data={}
                data["id"] = path + "" + str(++i)
                data["text"] = line;
                datas.append(data)
        return datas