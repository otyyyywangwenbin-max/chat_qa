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
                i += 1
                data={}
                data["id"] = path + "_" + str(i).zfill(6)
                data["text"] = line;
                datas.append(data)
        return datas