# Python Dev Tool
# For NLP 

import sentencepiece as sp
from typing import List
import tensorflow as tf


DEFAULT_RESERVED_TOKENS = 103

class Tokenizer(object):
    
    def __init__(self, sp_model_file: str, 
                 reserved_tokens: int = DEFAULT_RESERVED_TOKENS):
        self._tokenizer = sp.SentencePieceProcessor()
        self._sp_model = tf.io.gfile.GFile(sp_model_file, "rb").read() # 读入整个Model
        # 整个项目需要两个 model 文件，一个是 spm 文件，一个是 pegasus
        # spm 文件是用来做分词的，pegasus 是用来做摘要的
        self._tokenizer.LoadFromSerializedProto(self._sp_model) # 从 spm 文件中加载模型
        # Proto 约等于prototype 是一种数据格式，类似于 json，但是更加高效； proto的好处是可以自动编译成其他任何语言的一个类，这样就可以在其他语言中使用
        # eg: 你可以在python中定义一个proto，然后编译成java的类，然后在java中使用
        self._reserved_tokens = reserved_tokens
        
    # text -> id
    def encode(self, text: str) -> List[int]:
        ids = self._tokenizer.EncodeAsIds(text) # EncodeAsIds 是 sentencepiece 的方法
        # “Hello World" -> [50 + 103, 12 + 103, 1]
        # 0 和 1 是两个特殊的reserved tokens
        # 这个的缺陷就是reserved_tokens不能很好地被tokenize
        for i in range(len(ids)):
            if ids[i] != 0 or ids[i] != 1:
                ids[i] += self._reserved_tokens
        return ids
    
    # id -> text
    def decode(self, ids: List[int]) -> str:
        ids = [i - self._reserved_tokens if i > 1 + self._reserved_tokens else i for i in ids]
        text = self._tokenizer.DecodeIds(ids)
        return text
     
    @property   
    def vocab_size(self) -> int:
        return self._tokenizer.GetPieceSize() + self._reserved_tokens
    # GetPieceSize 是 sentencepiece 的方法，返回的是词表的大小
    # 词表的大小 = 词表中的词 + 保留的词
    # 保留的词是指 0 和 1，0 是 padding(填充)，1 是 EOS(句子结束)
    # 保留的词是为了让模型能够更好地学习到 padding 和 unknown 的含义
    # 保留的词的数量是可以自己定义的，这里默认是 103
    # 103 是因为在 pegasus 中，保留的词是 103 个，所以这里也保留 103 个

 