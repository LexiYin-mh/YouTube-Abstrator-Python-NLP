import tensorflow as tf
import numpy as np
import tokenizer
import argparse

MODEL_INPUT_SIZE = 1024

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pegasus related argument parsing')

    parser.add_argument("--article", help = "path of the article abstract",
                    default= '../input_article')
    parser.add_argument("--model", help = "path of the pegasus model directory",
                    default= '../model/')
    parser.add_argument("--ckpt", help = "path of the ckpt model",
                    default= '../ckpt/c4.unigram.newline.10pct.96000.model')
    args = parser.parse_args()

    # parser 适合面向对象的argument管理，特别适合大型项目。
    # argparse 是python自带的一个命令行参数解析包，可以用来方便地读取命令行参数。
    # argparse模块还可以自动生成帮助和使用手册，并且可以自定义参数的类型和范围。


    # read the article
    text = open(args.article, 'r', encoding= "utf-8").read()

    # tokenize the article
    sp_tokenizer = tokenizer.Tokenizer(args.ckpt)
    ids = sp_tokenizer.encode(text)   # here the size of ids is 103 + len(text) (just 103 + the length of article)

    # Resize ids for inference feature input
    # 1. if the length of ids is less than 1024, then pad it with 0
    # 2. if the length of ids is more than 1024, then truncate it to 1024
    if len(ids) < MODEL_INPUT_SIZE:
        ids = ids + [0] * (MODEL_INPUT_SIZE - len(ids))
    else:
        ids = ids[:MODEL_INPUT_SIZE]
    ids = np.array(ids, dtype=np.int64)

    # Inference
    # 1. Load the model
    imported_model = tf.saved_model.load(args.model, tags='serve')

    # 2. Initialize TF template
    example = tf.train.Example()
    example.features.feature["inputs"].int64_list.value.extend(ids.astype(int))

    # 3. Generate output
    output = imported_model.signatures['serving_default'](
        examples = tf.constant([example.SerializeToString()]))

    # Detokenization
    # 1. convert the output to a list of ids
    output = output["outputs"].numpy().flatten().tolist()

    # 2. detokenize the output
    output = sp_tokenizer.decode(output)

    print(output)