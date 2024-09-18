import os
import pandas as pd
import tqdm
import numpy as np
import argparse
import random
import sys
import openai
import torch
import re


from langchain import OpenAI, ConversationChain
from langchain.prompts import PromptTemplate
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel


def format_data(data, preference):
    examples = [
        "例如，给定一个视频，它的\"标题\"为\"拒绝杨超越妹妹的撒娇，你会没有女朋友的！\"，\"asr\"为\"道。这不是你游远一些。我要是波浪那"
        "个人，我不要，我不要这种。\"，{}生成机器人推断出合理的\"{}\"为\"娱乐，情感分析，姐妹情，撒娇，社交互动\"。".format(preference, preference),
        "例如，给定一个视频，它的\"标题\"为\"为什么拉你的舌头，你自己不会想想吗\"，\"asr\"为\"乖乖乖乖\"，{}生成机器人推断出合理"
        "的\"{}\"为\"询问原因，自省，思考，自我意识，疑问\"。".format(preference, preference),
        "例如，给定一个视频，它的\"标题\"为\"中国最著名的四大神医你知道是谁吗？\"，\"asr\"为\"中国最著名的四大你都知道吗？四大名医，"
        "扁鹊，华佗膏！仲景，李时珍四大发明，造纸术，指南针，火药，活字印刷术。大古都，西安，南京，北京，洛阳，四大国粹中国，武术中国，医药中"
        "国，经济中国，书法。\"，{}生成机器人推断出合理的\"{}\"为\"知名医生，传统医学，中医，西医，医学历史\"。".format(preference, preference),
        "例如，给定一个视频，它的\"标题\"为\"终于解开了我多年的疑惑，原来新出厂的火车是这样上铁轨的\"，\"asr\"为\"None\"，{}生成机器"
        "人推断出合理的\"{}\"为\"汽车，交通，出行，新工厂\"。".format(preference, preference),
        "例如，给定一个视频，它的\"标题\"为\"这短短的一生，你不妨大胆一些\"，\"asr\"为\"这个。我们就这么多。你失去。一个月。\"，{}生成机"
        "器人推断出合理的\"{}\"为\"生活，感悟，励志，情感，人生\"。".format(preference, preference)
    ]
    sentences = []
    prompt = PromptTemplate(
        input_variables=["preference", "title", "asr", "example"],
        template="你是一个视频的{preference}生成机器人，根据输入的视频标题、ASR 推理出合理的\"{preference}\"。{example}那么，给定一"
                 "个新的视频，它的\"标题\"为\"{title}\"，\"ASR\"为\"{asr}\"，请推断出该视频的\"{preference}\"："
    )
    for ind, row in enumerate(data.iterrows()):
        example = examples[random.randint(0, 4)]
        title = str(row[1]['title'][:100]) if pd.notnull(row[1]['title']) else ""
        asr = str(row[1]['asr'][:100]) if pd.notnull(row[1]['asr']) else ""  # 处理NaN值
        text = prompt.format(
            preference=preference,
            title=title,
            asr=asr,
            example=example,
        )

        sentences.append(text)

    with open('./data/sentences.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(sentences))
    f.close()


def tag_gen(data_path, openai_key, gen_feq):
    openai.api_key = openai_key

    sentences = []
    with open(data_path, 'rb') as f:
        for line in f.readlines():
            sentences.append(line.decode('utf-8').strip())
    f.close()

    num = 0
    final_res = []
    history = []
    tokenizer = AutoTokenizer.from_pretrained("D:\chatglm2-6b-int4", trust_remote_code=True)
    model = AutoModel.from_pretrained("D:\chatglm2-6b-int4",trust_remote_code=True).cuda()
    model = model.eval()
    for sentence in tqdm.tqdm(sentences):
        if True:
            response, history = model.chat(tokenizer, sentence, history=[])
            messages = [{"role":"user","content":sentence},{"role":"model","content":response}]
            print(response)

            res = str(num) + "||"
            for j in range(gen_feq):
                ans = messages[-1]["content"]
                ans = ans.replace("\n", "")
                res += str(ans) + "||"

            final_res.append(res)

        num += 1
        if len(final_res) == 10:
            f = open("./data/tag_gen.txt", 'a', encoding='utf-8')
            f.write("\n".join(final_res))
            f.close()
            final_res = []


def posterior_process(data_path):
    f = open(data_path, 'r', encoding='utf-8')
    out = ""
    tag_all = []

    for line in f:
        # 手动替换标点符号和换行符
        line = line.replace(".", "").replace("。", "").replace(",", "、").replace("，", "、").replace("'", "").replace("\n",
                                                                                                                   "").replace(
            "\"", "")

        tmp = line.strip().split('||')
        out += str(tmp) + "\n"
        for t in tmp:
            if '、' in t:
                tags = t.split('、')
                tag_all += tags
    f.close()

    ans = Counter(tag_all)
    ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)

    tags = []
    for tmp in ans:
        if tmp[1] > 2:
            tags.append(tmp[0].replace(' ', ''))

    with open('./data/tags.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(tags))

    encoder = SentenceTransformer('hfl/chinese-roberta-wwm-ext-large')
    tags_embed = encoder.encode(tags)
    tags_dis = [np.sqrt(np.dot(_, _.T)) for _ in tags_embed]
    mark = [0 for _ in range(len(tags))]
    include = [[] for _ in range(len(tags))]

    for i in tqdm.trange(len(tags)):
        if mark[i] == 0:
            score = np.dot(tags_embed[i], tags_embed[i:].T)
            for j in range(i, len(tags)):
                if i != j:
                    score[j - i] = score[j - i] / (tags_dis[i] * tags_dis[j])
                    if score[j - i] > 0.95:
                        mark[j] = 1
                        include[i].append(tags[j])

    out = ""
    for i in range(len(tags)):
        if mark[i] == 0:
            out += tags[i] + "||" + str(include[i]) + "\n"

    f = open('./data/final_tags.csv', 'w')
    f.write(out)
    f.close()


class Data:
    def __init__(self, path):
        self.path = path
        self.dataframe = self.data_loader()

    def data_loader(self):
        df = pd.read_csv(self.path)
        df_f = df[['title', 'asr']]

        return df_f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="data path", default="")
    parser.add_argument("--tag_path", type=str, help="tag path", default="")
    parser.add_argument("--func", type=str, help="func", default="")
    parser.add_argument("--openai_key", type=str, help="openai key", default="")
    parser.add_argument("--gen_feq", type=int, help="gen_feq", default=5)

    paras = parser.parse_args()

    data_path = paras.data_path
    tag_path = paras.tag_path
    func = paras.func
    gen_feq = paras.gen_feq
    openai_key = paras.openai_key

    if func == "data_format":
        format_data(data=Data(path=data_path).dataframe, preference="兴趣标签")
        print("Data formatting completed")
    elif func == "tag_gen":
        tag_gen(data_path, openai_key, gen_feq)
        print("Tag generation completed")
    elif func == "posterior_process":
        posterior_process(data_path)
        print("Posterior processing completed")

if __name__ == "__main__":
    main()
