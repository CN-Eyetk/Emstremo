from src.transformers import BlenderbotSmallTokenizer
from metric import NLGEval
import nltk
import numpy as np
import re
import torch
from metric.myMetrics import split_punct

import os
os.environ["HF_HOME"]="/disk/public_data/huggingface"
os.environ["HF_HUB_CACHE"] = "/disk/public_data/huggingface/hub"
def read_text(path):
    text = json.load(open(path, "r+"))
    return text

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    # if len(preds) == 0:
    labels = [label.strip() for label in labels]
    return preds, labels


class NLTK_Metric:
    def __init__(self, hyp_path, ref_path):
        self.refs = []
        self.hyps = []
        with open(hyp_path, 'r', encoding='utf-8') as f:
            hyps = json.load(f)
        with open(ref_path, 'r', encoding='utf-8') as f:
            refs = json.load(f)
        assert len(hyps) == len(refs)
        self.res = []
        refs, hyps = postprocess_text(refs, hyps)
        self.forword(hyps, refs)
        
        
    def forword(self, decoder_preds, decoder_labels, no_glove=False):
        ref_list = []
        hyp_list = []
        for ref, hyp in zip(decoder_labels, decoder_preds):
            #print("ref",ref)
            ref = ' '.join(nltk.word_tokenize(split_punct(ref).lower()))
            hyp = ' '.join(nltk.word_tokenize(split_punct(hyp).lower()))
            if len(hyp) == 0:
                hyp = '&'
            ref_list.append(ref)
            hyp_list.append(hyp)
        from metric import NLGEval
        metric = NLGEval(no_glove=no_glove)
        metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list, )
        #metric_res_list = {k:np.mean(v) for k,v in metric_res_list.items()}
        #print(metric_res_list)
        self.res = metric_res
additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
tokenizer.add_tokens(additional_special_tokens)
comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]
tokenizer.add_tokens(comet_additional_special_tokens)
tokenizer.add_special_tokens({'cls_token': '[CLS]'})

#bertscore = load("bertscore")

emb_type = 'other'
emb_path = '/mnt/HD-8T/lijunlin/metric/word2vec/glove.6B.300d.model.bin'


from metric.myMetrics import Metric
import pandas as pd
import json
import os

dirs = [
    "our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-ct0.2am610/",
    "our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-w_eosstg-w_emocat-w_stgcat-ct0.2pm301/"
    ]

all_res = {}

for dir in dirs:
    print(dir)
    hyp_path = f"{dir}/hyp_strategy.json"
    ref_path = f"{dir}/ref_strategy.json"
    summary_path = f"{dir}/summary.txt"
    with open(hyp_path, 'r', encoding='utf-8') as f:
        hyps = json.load(f)
    with open(ref_path, 'r', encoding='utf-8') as f:
        refs = json.load(f)

    metric = Metric(toker=tokenizer, hyp_path=hyp_path, ref_path=ref_path, use_nltk=True)
    #metric_2 = NLTK_Metric( hyp_path=hyp_path, ref_path=ref_path)

    result, result_list = metric.close()
    #result_2 = metric_2.res

    print(result)
    #print(result_2)

    print("="*100)

    #result.update(result_2)

    all_res[dir.replace("our_generated_data","")] = {k:round(v,3) for k,v in result.items()}

df = pd.DataFrame(all_res)
df.to_csv("res.csv")