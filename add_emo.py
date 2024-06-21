from transformers import AutoTokenizer, RobertaForSequenceClassification, AutoModelForSequenceClassification
import argparse
from typing  import Any, List
import numpy as np
from scipy.special import softmax
import csv
import nltk
from nltk.stem import WordNetLemmatizer
import json
import urllib.request
model_dirs = ["j-hartmann/emotion-english-distilroberta-base","SamLowe/roberta-base-go_emotions"]
args = {
    "model_dir":model_dirs[1]
}
def Penn2Wn(tag):
    if tag.startswith('J'):
        tag = 'a'
    elif tag.startswith('V'):
        tag = 'v'
    elif tag.startswith('R'):
        tag = 'r'
    elif tag.startswith('N'):
        tag = 'n'
    else:
        tag = 'n' 
    return tag

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

parse_utt_to_nltk = lambda example:{"utterance_postag":[nltk.pos_tag(example.split())][0]}
nltk_parse_to_wn_pos = lambda e: {"wn_pos":[get_wordnet_pos(w[1]) for w in e["utterance_postag"]]}
wnl = WordNetLemmatizer()
lemmatizer = lambda example:{"utterance_lemma":[wnl.lemmatize(w[0],Penn2Wn(w[1])) for w in example["utterance_postag"]]}

def get_lemma(utt):
    nltk_postags = parse_utt_to_nltk(utt)
    #wn_pos = nltk_parse_to_wn_pos(nltk_postags)
    lemmas = lemmatizer(nltk_postags)
    return lemmas
    
class EmoExtracter:
    def __init__(self, model_dir = model_dirs[1]) -> None:
        self.model_dir = model_dir
        self.load_model()
        self.load_dict()
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        model = RobertaForSequenceClassification.from_pretrained(self.model_dir)
        self.tokenizer = tokenizer
        self.model = model.cuda()
        self.label_2_id = model.config.label2id
        self.id_2_label = {y:x for x,y in self.label_2_id.items()}
    def load_dict(self):
        vad_dict = json.load(open("dataset/VAD/VAD.json","r+"))
        self.vad_dict = vad_dict
    def encode(self, corpus: List[str]):
        inputs = self.tokenizer(corpus, return_tensors  = "pt", padding=True, truncation = True)
        inputs = {k:v.to(self.model.device) for k,v in inputs.items()}
        outputs = self.model(**inputs).logits.detach().softmax(-1)
        pred = outputs.argmax(-1).tolist()
        pred = [self.id_2_label[x] for x in pred]
        outputs = outputs.tolist()
        return outputs, pred #[batch, n_emo]
    def get_intensity(self, corpus: List[str]):
        stem_seqs = [get_lemma(x)['utterance_lemma'] for x in corpus]
        scores = []
        for seq in stem_seqs:
            w_scores = []
            #print(w_scores)
            for w in seq:
                #print("w=",w)
                if w in self.vad_dict.keys():
                    w_score = self.vad_dict[w]
                    w_scores.append(w_score)
            if len(w_scores) == 0:
                scores.append([0.5,0.5,0.5])
            scores.append([np.mean([s[i] for s in w_scores]) for i in range(3)])
        return scores
            
    def __call__(self, corpus: List[str]):
        return self.encode(corpus)

class EmojiExtracter:
    def __init__(self, model_dir = model_dirs[1]) -> None:
        self.model_dir = model_dir
        self.load_model()
        self.load_label()
    def load_label(self):
        labels=[]
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]
        self.labels = labels
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.tokenizer = tokenizer
        self.model = model.cuda()
        self.label_2_id = model.config.label2id
        self.id_2_label = {y:x for x,y in self.label_2_id.items()}
    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    def encode(self, corpus: List[str]):
        inputs = self.tokenizer([self.preprocess(x) for x in corpus], return_tensors  = "pt", padding=True, truncation = True)
        inputs = {k:v.to(self.model.device) for k,v in inputs.items()}
        outputs = self.model(**inputs).logits.detach().softmax(-1)
        pred = outputs.argmax(-1).tolist()
        pred = [self.id_2_label[x] for x in pred]
        outputs = outputs.tolist()
        return outputs, pred #[batch, n_emo]
if __name__ == "__main__":
    args = argparse.Namespace(**args)
    extracter = EmoExtracter(model_dir = args.model_dir)
    inputs = ["What makes your job stressful for you?",
              "yea try to do something like that tomorrow and ah I love volley ball but yea that's hard now",
              "I am sorry to hear that. You should do that.",
            "But you offer them a better future than what they have currently. It may not be what they wanted, but it helps them in the long run.",
            "I've had to deal with collections before when I was in  bad financial condition. The person on the other line was really helpful though. She was understanding,"]
    ints = extracter.get_intensity(inputs)
    print(ints)
    #print(preds)
    #print(get_lemma("I loved you"))