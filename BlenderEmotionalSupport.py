#hide
# Imports

"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import os
import sys
import copy
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
from metric.myMetrics import Metric
import glob
import logging
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import time
from pathlib import Path
import json
import wandb
#from attach_vad.VADTokenizer import W2VAD
from cleantext import clean
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
    
from src.transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    #AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from src.transformers import (BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, BlenderbotSmallConfig)
from src.transformers import (BartForConditionalGeneration, BartTokenizer, BartConfig)
#from utils.data_parallel import BalancedDataParallel
from add_emo import EmoExtracter
model_dirs = ["j-hartmann/emotion-english-distilroberta-base","SamLowe/roberta-base-go_emotions"]
emo_extracter = EmoExtracter(model_dir=model_dirs[1])



# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# Args to allow for easy convertion of python script to notebook

def load_dataset(args, tokenizer):
    with open(args.data_path+"/"+ args.train_comet_file, "r", encoding="utf-8") as f:
        comet_trn = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_train_comet_file, "r", encoding="utf-8") as f:
        st_comet_trn = f.read().split("\n")
    with open(args.data_path+"/"+ args.train_file_name, "r", encoding="utf-8") as f:
        df_trn = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_train_file, "r", encoding="utf-8") as f:
        st_trn = f.read().split("\n")

    with open(args.data_path+"/"+ args.eval_comet_file, "r", encoding="utf-8") as f:
        comet_val = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_eval_comet_file, "r", encoding="utf-8") as f:
        st_comet_val = f.read().split("\n")
    with open(args.data_path+"/" + args.eval_file_name, "r", encoding="utf-8") as f:
        df_val = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_eval_file, "r", encoding="utf-8") as f:
        st_val = f.read().split("\n")

    with open(args.data_path+"/"+ args.test_comet_file, "r", encoding="utf-8") as f:
        comet_test = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_test_comet_file, "r", encoding="utf-8") as f:
        st_comet_test = f.read().split("\n")
    with open(args.data_path+"/" + args.test_file_name, "r", encoding="utf-8") as f:
        df_test = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_test_file, "r", encoding="utf-8") as f:
        st_test = f.read().split("\n")
    train_dataset = load_and_cache_examples(args, tokenizer, df_trn, comet_trn, st_comet_trn, evaluate=False, strategy=args.strategy, situations = st_trn)
    eval_dataset = load_and_cache_examples(args, tokenizer, df_val, comet_val, st_comet_val, evaluate=True, strategy=args.strategy, test=False, situations = st_val)
    test_dataset = load_and_cache_examples(args, tokenizer, df_test, comet_test, st_comet_test, evaluate=True, strategy=args.strategy, test=True, situations = st_test)
    return train_dataset, eval_dataset, test_dataset

def save_checkpoint(args, model, tokenizer, output_dir, checkpoint_prefix, optimizer, scheduler):
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    _rotate_checkpoints(args, checkpoint_prefix)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)

def check(example):
    input_ids = example.input_ids
    #print("input_ids",input_ids)
    
    spt_eos_indice = example.spt_eos_indice
    #print("spt_eos_indice",spt_eos_indice)
    assert len(spt_eos_indice) == len(example.strat_seq)
    for e,i in enumerate(input_ids):
        #print(f"{e}-{i}")
        
        if e in spt_eos_indice:
            #try:
            #print(f"!{e}")
            if e > 0:
                assert input_ids[e-1] in [2,  54961]
            assert input_ids[e+1] not in [0]
            #except:
                #print(f"warning i = {i} wehre e = {e}")
                #print(input_ids)

        
def load_config(args, eval = False):
    
    if not eval:
        if args.use_bart:
            config = BartConfig.from_json_file("config/bart/config.json")
        else:
            config = BlenderbotSmallConfig.from_pretrained("config/blenderbot/config.json")
    else:
        if args.use_bart:
            config = BartConfig.from_pretrained(args.output_dir)
        else:
            config = BlenderbotSmallConfig.from_pretrained(args.output_dir)
    print("config = ", config)
    
    return config

def load_model_for_eval(args):
    config = load_config(args, eval = True)
    #if args.pretrained_model_path is not None:
    #    if args.use_bart:
    #         model = BartForConditionalGeneration.from_pretrained(args.pretrained_model_path
    print(f"****loading {args.output_dir} for evaluation ***")
    if args.use_bart:
        model = BartForConditionalGeneration.from_pretrained(args.output_dir, from_tf=False, config = config)
    else:
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.output_dir, from_tf=False, config = config)
    
    return model

def load_model(args, tokenizer):
    config = load_config(args)
    if args.use_bart:
        model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config = config)
    else:
        print("loading blenderbot")
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_or_path, config = config, cache_dir=args.model_cache_dir)
    print(model.config)
    #model = BlenderbotSmallForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(args.device)
    return model

def load_tokenizer(args):
    if args.use_bart:
        config = BartConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    else:
        config = BlenderbotSmallConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
        tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    print(f"vocab size = {tokenizer.vocab_size}")
    if args.data_path == "converted_dataset":
        additional_special_tokens = ["[Question]","[Reflection of feelings]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions or Information]","[Greeting]"]
    else:
        additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
    comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]

    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_tokens(comet_additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

    role_special_tokens = ["[SEK]","[SPT]"]
    tokenizer.add_tokens(role_special_tokens)
    #if args.prepend_emotion:
    #    emotion_special_tokens = [f"[{label}]" for label in emo_extracter.label_2_id.keys()]
    #    print(emotion_special_tokens)
    #    tokenizer.add_tokens(emotion_special_tokens)

    return config, tokenizer

def load_emo_emb(model, tokenizer, emo_extracter):
    labels = emo_extracter.label_2_id.keys()
    vecs = torch.zeros((len(labels), model.config.d_model))
    with torch.no_grad():
        for i, label in enumerate(labels):
            inputs = tokenizer(label, return_tensors = "pt")
            hidden = model(**{k:v.to(model.device) for k,v in inputs.items()})[0][:,0,:]
            vecs[i] = hidden
    return vecs

def load_stra_emb(model, tokenizer, strat_labels):
    vecs = torch.zeros((len(strat_labels), model.config.d_model))
    with torch.no_grad():
        for i, label in enumerate(strat_labels):
            inputs = tokenizer(label, return_tensors = "pt")
            hidden = model(**{k:v.to(model.device) for k,v in inputs.items()})[0][:,0,:]
            vecs[i] = hidden
    return vecs

class InputFeatures_train(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                 role_ids, lm_labels, cls_position, cls_label, strategy_ids, situ_ids, is_strat_targ = None, is_emo_targ = None, spt_eos_indice = None, input_len=None, vad_ids = None):
        self.conv_id = conv_id
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.role_ids = role_ids
        self.lm_labels = lm_labels
        self.cls_position = cls_position
        self.cls_label = cls_label
        self.strategy_ids = strategy_ids
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len
        self.situ_ids = situ_ids
        self.is_strat_targ = is_strat_targ
        self.is_emo_targ = is_emo_targ
        self.spt_eos_indice = spt_eos_indice
        self.vad_ids = vad_ids
        #self.hist_strat_labels = hist_strat_labels
        


class InputFeatures_blender(object):
    def __init__(self, encoder_feature, decoder_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask, emo_dist = None, emo_in_dist = None, intensity = None):
        self.conv_id = encoder_feature.conv_id
        self.input_ids = encoder_feature.input_ids
        self.vad_ids = encoder_feature.vad_ids
        self.position_ids = encoder_feature.position_ids
        self.token_type_ids = encoder_feature.token_type_ids
        self.role_ids = encoder_feature.role_ids
        self.lm_labels = encoder_feature.lm_labels
        self.cls_position = encoder_feature.cls_position
        self.cls_label = encoder_feature.cls_label
        self.strategy_ids = encoder_feature.strategy_ids
        self.situ_ids = encoder_feature.situ_ids
        self.decoder_input_ids = decoder_feature.input_ids
        self.decoder_position_ids = decoder_feature.position_ids
        self.decoder_token_type_ids = decoder_feature.token_type_ids
        self.decoder_role_ids = decoder_feature.role_ids
        self.decoder_lm_labels = decoder_feature.lm_labels
        self.decoder_cls_position = decoder_feature.cls_position
        self.decoder_cls_label = decoder_feature.cls_label
        self.decoder_strategy_ids = decoder_feature.strategy_ids
        self.comet_ids = comet_ids
        self.comet_mask = comet_mask
        self.emotion = emotion
        self.comet_st_ids = comet_st_ids
        self.comet_st_mask = comet_st_mask
        self.emo_dist = emo_dist
        self.emo_in_dist = emo_in_dist
        #self.spt_eos_indice = encoder_feature.spt_eos_indice
        #self.strat_seq = encoder_feature.hist_strat_labels[:-1] + [self.decoder_strategy_ids[0]]
        self.is_strat_targ = encoder_feature.is_strat_targ
        self.is_emo_targ = encoder_feature.is_emo_targ
        self.intensity = intensity
        #self.vad_ids = vad_ids
        #self.decoder_vad_ids = decoder_vad_ids
#        assert len(self.spt_eos_indice) == len(self.strat_seq)
        #print("strat_seq",self.strat_seq)
        #print("spt_eos_indice",self.spt_eos_indice)
        #self.check_feat()
        #print("input_ids in feature",self.input_ids)
    def check_feat(self):
        check(self)

def process_row_to_comet_query(row):
    sents = row.strip().split('EOS')
    n_sent = len(sents)
    all_seeker_uttrs = []
    for i in range(n_sent-1, -1, -1):
        # print(sents[i].strip().split(' '))
        tokens = sents[i].strip().split(' ')
        if int(tokens[1]) == 0:
            if int(tokens[1]) == 0:
                return ' '.join(tokens[3:])
                # all_seeker_uttrs.append(' '.join(tokens[3:]))
    # return '\t'.join(all_seeker_uttrs)


def summary(test_file_path, generate_file_path, reference_file_path, summary_file_path, all_top_k_blocks, all_top_k_blocks_st, chat_texts, test_situation_file_path):
    with open(test_file_path, "r", encoding="utf-8") as f:
        ctx = f.read().split("\n")
    with open(test_situation_file_path, "r", encoding="utf-8") as f:
        st = f.read().split("\n")
    ctx = ctx[:-1]
    st = st[:-1]
    with open(generate_file_path, "r", encoding="utf-8") as f:
        gen_rep = json.load(f)
    with open(reference_file_path, "r", encoding="utf-8") as f:
        ref_rep = json.load(f)
    if all_top_k_blocks is not None:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            for (ctx_row, ref_rep_row, gen_rep_row, top_k_blocks, top_k_blocks_st, chat_text, st_row) in zip(ctx, ref_rep, gen_rep, all_top_k_blocks, all_top_k_blocks_st, chat_texts, st):
                query = process_row_to_comet_query(chat_text)
                if query is None:
                    query = ""
                line = '[contxt]\t' + ctx_row + '\n[reference_response]\t' + ref_rep_row + '\n[hypothesis_response]\t' + gen_rep_row + '\n[comet query]\t' + query + '\n[comet blocks (attention top5)]\t' + '  '.join(top_k_blocks) +'\n[situation]\t' + st_row + '\n[situation comet blocks (attention top5)]\t' + '  '.join(top_k_blocks_st) + '\n' * 2
                f.writelines(line)
    else:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            for (ctx_row, ref_rep_row, gen_rep_row, chat_text, st_row) in zip(ctx, ref_rep, gen_rep, chat_texts, st):
                query = process_row_to_comet_query(chat_text)
                if query is None:
                    query = ""
                line = '[contxt]\t' + ctx_row + '\n[reference_response]\t' + ref_rep_row + '\n[hypothesis_response]\t' + gen_rep_row + '\n[comet query]\t' + query + '\n[comet blocks (attention top5)]\t'  +'\n[situation]\t' + st_row + '\n[situation comet blocks (attention top5)]\t' + '\n' * 2
                f.writelines(line)

def extract_top_k_attention_comet_block(mutual_attentions, comet_rows, k):
    all_top_k_blocks = []
    num_block = len(mutual_attentions[0])
    for mutual_attention, comet_row in zip(mutual_attentions, comet_rows):
        comet_blocks = comet_row.split('__EOS__')[:-1]
        if len(comet_blocks) < num_block:
            comet_blocks += (['[PAD]'] * (num_block - len(comet_blocks)))
        index = torch.topk(mutual_attention, k).indices
        top_k_blocks = [comet_blocks[i] for i in index.numpy().tolist()]
        all_top_k_blocks.append(top_k_blocks)
    return all_top_k_blocks

def _get_comet_input(args, comet_row, tokenizer, max_num_attr=30, max_len_attr=10):
    attrs = comet_row.split('__EOS__')[:-1]
    comet_ids = []
    comet_mask = [] #对每一个comet attr + tail 的mask
    for ids, attr in enumerate(attrs):
        if ids == max_num_attr:
            break
        if args.use_bart:
           comet_attr_ids = [tokenizer.bos_token_id] + tokenizer.encode(attr)[1:]
        else:
            comet_attr_ids = tokenizer.encode(attr)
        if len(comet_attr_ids) < max_len_attr:
            comet_attr_ids += [tokenizer.pad_token_id]*(max_len_attr - len(comet_attr_ids)) 
        else:
            comet_attr_ids = comet_attr_ids[:max_len_attr]
        comet_ids.append(comet_attr_ids)
        comet_mask.append(1)

    if len(comet_ids) < max_num_attr:
        comet_ids += ([[tokenizer.pad_token_id]*max_len_attr]) * (max_num_attr - len(comet_ids))
        comet_mask += [0] * (max_num_attr - len(comet_mask))
    # print(attrs) 
    # print(comet_ids)
    # print(comet_mask)
    # print(error)
    
    assert len(comet_ids) == max_num_attr
    assert len(comet_mask) == max_num_attr
    return comet_ids, comet_mask


def _make_feature(args, id_, sents, rls, ts, eos, pad_token_id, pad=False, block_size=512, strategy_labels=None, situ_ids = None, evaluate=False, str_embd=False, generation=False, vad_ids = None):
    # we did't use role label and turn number in modeling as they did't carry significant improvement. However, codes still remain here.
    #print("strategy_labels",strategy_labels)
    if len(sents) == 0:
        return InputFeatures_train([], [], [], [], [],
                            [], [] , [], [])
    input_ids = [i for s in sents for i in s+[eos]] #添加eos
    if 1 == 2:
        vad_ids = [i for s in vad_ids for i in s+[-1]]
        assert len(input_ids) == len(vad_ids)


    input_ids = input_ids
    #print("input_ids 323",input_ids)
    lm_labels = []
    token_type_ids = []
    roles = []
    strategy_ids = []
    is_strat_targ = []
    is_emo_targ = []


    for i, s in enumerate(sents):
        token_type_ids += [ts[i]] * (len(s) + 1)
        flag_str = -1
        if str_embd: #use for strategy embed but currently we treat strategy as token
            strategy_ids += [strategy_labels[-1]] * (len(s) + 1)
        else:
            strategy_ids += [8] * (len(s) + 1)
        if i < len(sents) - 1:
            lm_labels += [-100] * (len(s) + 1)
            roles += [rls[i]] * (len(s) + 1)
        else:
            lm_labels += (  s + [eos])
            roles += [rls[i]] * (len(s) + 1)

            cur_rl = rls[i]     
        if rls[i] == 1:
            is_strat_targ +=  [0] * (len(s)) + [1]#attent to <eos> of seeker utternaces
            is_emo_targ += [0] * (len(s) + 1)
            #strat_step += 1
        else:
            is_strat_targ +=  [0] * (len(s) + 1)
            is_emo_targ +=  [0] * (len(s)) + [1]


    i = len(lm_labels) - 1
    if len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)
    while i >= 0:
        if lm_labels[i] != -100:
            break
        i -= 1
    input_ids = input_ids[:i+1]
    if 1 == 2:
        vad_ids = vad_ids[:i+1]
        assert len(input_ids) == len(vad_ids)
    
    #vad_ids = vad_ids[:i+1]
    lm_labels = lm_labels[:i+1]
    token_type_ids = token_type_ids[:i+1]
    roles = roles[:i+1]
    is_strat_targ = is_strat_targ[:i+1]
    is_emo_targ = is_emo_targ[:i+1]
    #print("input_ids 395",input_ids)
    if not str_embd:
        strategy_ids = [8]*len(input_ids) # strategy is not used
    else:
        strategy_ids = strategy_ids[:i+1]
    if len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)


    assert (len(input_ids) == len(token_type_ids)
            == len(lm_labels) == len(roles) == len(strategy_ids) == len(is_strat_targ) == len(is_emo_targ))
    # cut according to block size
    if len(input_ids) > block_size:
        cut_index = input_ids.index(eos,-512) + 1
        input_ids = input_ids[cut_index: ]
        if 1 == 2:
            vad_ids = vad_ids[cut_index:]
        token_type_ids = token_type_ids[cut_index: ]
        lm_labels = lm_labels[cut_index: ]
        roles = roles[cut_index: ]
        strategy_ids = strategy_ids[cut_index: ]
        is_strat_targ = is_strat_targ[cut_index: ]
        is_emo_targ = is_emo_targ[cut_index: ]
        
    # pad to multiples of 8
    if pad:
        while len(input_ids) % 8 != 0:
            input_ids.append(pad_token_id)
            if 1 == 2:
                vad_ids.append(-1)
            token_type_ids.append(0)
            lm_labels.append(-100)
            if args.use_bart:
                roles.append(-1)
            else:
                roles.append(0)
            strategy_ids.append(8)
            is_strat_targ.append(0)
            is_emo_targ.append(0)
        assert len(input_ids) % 8 == 0
    position_ids = list(range(len(input_ids)))
    assert (len(input_ids) == len(position_ids) == len(token_type_ids)
            == len(lm_labels) == len(roles) == len(strategy_ids) == len(is_strat_targ) == len(is_emo_targ))
    if len(input_ids) == 0:
        import pdb
        pdb.set_trace()
    elif len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)
    if True:
        # if it is for generation, the last sentence of context is the last sentence
        cls_position = len(input_ids)-1-input_ids[::-1].index(eos)
    else:
        # if not, the last sentence of context is the second last sentence
        cls_position = len(input_ids)-1-input_ids[::-1].index(eos,input_ids[::-1].index(eos)+1)
    if evaluate and strategy_labels[-1]!=8:
        try:
            lm_labels[lm_labels.index(strategy_labels[-1]+args.base_vocab_size)] = -100
        except Exception:
            pass
    #print("input_ids:",input_ids)
    #print("situation:",situation)
    spt_eos_indice = [i for i, e in enumerate(is_strat_targ) if e != 8]
    #hist_strat_labels = [e for i,e in enumerate(is_strat_targ) if e != 8]
    #print("strategy_ids",strategy_ids)
    feature = InputFeatures_train(id_, input_ids, position_ids, token_type_ids, roles,
                            lm_labels, cls_position , strategy_labels[-1], strategy_ids, situ_ids, is_strat_targ = is_strat_targ, is_emo_targ = is_emo_targ,
                            spt_eos_indice = spt_eos_indice,
                            vad_ids = vad_ids
                            #hist_strat_labels = hist_strat_labels,
                            )
    return feature

def _norm_text(text):
    text = text.replace("’","'")
    #text = text.replace("é","e")
    #text = text.replace("has ime for me","has time for me")
    #text = clean(text, no_emoji=True)
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    emo, r, t, *toks = text.strip().split()
    try:
        emo = int(emo)
        r = int(r)
        t = int(t)
        toks = ' '.join(toks[:len(toks)])
    except Exception as e:
        raise e
    #toks = re.compile(r"(?<=\w)(?=[^\s\w])").sub(" ", toks) #3 March, When did I add this 
    return emo, r, t, toks


def _get_inputs_from_text(text, tokenizer, strategy=True, cls = False, get_emo_dist = False, get_emo_in_dist = False, prepend_emotion = False, args = None, is_decoder = False, vad_tokenizer = None): #bart 
    srcs = text.strip()
    inputs = []
    roles = []
    turns = []
    vad_ids = []
    strategy_labels=[]
    srcs = srcs.split(" EOS")
    emotion = None
    
    for idx, src in enumerate(srcs):


        if src =="":
            continue
        src_emo, src_role, src_turn, src = _norm_text(src)
        if emotion is None:
            emotion = src_emo
        if args.use_bart:
            context_id = [tokenizer.bos_token_id] + tokenizer.encode(src)[1:-1]
            is_fast_tokenizer = False
            char_to_remove = "Ġ"
        else:
            context_id = tokenizer.encode(src)
            is_fast_tokenizer = False
            char_to_remove = "@"

        if not strategy:
            context_id = [i  for i in context_id if i< args.base_vocab_size]
        elif cls:
            context_id = tokenizer.cls + [i for i in context_id if i< args.base_vocab_size]
        else:
            pass

        
        if src_role==1: #src_role = 1 -> supporter
            try:
                label = "["+src.split("[")[1].split("]")[0]+"]"
                #print("label", label)
                
            except Exception as e:
                strategy_labels.append(8)
            else:
                strategy_label = tokenizer.convert_tokens_to_ids([label])[0] - args.base_vocab_size
                strategy_labels.append(strategy_label)
        else:
            strategy_labels.append(8)

        
        vad_ids = None
            #assert len(context_id) == len(context_id_new)
            #vad_ids.append(token_vads)
            #else:
            #    vad_ids = None
        inputs.append(context_id)
        roles.append(src_role)

        turns.append(src_turn)
    clean_src = re.compile("^\[[a-zA-Z- ]+\]\s").sub("",src)
    if get_emo_dist:
        #clean_src = re.compile("^\[[a-zA-Z- ]+\]\s").sub("",src)
        emo_dist, pred = emo_extracter(clean_src)
    else:
        emo_dist = None
    if get_emo_in_dist:
        #clean_src = re.compile("^\[[a-zA-Z- ]+\]\s").sub("",src)
        emo_in_dist, pred = emo_extracter(clean_src)
    else:
        emo_in_dist = None
    intensity = emo_extracter.get_intensity([clean_src])[0]
    return inputs, roles, turns, strategy_labels, emotion, emo_dist, emo_in_dist, intensity, vad_ids

def construct_conv_ESD(args, idx, row, comet_row, comet_st_row, tokenizer, eos = True, pad=True, cls=False, evaluate=False, strategy=True, generation=False, situation = None, prepend_emotion = False, use_emo_in_dist = False, vad_tokenizer = None):

    #  process input text
    #print("row",row)
    
    inputs, roles, turns, strategy_labels, _, _, emo_in_dist, _, vad_ids= _get_inputs_from_text("EOS".join(row.split("EOS")[:-1]), tokenizer, strategy=strategy, get_emo_dist = False, get_emo_in_dist = (True if use_emo_in_dist else False),
                                                                                     args = args,
                                                                                     vad_tokenizer = None 
                                                                                     )
    # process output (decoder input) text
    d_inputs, d_roles, d_turns, d_strategy_labels, emotion, emo_dist, _, intensity, d_vad_ids = _get_inputs_from_text(row.split("EOS")[-1], tokenizer, strategy=strategy, get_emo_dist = True, prepend_emotion =  prepend_emotion, args = args, is_decoder = True,
                                                                                                                      vad_tokenizer = None 
                                                                                                                      )
    #print("d_strategy_labels",d_strategy_labels)
    situ_ids = tokenizer.encode(situation) 
    if situ_ids[-1] !=  tokenizer.eos_token_id:
        situ_ids.append(tokenizer.eos_token_id)
    feature = _make_feature(args, idx, inputs, roles, turns, tokenizer.eos_token_id, tokenizer.pad_token_id, pad=pad, strategy_labels=strategy_labels, situ_ids = situ_ids, evaluate=evaluate, str_embd=True, generation=generation, vad_ids = vad_ids)
    # make feature for output (decoder input) text
    d_feature = _make_feature(args, idx, d_inputs, d_roles, d_turns, tokenizer.eos_token_id, tokenizer.pad_token_id, pad=pad, strategy_labels=d_strategy_labels, situ_ids = situ_ids, evaluate=evaluate, str_embd=True, generation=generation, vad_ids = d_vad_ids)
    
    comet_ids, comet_mask = _get_comet_input(args, comet_row, tokenizer)
    comet_st_ids, comet_st_mask = _get_comet_input(args, comet_st_row, tokenizer, max_num_attr=20)
    feature = InputFeatures_blender(feature, d_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask, emo_dist = emo_dist, emo_in_dist = emo_in_dist, intensity = intensity)
    #print("feature",feature.decoder_strategy_ids)
    return feature


    

class ESDDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, comet, comet_st, block_size=512, evaluate=False, strategy=True, test=False, situations = None, vad_tokenizer = None):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        self.tokenizer = tokenizer
        self.vad_tokenizer = vad_tokenizer
        self.collate_verbose_step = 10
        directory = args.data_cache_dir
        self.args = args
        self.role_id_2_token_id = [tokenizer.convert_tokens_to_ids("[SEK]"),tokenizer.convert_tokens_to_ids("[SPT]")]
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if evaluate:
            if not test:
                cached_features_file = os.path.join(
                    directory, 'val_' + args.model_type + "_cached_lm_" + str(block_size)
                )
            else:
                cached_features_file = os.path.join(
                    directory, 'test_' + args.model_type + "_cached_lm_" + str(block_size)
                )
        else:
            cached_features_file = os.path.join(
                directory, 'trn_' + args.model_type + "_cached_lm_" + str(block_size)
            )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.features = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            print(len(df) , len(comet), len(comet_st))
            assert len(df) == len(comet) == len(comet_st)
            if situations is not None:
                assert len(comet_st) == len(situations)
            self.features = []

            for idx, (row, comet_row, comet_st_row) in tqdm(enumerate(zip(df[:-1], comet[:-1], comet_st[:-1])), total = len(df[:-1])):

                
                if "EOS" not in row:
                    continue
                conv = construct_conv_ESD(args, idx, row, comet_row, comet_st_row, tokenizer, cls=False, strategy=strategy ,evaluate=evaluate, situation = situations[idx], prepend_emotion = False, use_emo_in_dist = False, vad_tokenizer = None)
                #conv.vad_ids = [vad_id if not vad_id == tokenizer.unk_token_id else tokenizer.vocab["[0v0a0d]"] for vad_id in conv.vad_ids]
                #print("conv.vad_ids",conv.vad_ids)
                if len(conv.input_ids) >= block_size:
                    conv.input_ids = conv.input_ids[-block_size:]
                    conv.role_ids = conv.role_ids[-block_size:] #leave the 0-th id to pad in collate function
                    conv.is_strat_targ = conv.is_strat_targ[-block_size:]
                    conv.is_emo_targ = conv.is_emo_targ[-block_size:]
                    conv.input_ids[0] = tokenizer.encode(tokenizer.cls_token, add_special_tokens = False)[0]
                    conv.is_strat_targ[0] = 1
                    conv.is_emo_targ[0] = 1
                else:
                    conv.input_ids = tokenizer.encode(tokenizer.cls_token, add_special_tokens = False) + conv.input_ids
                    conv.is_strat_targ = [1] + conv.is_strat_targ 
                    conv.is_emo_targ = [1] + conv.is_emo_targ
                self.features.append(conv)

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Finished~")


    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    #@staticmethod
    def collate(self,features):
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long)
                                for f in features],
                                batch_first=True, padding_value=self.tokenizer.pad_token_id)

        position_ids = pad_sequence([torch.tensor(f.position_ids,
                                                dtype=torch.long)
                                    for f in features],
                                    batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids,
                                                    dtype=torch.long)
                                    for f in features],
                                    batch_first=True, padding_value=0)
        role_ids = pad_sequence([torch.tensor([self.tokenizer.pad_token_id] + [self.role_id_2_token_id[i]  if i > -1 else self.tokenizer.pad_token_id for i in f.role_ids], 
                                            dtype=torch.long)
                                    for f in features],
                                    batch_first=True, padding_value=self.tokenizer.pad_token_id)

        role_ids = role_ids[:,:input_ids.size(1)]

        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=-100)
        if features[0].situ_ids is not None:
            situations = pad_sequence([torch.tensor(f.situ_ids, dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=self.tokenizer.pad_token_id)
        else:
            situations = None
        cls_positions = torch.tensor([f.cls_position for f in features], dtype=torch.long)
        
        cls_labels = torch.tensor([f.cls_label for f in features], dtype=torch.long)
        
        strategy_ids = pad_sequence([torch.tensor(f.strategy_ids, dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=8)

        decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long)
                                for f in features],
                                batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_position_ids = pad_sequence([torch.tensor(f.decoder_position_ids,
                                                dtype=torch.long)
                                    for f in features],
                                    batch_first=True, padding_value=0)
        decoder_token_type_ids = pad_sequence([torch.tensor(f.decoder_token_type_ids,
                                                    dtype=torch.long)
                                    for f in features],
                                    batch_first=True, padding_value=0)
        decoder_role_ids = pad_sequence([torch.tensor(f.decoder_role_ids,
                                            dtype=torch.long)
                                    for f in features],
                                    batch_first=True, padding_value=0)
        decoder_labels = pad_sequence([torch.tensor(f.decoder_lm_labels, dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=-100)

        decoder_cls_positions = torch.tensor([f.decoder_cls_position for f in features], dtype=torch.long)

        decoder_cls_labels = torch.tensor([f.decoder_cls_label for f in features], dtype=torch.long)

        decoder_strategy_ids = pad_sequence([torch.tensor(f.decoder_strategy_ids, dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=8)

        # print([f.comet_ids for f in features])
        # print([f.comet_mask for f in features])
        comet_ids = torch.tensor([f.comet_ids for f in features], dtype=torch.long)
        comet_mask = torch.tensor([f.comet_mask for f in features], dtype=torch.long)
        emotion = torch.tensor([f.emotion for f in features], dtype=torch.long)
        comet_st_ids = torch.tensor([f.comet_st_ids for f in features], dtype=torch.long)
        comet_st_mask = torch.tensor([f.comet_st_mask for f in features], dtype=torch.long)
        if features[0].emo_dist is not None:
            emo_dist = torch.tensor([f.emo_dist for f in features], dtype=torch.float64).squeeze(1)
        else:
            emo_dist = None
        if features[0].emo_in_dist is not None:
            emo_in_dist = torch.tensor([f.emo_in_dist for f in features], dtype=torch.float64).squeeze(1)
        else:
            emo_in_dist = None
        strat_positions = pad_sequence([torch.tensor([i for i,x in enumerate(f.is_strat_targ) if x == 1], dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=-1)
        emo_positions = pad_sequence([torch.tensor([i for i,x in enumerate(f.is_emo_targ) if x == 1], dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=-1)

        intensity = None
     
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids,
            'role_ids': role_ids,
            'labels': labels,
            'cls_positions': cls_positions,
            'cls_labels': cls_labels,
            'strategy_ids': strategy_ids,
            'decoder_input_ids': decoder_input_ids,
            'decoder_position_ids': decoder_position_ids,
            'decoder_token_type_ids': decoder_token_type_ids,
            'decoder_role_ids': decoder_role_ids,
            'decoder_labels': decoder_labels,
            'decoder_cls_positions': decoder_cls_positions,
            'decoder_cls_labels': decoder_cls_labels,
            'decoder_strategy_ids': decoder_strategy_ids,
            'comet_ids': comet_ids,
            'comet_mask': comet_mask,
            'emotion': emotion,
            'comet_st_ids': comet_st_ids,
            'comet_st_mask': comet_st_mask,
            'emo_dist': emo_dist,
            'emo_in_dist': emo_in_dist,
            'situations': situations,
            'strat_positions': strat_positions,
            'emo_positions': emo_positions,
            'intensity': intensity,
            'vad_ids': None
        }


def load_and_cache_examples(args, tokenizer, df, comet, comet_st, evaluate=False, strategy=True, test=False, **kwargs):
    if "situations" in kwargs.keys():
        situations = kwargs["situations"]
    else:
        situations = None
    #vad_tokenizer.load_transformer_tokenizer(args.model_name_or_path, args = args)
    #args.zero_vad_token = "[0v0a0d]"
    #args.zero_vad_token_id = tokenizer.convert_tokens_to_ids(args.zero_vad_token)
    return ESDDataset(tokenizer, args, df, comet, comet_st, block_size=args.block_size, evaluate=evaluate, strategy=strategy, test=test, situations = situations, vad_tokenizer = None)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.do_train:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    #if args.n_gpu > 0:
    #    torch.cuda.manual_seed_all(args.seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

# Training of model
def freeze_emo_stg_paras(model):
    emo_stg_para_formats = ["emo","strategy","trans_mat"]
    frozen_paras_names = []
    for n, param in model.named_parameters():
        if any(fm in n for fm in emo_stg_para_formats):
            frozen_paras_names.append(n)
            param.requires_grad = False
            logger.info(f"freeze parameter,{n}")
    print(f"frozen_paras_names {frozen_paras_names}")

def load_optimizer_grouped_params_with_doubled_lr(model, no_decay, paras_to_double, lr, double_ratio):
    #emo_stg_para_formats = ["emo","strategy","trans_mat"]
    optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not n in paras_to_double],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not n in paras_to_double], 
        "weight_decay": 0.0},
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n in paras_to_double], 
        "weight_decay": 0.0, "lr":lr * double_ratio},
    ]
    return optimizer_grouped_parameters

def load_optimizer(args, model, len_dataloader):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    t_total = len_dataloader // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        scheduler = None
    return optimizer, scheduler


def train(args, logger, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    print("args.local_rank = ", args.local_rank)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate, drop_last = False
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    if 1 == 2:
        assert 1 == 2
        emo_stg_para_formats = ["emo","strategy","trans_mat"]
        frozen_paras_names = []
        for n, param in model.named_parameters():
            if any(fm in n for fm in emo_stg_para_formats):
                frozen_paras_names.append(n)
        optimizer_grouped_parameters = load_optimizer_grouped_params_with_doubled_lr(
            model, no_decay, frozen_paras_names, args.lr, 0.05
        )
    else:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        scheduler = None

    # Check if saved optimizer or scheduler states exist
    if False and (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    #if args.n_gpu > 1:
        #model = BalancedDataParallel(2,model, dim=0).to(args.device)
    #    print("n gpu > 1")
    #    model = torch.nn.DataParallel(model, device_ids = [0, 1]).cuda()



    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        ).to(args.device)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
 
    # Check if continuing training from a checkpoint
    if False and args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss, tr_lm_loss, logging_lm_loss, tr_emo_loss, \
    logging_emo_loss, tr_strategy_loss, logging_strategy_loss, tr_intensity_loss, logging_intensity_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    best_ppl = 1e8
    best_bleu_2 = 0

    model.zero_grad()
    #train_iterator = range(epochs_trained, int(args.num_train_epochs))
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=True)
    set_seed(args)  # Added here for reproducibility
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    for epoch in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in tqdm(enumerate(epoch_iterator), total = len(epoch_iterator)):
            #print("step:",step)
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            if args.local_rank != -1:
                #print("here")
                outputs, _ = shared_steps(batch, model = model.module, tokenizer= tokenizer,args= args, phase = "train")
            else:
                outputs, _ = shared_steps(batch, model = model, tokenizer= tokenizer,args= args, phase = "train")

            loss = outputs.loss
            lm_loss = ppl = outputs.lm_loss
            emo_loss = outputs.emo_loss
            intensity_loss = outputs.intensity_loss
            strategy_loss = outputs.strategy_loss
            
            # if not args.no_cuda and args.n_gpu >= 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
            #     ppl = ppl.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                backward_loss = outputs.loss
                # backward_loss = outputs.lm_loss
                # if epoch == 0 or epoch == 1:
                #     backward_loss = outputs.strategy_loss
                # else:
                #     backward_loss = outputs.loss
                backward_loss.backward()

            tr_loss += loss.item()
            tr_lm_loss += lm_loss.item()
            tr_emo_loss += emo_loss.item()
            tr_strategy_loss += strategy_loss.item()
            #tr_intensity_loss += intensity_loss.item()
            tr_intensity_loss = 0


            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 and global_step >t_total*0.0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        with torch.no_grad():
                            results = evaluate(args, model, tokenizer, args.eval_dataset, "{}-{}".format("checkpoint", global_step))
                            wandb.log(results)
                            if epoch > 3 :
                                test_result = generate_new(args, model, verbose = False, prefix = "{}-{}-".format("checkpoint", global_step))
                                wandb.log(test_result)
                            
                    else:
                        with torch.no_grad():
                            results = evaluate(args, model.module, tokenizer, args.eval_dataset, "{}-{}".format("checkpoint", global_step))
                            wandb.log(results)
                            if epoch > 3:
                                pass
                                test_result = generate_new(args, model, verbose = False, prefix = "{}-{}".format("checkpoint", global_step))
                                wandb.log(test_result)
                            
                        #test_results = evaluate(args, model, tokenizer, args.eval_dataset, "{}-{}".format("checkpoint", global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    if scheduler is not None:
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logger.info("lr: %f, step: %d, loss: %f, lm_loss: %f, emo_loss: %f, strategy_loss: %f, intensity_loss: %f", scheduler.get_last_lr()[0],
                                    global_step, (tr_loss - logging_loss) / args.logging_steps, (tr_lm_loss - logging_lm_loss) / args.logging_steps,
                                    (tr_emo_loss - logging_emo_loss) / args.logging_steps, (tr_strategy_loss - logging_strategy_loss) / args.logging_steps,
                                    (tr_intensity_loss - logging_intensity_loss) / args.logging_steps)
                    
                    logging_loss = tr_loss
                    logging_lm_loss = tr_lm_loss
                    logging_emo_loss = tr_emo_loss
                    logging_strategy_loss = tr_strategy_loss
                    logging_intensity_loss = tr_intensity_loss
                    if epoch > 3:
                        #if test_result["bleu-2"] > best_bleu_2:
                        #    best_bleu_2 = test_result["bleu-2"]
                        #    checkpoint_prefix = "bleu_checkpoint"
                        #    output_dir = os.path.join(args.output_dir, "bleu2")
                        #    save_checkpoint(args, model, tokenizer, output_dir, checkpoint_prefix, optimizer, scheduler)
                        pass
                            
                    if results['eval_perplexity']< best_ppl :
                        best_ppl = results['eval_perplexity']

                        checkpoint_prefix = "checkpoint"

                        output_dir = args.output_dir
                        save_checkpoint(args, model, tokenizer, output_dir, checkpoint_prefix, optimizer, scheduler)
                        

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    print("Train finished~")
    return global_step, tr_loss / global_step

# Evaluation of some model
def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix="", eval_output_dir = None, show_emotion = False) -> Dict:
    print("evaluating")
    import numpy as np
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    if eval_output_dir is None:
        eval_output_dir = args.output_dir

        #eval_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=True)
    os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate, drop_last = False
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    #multi-gpu evaluate
    #if args.n_gpu > 1:
    #    model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    intensity_loss = 0.0
    kl_loss = 0.0
    emo_out_loss = 0.0
    contrastive_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    strategy_probs = []
    cls_labels_list = []
    num_samples = []
    emo_hits = []
    # strategy_hits_topk = [[] for _ in range(7)]
    strategy_hits = []
    if show_emotion:
        turn_ids = []
        emotions = []
        
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating",disable=True, total = len(eval_dataloader)):
            outputs, (emotion, decoder_input_ids, decoder_strategy_ids, decoder_label_ids) = shared_steps(batch, model, tokenizer, args, phase = "eval")
            if show_emotion:
                cur_turn_ids = [max(x) for x in batch["decoder_token_type_ids"].detach().cpu().tolist()]
                turn_ids += cur_turn_ids
                cur_emotions = outputs.emo_out_prob.detach().exp().cpu().tolist()
                emotions += cur_emotions
        
            loss = outputs.loss
            ppl = outputs.lm_loss
            emo_logits = outputs.emo_logits
            strategy_logits = outputs.strategy_logits
            

            # print(strategy_logits.argmax(dim=-1))
            for idx, emo_logit in enumerate(emo_logits):
                if emo_logit.argmax() == emotion[idx]:
                    emo_hits.append(1)
                else:
                    emo_hits.append(0)

            # print(decoder_input_ids)
            # strategy_ids = decoder_input_ids[:, 0] - 54944

            for idx, strategy_logit in enumerate(strategy_logits):
                if strategy_logit.argmax() == decoder_strategy_ids[idx]:
                    strategy_hits.append(1)
                else:
                    strategy_hits.append(0)


            # if args.strategy:
            #     cls_labels_list.extend(decoder_cls_labels.cpu().numpy().tolist())
            #     strategy_probs.append(torch.nn.functional.softmax(outputs.lm_logits[0, 0, 54945:54945+8], dim=-1).cpu().numpy().tolist())

            lm_loss = outputs.lm_loss
            num_samples.append((decoder_label_ids.cpu().numpy() != -100).astype(np.int32).sum())
            eval_loss += lm_loss.sum().item() * (decoder_label_ids.cpu().numpy() != -100).astype(np.int32).sum()
            if outputs.intensity_loss is not None:
                intensity_loss += (outputs.intensity_loss.detach().cpu().item() - intensity_loss) / (nb_eval_steps + 1)
            #assert outputs.kl_loss is not None
            #if outputs.kl_loss is not None:
            #    kl_loss += (outputs.kl_loss.detach().cpu().item() - kl_loss) / (nb_eval_steps + 1)
            if outputs.contrastive_loss is not None:
                contrastive_loss += (outputs.contrastive_loss.detach().cpu().item() - contrastive_loss) / (nb_eval_steps + 1)
            if outputs.emo_out_loss is not None:
                emo_out_loss += (outputs.emo_out_loss.detach().cpu().item() - emo_out_loss) / (nb_eval_steps + 1)

        nb_eval_steps += 1
        
    if show_emotion:
        return turn_ids, emotions

    eval_loss = eval_loss/ sum(num_samples)
    
    perplexity = torch.exp(torch.tensor(eval_loss)).item()
    # np_strategy = np.array(strategy_probs)
    # np_cls_labels = np.array(cls_labels_list)
    # result = {"eval_perplexity": perplexity, "eval_emotion_predict_accuracy": sum(emo_hits)/len(emo_hits), "eval_strategy_predict_accuracy": sum(strategy_hits)/len(strategy_hits), "eval_number_of_evaluated_examples": len(emo_hits)}
    result = {"eval_perplexity": perplexity, "eval_emotion_predict_accuracy": sum(emo_hits) / len(emo_hits),"eval_strategy_predict_accuracy": sum(strategy_hits) / len(strategy_hits),
              "eval_number_of_evaluated_examples": len(emo_hits),
              "contrastive_loss":contrastive_loss,
              "emo_out_loss":emo_out_loss
              }
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")


    with open(output_eval_file, "a+") as writer:
        # print("***** Eval results {} *****".format(prefix))
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("***** Eval results {} *****".format(prefix) + "\n")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            # print("  %s = %s" % (key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))
    
    return result



def generate_new(args, model = None, verbose = True, prefix = "",test_output_dir = None):

    #additional_special_tokens = ["[Question]", "[Reflection of feelings]", "[Information]",
    #                            "[Restatement or Paraphrasing]", "[Others]", "[Self-disclosure]",
    #                            "[Affirmation and Reassurance]", "[Providing Suggestions]"]
    # comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]"]
    #comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]",
    #                                "[oEffect]", "[oReact]"]

    _, tokenizer = load_tokenizer(args)
    if test_output_dir is None:
        test_output_dir = args.output_dir
    os.makedirs(test_output_dir, exist_ok=True)
    if model is None:
        model = load_model_for_eval(args=args)
    model.eval()
    #C = model.model.encoder.strategy_embedding.weight[:8,:]
    #C = C.cpu().detach().numpy()
    #from sklearn.metrics.pairwise import cosine_similarity
    #print(cosine_similarity(C))

    #print(1/0)
    #model.resize_token_embeddings(len(tokenizer))
    #model.resize_token_embeddings(54944) 
    # Setup CUDA, GPU & distributed training
    if  not args.no_cuda:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        device = torch.device("cpu")
        args.device = device
        args.n_gpu = 0
    if not "test" in prefix:
        with open(args.data_path+"/"+args.eval_file_name,"r") as f:
            chat_texts = f.read().split("\n")
    else:
        with open(args.data_path+"/"+args.test_file_name,"r") as f:
            chat_texts = f.read().split("\n")
    set_seed(args)
    gts = []
    refs = []
    #prevs = []
    mutual_attentions = []
    mutual_attentions_st = []
    strategy_logit_str = []
    model.to(args.device)
    # Let's chat for 5 lines
    strategy_hits = []
    strategy_record = []
    strategy_hits_topk = [[] for _ in range(8)]
    if not "test" in prefix:
        dataset = args.eval_dataset
    else:
        dataset = args.test_dataset
    for idx in tqdm(range(len(dataset)), desc="Testing"):
        f = dataset[idx]
        batch = dataset.collate([f])
        input_ids, paras = shared_steps(batch, model, tokenizer, args, phase = "generation")
        #prev = tokenizer.batch_decode(input_ids, skip_special_tokens = False)[0].replace("__null__","").split("__end__")[-2]
        next_strategy_id = f.decoder_strategy_ids[0]
        decoder_strategy_ids = torch.tensor([f.decoder_strategy_ids], dtype=torch.long)
        decoder_strategy_ids = decoder_strategy_ids.to(device)
        decoder_strategy_ids = decoder_strategy_ids[:, 0]

        gts.append(tokenizer.decode(f.decoder_input_ids, skip_special_tokens=True))
        with torch.no_grad():
            chat_history_ids, mutual_attention, mutual_attention_st, strategy_logits = model.generate(
                input_ids,
                **paras, max_length=512,
                min_length=5,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id, temperature=0.7,
                top_p=0.3, 
                top_k = 30, 
                do_sample=True, 
                #no_repeat_ngram_size=3,
                repetition_penalty=1.03
                ) #top_p 0.9, topk 30

        if mutual_attention is not None:
            chat_history_ids, mutual_attention, mutual_attention_st = chat_history_ids.cpu(), mutual_attention[-1][0].cpu(), mutual_attention_st[-1][0].cpu()
            mutual_attention = torch.mean(mutual_attention, dim=0) 
            mutual_attention_st = torch.mean(mutual_attention_st, dim=0)
            mutual_attentions.append(mutual_attention)
            mutual_attentions_st.append(mutual_attention_st)
        else:
            chat_history_ids = chat_history_ids.cpu()

        generated_text = tokenizer.decode(chat_history_ids[:, :][0], skip_special_tokens=True)
        refs.append(generated_text)
        #prevs.append(prev)
        if verbose:
            print(generated_text)
        
        strategy_record.append({"ref strategy":tokenizer.decode([next_strategy_id + args.base_vocab_size]),  "hyp strategy":tokenizer.decode([strategy_logits[0].argmax()+args.base_vocab_size])})
        # print({"ref strategy":tokenizer.decode([next_strategy_id + 54944]),  "hyp strategy":tokenizer.decode([chat_history_ids[:, :][0][1]])})
        if verbose:
            print({"ref strategy": tokenizer.decode([next_strategy_id + args.base_vocab_size]),
                "hyp strategy": tokenizer.decode([strategy_logits[0].argmax() + args.base_vocab_size])})
        if strategy_logits[0].argmax() == next_strategy_id:
            strategy_hits.append(1)
        else:
            strategy_hits.append(0)

        for k in range(8):
            _, topk = strategy_logits[0].topk(k+1, -1)
            strategy_hits_topk[k].append(sum((topk == next_strategy_id).cpu().numpy().tolist()))
        strategy_logits = strategy_logits[0].cpu().numpy().tolist()
        strategy_logits = ["%.4f" % logit for logit in strategy_logits]
        strategy_logit_str.append('\t'.join(strategy_logits))
        # print(strategy_logit_str)
        # print(strategy_hits_topk)
        # print(1 / 0)
    for i in range(8):
        print(sum(strategy_hits_topk[i]) / len(strategy_hits_topk[i]))
    print('strategy predict accuray', sum(strategy_hits)/len(strategy_hits))

    all_top_k_blocks = None
    all_top_k_blocks_st = None
    if not os.path.exists(args.generation_dir):
        os.makedirs(args.generation_dir)
    test_file_path = f"{args.data_path}/testWithStrategy_short.tsv"
    test_situation_file_path = f"{args.data_path}/testSituation.txt"
    strategy_record_file_path = os.path.join(args.generation_dir, "strategy_record.json")
    generate_file_path = os.path.join(args.generation_dir, "hyp_strategy.json")
    prev_file_path = os.path.join(args.generation_dir, "prev.json")
    reference_file_path = os.path.join(args.generation_dir, "ref_strategy.json")
    summary_file_path = os.path.join(args.generation_dir, "summary.txt")
    strategy_logits_file = os.path.join(args.generation_dir, "strategy_logits.txt")
    with open(strategy_logits_file, "w", encoding="utf-8") as f:
        for item in strategy_logit_str:
            f.write(item + '\n')

    with open(strategy_record_file_path, "w",encoding="utf-8") as f:
        json.dump(strategy_record,f,indent=2,ensure_ascii=False)
    with open(generate_file_path, "w",encoding="utf-8") as f:
        json.dump(refs,f,indent=2,ensure_ascii=False)
        wandb.log({"hyps":refs})
    with open(reference_file_path,"w",encoding="utf-8") as f:
        json.dump(gts,f,indent=2,ensure_ascii=False)
        wandb.log({"refs":gts})
    metric = Metric(toker=tokenizer, hyp_path=generate_file_path, ref_path=reference_file_path, use_nltk = True)
    result, result_list = metric.close()
    print(result)
    print("=" * 100)
    summary(test_file_path, generate_file_path, reference_file_path, summary_file_path, all_top_k_blocks, all_top_k_blocks_st, chat_texts, test_situation_file_path)

    print("write result to:", summary_file_path)
    print("Generate finished~")
    output_test_file = os.path.join(test_output_dir, "test_results.txt")
    with open(output_test_file, "a+") as writer:
        # print("***** Eval results {} *****".format(prefix))
        logger.info("***** Test results {} *****".format(prefix))
        writer.write("***** Eval results {} *****".format(prefix) + "\n")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            # print("  %s = %s" % (key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return result



def shared_steps(batch, model, tokenizer, args, phase = "train"):
    if phase == "train":
        model.train()
    else:
        model.eval()
    
    input_ids, position_ids, turn_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_turn_ids, \
            decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask, emotion, comet_ids_st, comet_mask_st, emo_dist, emo_in_dist, situ_ids, strat_positions, emo_positions, intensity, vad_ids = batch.values()
    decoder_strategy_ids = decoder_strategy_ids[:, 0]
    decoder_strategy_ids = decoder_strategy_ids.to(args.device)
    assert input_ids.shape[1] <= 512 
    emotion = emotion.to(args.device)
    situation_hidden_states = None
    situ_attention_mask = None


    #if not args.wo_comet:
    comet_ids = comet_ids.to(args.device)
    batch_size, n_attr, len_attr = comet_ids.shape
    comet_ids = comet_ids.view(-1, len_attr)
    with torch.no_grad():
        if isinstance(model,torch.nn.DataParallel):
            comet_embs = model.module.model.encoder(comet_ids, attention_mask = comet_ids.ne(tokenizer.pad_token_id))[0][:,0,:]
        else:
            comet_embs = model.model.encoder(comet_ids, attention_mask = comet_ids.ne(tokenizer.pad_token_id))[0][:,0,:]
    comet_embs = comet_embs.view(batch_size, n_attr, -1)
    comet_ids_st = comet_ids_st.to(args.device)
    batch_size, n_attr, len_attr = comet_ids_st.shape
    comet_ids_st = comet_ids_st.view(-1, len_attr)
    with torch.no_grad():
        if isinstance(model,torch.nn.DataParallel):
            comet_embs_st = model.module.model.encoder(comet_ids_st, attention_mask=comet_ids_st.ne(tokenizer.pad_token_id))[0][:, 0, :]
        else:
            comet_embs_st = model.model.encoder(comet_ids_st, attention_mask=comet_ids_st.ne(tokenizer.pad_token_id))[0][:, 0, :]
    comet_embs_st = comet_embs_st.view(batch_size, n_attr, -1)
    comet_mask = comet_mask.to(args.device)
    comet_mask_st = comet_mask_st.to(args.device)
    del comet_ids
    del comet_ids_st

    input_ids = input_ids.to(args.device)
    intensity = None
    
    decoder_input_ids = decoder_input_ids.to(args.device)
    decoder_label_ids = decoder_labels.to(args.device)
    if phase == "eval":
        decoder_cls_labels = decoder_cls_labels.to(args.device) 
    emo_dist = emo_dist.to(args.device) if emo_dist is not None else None
    emo_in_dist = emo_in_dist.to(args.device) if emo_in_dist is not None else None
    strat_positions = None
    emo_positions = emo_positions.to(args.device)
    role_ids = role_ids.to(args.device)
    vad_ids = None
    if phase == "train":
        outputs = model(input_ids, attention_mask = input_ids.ne(tokenizer.pad_token_id), 
                        decoder_input_ids=decoder_input_ids, 
                        decoder_turn_ids=None, 
                        decoder_role_ids=None, 
                        turn_ids=None, 
                        role_ids=role_ids,
                        vad_ids=vad_ids,
                        labels = decoder_label_ids,
                        decoder_strategy_ids=decoder_strategy_ids, 
                        comet_embs=comet_embs, 
                        comet_mask=comet_mask, 
                        comet_embs_st=comet_embs_st, 
                        comet_mask_st=comet_mask_st, 
                        emotion=emotion,
                        emo_dist = emo_dist, 
                        emo_in_dist = emo_in_dist, 
                        strat_positions = strat_positions, 
                        emo_positions = emo_positions, 
                        intensity = intensity,
                        situation_hidden_states = situation_hidden_states,
                        situation_attention_mask = situ_attention_mask
                        )
    elif phase in ["generation"]:
        paras = {}
        paras["attention_mask"] =  input_ids.ne(tokenizer.pad_token_id)
        paras["comet_embs"] = comet_embs
        paras["comet_mask"] = comet_mask
        paras["comet_embs_st"] = comet_embs_st
        paras["comet_mask_st"] = comet_mask_st
        paras["situation_hidden_states"] = situation_hidden_states
        paras["situation_attention_mask"] = situ_attention_mask
        paras["emo_dist"] = emo_dist
        paras["emo_in_dist"] = emo_in_dist
        paras["output_mutual_attentions"] = False
        paras["strat_positions"] = strat_positions
        paras["emo_positions"] = emo_positions
        paras["intensity"] = intensity
        paras["role_ids"] = role_ids
        paras["generate_with_predicted_strategy"] = args.generate_with_predicted_strategy

        return input_ids, paras
    else:
        with torch.no_grad():
            outputs = model(input_ids, 
                            attention_mask = input_ids.ne(tokenizer.pad_token_id), 
                            decoder_input_ids=decoder_input_ids,
                            decoder_turn_ids=None, 
                            decoder_role_ids=None, 
                            turn_ids=None, 
                            role_ids=role_ids,
                            vad_ids=vad_ids,
                            labels = decoder_label_ids, 
                            decoder_strategy_ids=decoder_strategy_ids, 
                            comet_embs=comet_embs, 
                            comet_mask=comet_mask, 
                            comet_embs_st=comet_embs_st, 
                            comet_mask_st=comet_mask_st, 
                            emotion=emotion, 
                            emo_dist = emo_dist, 
                            emo_in_dist = emo_in_dist, 
                            strat_positions = strat_positions, 
                            emo_positions = emo_positions, 
                            intensity = intensity,
                            situation_hidden_states = situation_hidden_states,
                            situation_attention_mask = situ_attention_mask
                            )
    return outputs, (emotion, decoder_input_ids, decoder_strategy_ids, decoder_label_ids)
if __name__ == "__main__":
    args = Args()
    generate_new(args)
