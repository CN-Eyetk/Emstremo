import argparse
import wandb
import re
import numpy as np
from BlenderEmotionalSupport import load_dataset
import os

import torch
import argparse
import os
import logging
import json

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type = str, default=".")
parser.add_argument("--data_path", type = str, default="converted_dataset")
parser.add_argument("--explain", action= "store_true")
parser.add_argument("--use_bart", action= "store_true")
parser.add_argument("--do_train",action="store_true")
parser.add_argument("--log_on_wandb",action="store_true")
parser.add_argument("--over_write", action= "store_true")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--tag", type=str)
parser.add_argument("--warmup_steps", type = int, default = 100)
parser.add_argument("--pretrained_model_path", type = str, default = None)
parser.add_argument("--generate_with_predicted_strategy",action="store_true")
parser.add_argument("--block_size",type=int, default=512) #No strategy control over response
args_g = parser.parse_args()
root_path = args_g.root_path

OVERWRITE = args_g.over_write
BART = args_g.use_bart


os.environ["WANDB_DISABLED"] = "true" if not args_g.log_on_wandb else "false"
if args_g.pretrained_model_path is not None:
    TAG = args_g.pretrained_model_path.split("/")[-1]
    GROUP = args_g.pretrained_model_path.split("/")[-2]
else:
    TAG = args_g.tag
                                

    GROUP = "pretrained_model"



from BlenderEmotionalSupport import (
                                    load_and_cache_examples, 
                                    InputFeatures_blender,
                                    train,
                                    evaluate,
                                    generate_new,
                                    load_tokenizer,
                                    set_seed,
                                    load_model_for_eval,
                                    load_model,
                                    logger,
                                    load_optimizer
                                    )
if  args_g.pretrained_model_path is not None:
    output_dir = args_g.pretrained_model_path
    #generation_dir = "our_generated_data/" + GROUP + "/" + TAG
    generation_dir = output_dir.replace(args_g.root_path, "our_generated_data")
else:
    if BART:
        output_dir = os.path.join(root_path, 'bart-our', GROUP, TAG)
    else:
        output_dir = os.path.join(root_path, 'blender-our', GROUP, TAG)
    generation_dir = "our_generated_data/" + GROUP + "/" + TAG
if args_g.generate_with_predicted_strategy:
    generation_dir = os.path.join(generation_dir, "non_mix")
#from src.transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallForConditionalGeneration
logger = logging.getLogger(__name__)

def load_arg():
    #torch.distributed.init_process_group(backend="nccl")
    #local_rank = torch.distributed.get_rank()
    args = {"do_train":args_g.do_train,
           "data_path":args_g.data_path, 
            "train_comet_file":"trainComet.txt",
            "situation_train_file":"trainSituation.txt",
            "situation_train_comet_file":"trainComet_st.txt",
            "train_file_name":"trainWithStrategy_short.tsv",
            "eval_comet_file":"devComet.txt",
            "situation_eval_file":"devSituation.txt",
            "situation_eval_comet_file":"devComet_st.txt",
            "eval_file_name":"devWithStrategy_short.tsv",
            "test_comet_file":"testComet.txt",
            "situation_test_file":"testSituation.txt",
            "situation_test_comet_file":"testComet_st.txt",
            "test_file_name":"testWithStrategy_short.tsv",
            "data_cache_dir":"{}/124_II_{}_{}_{}{}{}{}cached".format(root_path,"noprep", "bart_" if BART else "", "", "", args_g.data_path if not args_g.data_path == "converted_dataset" else "",args_g.block_size if args_g.block_size != 512 else ""),
            "model_type":"mymodel",
            "overwrite_cache":OVERWRITE,
            "model_name_or_path":"facebook/blenderbot_small-90M" if not BART else "facebook/bart-base",
            "base_vocab_size":54944 if not BART else 50265,
            "model_cache_dir":"./blender-small",
            "strategy":False,
            "local_rank":-1,#local_rank,
            "per_gpu_train_batch_size":20,
            "per_gpu_eval_batch_size":20,
            "save_total_limit":1,
            "n_gpu":1,
            "max_steps":-1,
            "gradient_accumulation_steps":1,
            "weight_decay":0,
            "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "learning_rate":args_g.lr,
            "adam_epsilon":1e-8,
            "warmup_steps":args_g.warmup_steps,#once 510
            "fp16":False,
            "fp16_opt_level":'O1',
            "num_train_epochs":10 if BART else 8,
            "role":False,
            "turn":False,
            "logging_steps":300,#1 March from 510 to 300
            "evaluate_during_training":True,
            "output_dir":output_dir,
            "seed":42,
            "max_grad_norm":1.0,
            "no_cuda":False,
            "block_size":args_g.block_size,
            "generation_dir":generation_dir,
            "use_bart":BART,
            
            }

    args = argparse.Namespace(**args)
    print("data_cache_dir",args.data_cache_dir)
    return args




def plot(model, strat_labels, emo_in_labels, emo_out_labels):
    import pandas as pd
    with torch.no_grad():
        mats = model.model.encoder.trans_mat.matrices
        weights = []
        for i,mat in enumerate(mats):
            cur_strat = strat_labels[i]
            weight = mat.detach().cpu().numpy()
            df = pd.DataFrame(weight, columns=emo_out_labels, index=emo_in_labels)
            print(df.shape)
            print(df)
            df.to_csv(f"matrices/{cur_strat}.csv", sep = "\t")
            weights.append(df)
    return weights

    

def explain(args):
    if args.data_path == "converted_dataset":
        stra_labels = ["[Question]","[Reflection of feelings]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions or Information]","[Greeting]"]
    else:
        stra_labels = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
    emo_in_labels = open("dataset/labels/esconv_emo_labels.txt","r+").read().split("\n")
    emo_out_lables =  json.load(open("dataset/labels/emo_out_labels.json"))
    emo_out_labels = [v for k,v in emo_out_lables.items()]
    plot(model, strat_labels=stra_labels, emo_in_labels=emo_in_labels, emo_out_labels=emo_out_labels)

def show_emotion(args):
    import pandas as pd
    turns, emotions = evaluate(args, model, tokenizer, args.test_dataset, "of test set", show_emotion = True)
    print("-----turns-------")
    print(turns[:5])
    print("-----emotions-------")
    print(emotions[:5])
    emo_out_lables =  json.load(open("dataset/labels/emo_out_labels.json"))
    res = {}
    res["turn"] = turns
    for i,emo in enumerate(emo_out_lables):
        res[emo] = [emotion[i] for emotion in emotions]
    df = pd.DataFrame(res)
    df.to_csv("emotion_output.csv",sep = "\t")
        
    
    
if __name__ == "__main__":
    args = load_arg()
    wandb.init(config=args)
    print(args.output_dir)
    set_seed(args)
    _, tokenizer = load_tokenizer(args = args)


    train_dataset, eval_dataset, test_dataset = load_dataset(args, tokenizer)
    args.train_dataset = train_dataset
    args.eval_dataset = eval_dataset
    args.test_dataset = test_dataset

    if args.do_train:
        model = load_model(args, tokenizer)

        global_step, tr_loss = train(args, logger, args.train_dataset, model, tokenizer)
    model = load_model_for_eval(args)
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        #matrices = model.
        if args_g.explain:
            explain(args)
        elif args_g.do_show_emotion:
            show_emotion(args)
        else:
            test_results = evaluate(args, model, tokenizer, args.test_dataset, "of test set")
            result = generate_new(args, model = model, prefix="of test set")
            prefix = args.generation_dir.split("/")[-2] if re.compile(r"^.*?/$").search(args.generation_dir) else generation_dir.split("/")[-1]
