import torch
import torch.nn as nn
import requests
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
import datetime
import copy
import math
import time
import gc
import os
import string

from utils import *
from train import *
from eval_ppl import *
from eval_noise import *
from eval_lmevalharness import *
from eval_typo import *
from eval_typoglycemia import *
from eval_noise_summary import *

from distutils.util import strtobool

def main():
    
    args = parse_arguments()
    
    fix_seed(args.random_seed)

    if args.scenario == "train":
        result = train(args)
    elif args.scenario == "eval_noise":
        result = eval_noise(args)
    elif args.scenario == "eval_typoglycemia":
        result = eval_typoglycemia(args)
    elif args.scenario == "eval_noise_summary":
        result = eval_noise_summary(args)
    elif args.scenario == "eval_typo":
        result = eval_typo(args)
    elif args.scenario == "eval_ppl":
        result = eval_ppl(args)
    elif args.scenario == "eval_lmevalharness":
        eval_lmevalharness(args)
        result = "Check log file."
    else:
        raise ValueError("scenario is not properly defined!")

    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d-%H%M%S')
    
    with open(args.result_file, mode='a') as f:
        result = f"{now}; {args.scenario}; {args.model_name}; {args.trained_param_file_suffix}; {args.restore_steps}; {args.random_seed}; {result}\n"
        f.write(result)

def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("--scenario", type=str, default="", choices=["train", "eval_ppl", "eval_lmevalharness", "eval_noise", "eval_noise_summary", "eval_typo", "eval_typoglycemia"], help="")
    parser.add_argument("--result_file", type=str, default="result.txt")
    
    parser.add_argument("--train_method", type=str, default="ours", choices=["ours", "baseline1", "baseline2", "baseline3"], help="")
    parser.add_argument("--bpe_dropout", action="store_true", help="")
    
    parser.add_argument("--checkpointing", action="store_true", help="")
    
    parser.add_argument("--trained_param_file_suffix", type=str, default=None, help="")
    parser.add_argument("--restore_steps", type=int, default=None, help="")
        
    parser.add_argument("--model_name", type=str, default="", choices=["gpt2-xl", "EleutherAI/pythia-2.8b", "google/gemma-2-2b", "mistralai/Mistral-7B-v0.1", "meta-llama/Meta-Llama-3-8B"])

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=8)
    
    parser.add_argument("--total_global_step", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=250)

    parser.add_argument("--eval_dataset_size", type=int, default=1000)
    parser.add_argument("--eval_ppl_dataset", type=str, default="", choices=["redpajama", "fineweb"])
    parser.add_argument("--eval_noise_dataset", type=str, default="ag_news", choices=["sst2", "ag_news", "github-typo-corpus", "typoglycemia", "EdinburghNLP/xsum", "bea60k", "SetFit/amazon_reviews_multi_en", "SetFit/sst5"])
    parser.add_argument("--lmeval_task", type=str, default="", choices=["multilingual", "downstream"], help="")

    parser.add_argument("--injection_typo_flag", action="store_true", help="")
    parser.add_argument("--balance_label_flag", action="store_true", help="")
    parser.add_argument("--noise_type", type=str, default="nothing", choices=["nothing", "drop", "addition", "uppercase", "randomcase", "mask", "decomposition", "swap", "multi_id", "multi_ood", "typoglycemia"])

    parser.add_argument("--noise_frequency_train", type=float, default=1.0)
    parser.add_argument("--noise_frequency_test", type=float, default=0.5)
    
    parser.add_argument("--parallel_flag", action="store_true", help="")
    parser.add_argument("--incontex_sample_size", type=int)
    
    parser.add_argument("--random_seed", type=int, default=1)

    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--disk_path", type=str, default="./")
    
    #########################################################################
    
    args = parser.parse_args() # For python file ...
    #args = parser.parse_args(args=[]) # For Jupyter Notebook ...

    args.dataset_cache_dir = os.path.join(args.disk_path, "datasets")
    
    args.HUGGINGFACE_TOKEN = os.environ["HF_TOKEN"]
    #print(args.HUGGINGFACE_TOKEN)

    if args.scenario == "eval_typoglycemia":
        args.noise_type = "typoglycemia"
    
    if args.scenario == "eval_lmevalharness":
        if args.lmeval_task == "multilingual":
            args.lmeval_task = "paws_en,paws_de,paws_fr,paws_es,xnli_en,xnli_de,xnli_fr,xnli_es"
        elif args.lmeval_task == "downstream":
            args.lmeval_task = "winogrande,piqa,openbookqa,hellaswag,ai2_arc"            
        else:
            raise ValueError("error! lmeval_task is not properly defined!")
    
    if "gpt2-xl" in args.model_name:
        args.max_context_length = 1024
        args.batch_size = 64
        args.mini_batch_size = 16
        # gpt2 has tied embedding.
        args.train_target_modules = ["wte", "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]        
        args.model_dtype = torch.bfloat16 #torch.float32
        args.append_bos_flag = False
        args.use_cache = True
    elif "gemma-2-2b" in args.model_name:
        args.max_context_length = 8192
        args.batch_size = 8
        args.mini_batch_size = 1
        # gemma2 has tied embedding.
        args.train_target_modules = ["embed_tokens", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        args.model_dtype = torch.bfloat16 #torch.float32
        args.append_bos_flag = True
        args.use_cache = True # to avoid "different devices" error when learning on multiple gpus. https://github.com/huggingface/transformers/issues/33147
    elif "pythia-2.8b" in args.model_name:
        args.max_context_length = 2048
        args.batch_size = 32
        args.mini_batch_size = 16
        args.train_target_modules = ["embed_in", "attention.query_key_value", "attention.dense", "dense_h_to_4h", "dense_4h_to_h", "embed_out"]
        args.model_dtype = torch.bfloat16 #torch.float32
        args.append_bos_flag = False
        args.use_cache = True
    elif "Mistral-7B" in args.model_name:
        args.max_context_length = 8192
        args.batch_size = 8
        args.mini_batch_size = 1
        args.train_target_modules = ["embed_tokens", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "lm_head"]
        args.model_dtype = torch.bfloat16 #torch.float32
        args.append_bos_flag = True
        args.use_cache = True
    elif "Llama-3-8B" in args.model_name:
        args.max_context_length = 8192
        args.batch_size = 8
        args.mini_batch_size = 1
        args.train_target_modules = ["embed_tokens", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "lm_head"]
        args.model_dtype = torch.bfloat16 #torch.float32
        args.append_bos_flag = True
        args.use_cache = True
    else:
        raise ValueError("error! model_name is not properly defined!")

    if args.eval_noise_dataset == "sst2":
        args.instruction = 'Choose a sentiment from "positive" or "negative".\n\n'
        args.num_text_limit = 100
        args.num_class = 2
        args.incontex_sample_size = 4
        args.label_list = ["negative", "positive"]
        args.input_lable = "sentence"
        args.output_lable = "label"
        args.prefix1 = "review:"
        args.prefix2 = "sentiment:"
        args.ds_train_lable = "train"
        args.ds_test_lable = "validation"
    elif args.eval_noise_dataset == "SetFit/sst5":
        args.instruction = 'Choose a sentiment from "very negative", "negative", "neutral", "positive", or "very positive".\n\n'
        args.num_text_limit = 100
        args.num_class = 5
        args.incontex_sample_size = 5
        args.label_list = ["very negative", "negative", "neutral", "positive", "very positive"]
        args.input_lable = "text"
        args.output_lable = "label"
        args.prefix1 = "review:"
        args.prefix2 = "sentiment:"
        args.ds_train_lable = "train"
        args.ds_test_lable = "test"
    elif args.eval_noise_dataset == "ag_news":
        args.instruction = 'Choose a topic from "world", "sports", "business" or "technology".\n\n'
        args.num_text_limit = 100
        args.num_class = 4
        args.incontex_sample_size = 4 #4 #8
        args.label_list = ["world", "sports", "business", "technology"]
        args.input_lable = "text"
        args.output_lable = "label"
        args.prefix1 = "input:"
        args.prefix2 = "type:"
        args.ds_train_lable = "train"
        args.ds_test_lable = "test"
    elif args.eval_noise_dataset == "SetFit/amazon_reviews_multi_en":
        args.instruction = 'Predict the ratings of the following reviews with a value of 1 to 5.\n\n'
        args.num_text_limit = 100
        args.num_class = 5
        args.incontex_sample_size = 5
        args.label_list = ["1", "2", "3", "4", "5"]
        args.input_lable = "text"
        args.output_lable = "label"
        args.prefix1 = "review:"
        args.prefix2 = "rating:"
        args.ds_train_lable = "train"
        args.ds_test_lable = "test"
    elif args.eval_noise_dataset == "EdinburghNLP/xsum": # https://huggingface.co/datasets/EdinburghNLP/xsum?row=6
        args.instruction = 'Summarize the following documents.\n\n'
        args.num_text_limit = 400
        args.incontex_sample_size = 1
        args.input_lable = "document"
        args.output_lable = "summary"
        args.prefix1 = "document:"
        args.prefix2 = "summary:"
        args.ds_train_lable = "train"
        args.ds_test_lable = "test"
        args.max_new_tokens = 100
    elif args.eval_noise_dataset == "github-typo-corpus":
        args.instruction = 'Fix typos in the following texts.\n\n'
        args.num_text_limit = 50
        args.incontex_sample_size = 4 #8
        args.prefix1 = "Source Text:"
        args.prefix2 = "Target Text:"
    elif args.eval_noise_dataset == "bea60k":
        args.instruction = 'Fix typos in the following texts.\n\n'
        args.num_text_limit = 50
        args.incontex_sample_size = 4 #8
        args.prefix1 = "Source Text:"
        args.prefix2 = "Target Text:"
    elif args.eval_noise_dataset == "typoglycemia":
        args.instruction = 'Fix typos in the following texts.\n\n'
        args.num_text_limit = 50
        args.incontex_sample_size = 4 #8
        args.prefix1 = "Source Text:"
        args.prefix2 = "Target Text:"
    
    print('*****************************')
    print(args)
    print('*****************************')

    return args

if __name__ == "__main__":
    main()
