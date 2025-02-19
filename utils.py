import torch
import torch.nn as nn
from datasets import load_dataset
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
import sys

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # 2024-12-28
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

# https://github.com/ccqq77/unnatural-error-correction/blob/main/process.py#L76
def shuffle_different(x):
    slc = random.sample(range(1, len(x)), 1)[0]
    for i in reversed(range(1, len(x))):
        if i == slc:
            # pick an element in x[:i] with which to exchange x[i]
            j = int(random.random() * i)
            x[i], x[j] = x[j], x[i]
        else:
            j = int(random.random() * (i+1))
            x[i], x[j] = x[j], x[i]

def scramble_word_keepfirstlast(word):
    letters = [c for c in word if c.isalpha()]
    if len(letters) <= 3:
        return word
    else:
        letters = letters[1:-1]
        shuffle_different(letters)
        head = []
        tail = []
        for l in range(len(word)):
            if not word[0].isalpha():
                head.append(word[0])
                word = word[1:]
            if not word[-1].isalpha():
                tail.append(word[-1])
                word = word[:-1]
            if word[0].isalpha() and word[-1].isalpha():
                head.append(word[0])
                for c in word[1:-1]:
                    if c.isalpha():
                        head.append(letters.pop(0))
                    else:
                        head.append(c)
                head.append(word[-1])
                break
        result = head + list(reversed(tail))
        return "".join(result)

def noise_delete(word):
    splitted_word = list(word)
    target_idx = random.choice(range(len(splitted_word)))
    splitted_word[target_idx] = ""
    qa_text = "".join(splitted_word)
    return qa_text

def noise_swap(word):
    splitted_word = list(word)
    target_idx1, target_idx2 = random.sample(range(len(word)), 2)
    target_char1 = splitted_word[target_idx1]
    target_char2 = splitted_word[target_idx2]
    splitted_word[target_idx1] = target_char2
    splitted_word[target_idx2] = target_char1
    qa_text = "".join(splitted_word)
    return qa_text

def noise_addition(word):
    splitted_word = list(word)
    target_idx = random.choice(range(len(splitted_word)))
    #splitted_word[target_idx] = splitted_word[target_idx]*2
    splitted_word[target_idx] = splitted_word[target_idx] + chr(random.choice(range(97, 123)))
    qa_text = "".join(splitted_word)
    return qa_text

def noise_random(word):
    splitted_word = list(word)
    target_idx = random.choice(range(len(splitted_word)))
    splitted_word[target_idx] = chr(random.choice(range(97, 123)))
    qa_text = "".join(splitted_word)
    return qa_text

def noise_decompose(word):
    #qa_text = f"-".join(list(word))
    splitted_word = list(word)
    target_idx = random.choice(range(len(splitted_word)-1))
    splitted_word[target_idx] = splitted_word[target_idx] + "-"
    qa_text = "".join(splitted_word)
    return qa_text

def noise_uppercase(word):
    #qa_text = word.upper()
    splitted_word = list(word)
    target_idx = random.choice(range(len(splitted_word)))
    splitted_word[target_idx] = splitted_word[target_idx].upper()
    qa_text = "".join(splitted_word)
    return qa_text

def noise_mask(word):
    splitted_word = list(word)
    target_idx = random.choice(range(len(splitted_word)))
    splitted_word[target_idx] = "_"
    qa_text = "".join(splitted_word)
    return qa_text

def make_word_noise(phase_flag, word=None, task_type=None):

  ##############################
  # For inference
  ##############################    
  if phase_flag == "inference":

    ### In-distribution
    if task_type == 1: # Delete
        qa_text = noise_delete(word)
    elif task_type == 2: # Swap
        qa_text = noise_swap(word)
    elif task_type == 3: # Addition
        qa_text = noise_addition(word)
    elif task_type == 4: # Random
        qa_text = noise_random(word)
    ### Out-of-distribution
    elif task_type == 5: # Decomposition
        qa_text = noise_decompose(word)
    elif task_type == 6: # Uppercase
        qa_text = noise_uppercase(word)
    elif task_type == 7: # Mask
        qa_text = noise_mask(word)
    else:
        raise ValueError("error! task_type is not properly defined!")

  ##############################
  # For training
  ##############################    
  elif phase_flag == "training":

    if task_type is None:
        task_type = random.randint(1, 4)
    
    if task_type == 1: # Delete
        qa_text = noise_delete(word)
    elif task_type == 2: # Swap
        qa_text = noise_swap(word)
    elif task_type == 3: # Addition
        qa_text = noise_addition(word)
    elif task_type == 4: # Random
        qa_text = noise_random(word)
    else:
        raise ValueError("error! task_type is not properly defined!")

  ##############################
  # else
  ##############################        
  else:
    print("ng!")

  return qa_text

def make_text_noise(args, text):

    if args.noise_type == "multi_id":
        task_type=random.randint(1, 4)
    elif args.noise_type == "multi_ood":
        task_type=random.randint(5, 7)
    elif args.noise_type == "typoglycemia":
        task_type=-2
    elif args.noise_type == "nothing":
        task_type=-1
    elif args.noise_type == "drop":
        task_type=1
    elif args.noise_type == "swap":
        task_type=2
    elif args.noise_type == "addition":
        task_type=3
    elif args.noise_type == "randomcase":
        task_type=4    
    elif args.noise_type == "decomposition":
        task_type=5
    elif args.noise_type == "uppercase":
        task_type=6
    elif args.noise_type == "mask":
        task_type=7
    else:
        raise ValueError("noise_type is not properly defined ...")
    
    if task_type == -1: # nothing
        noisy_text = text
    elif task_type == -2: # typoglycemia
        text_list = text.split()
        noisy_text_list = []
        for t in text_list:
            if len(t) >= 4 and random.random() <= args.noise_frequency_test:
                noisy_word = scramble_word_keepfirstlast(t)
            else:
                noisy_word = t
            noisy_text_list.append(noisy_word)
        noisy_text = " ".join(noisy_text_list)
    else:
        text_list = text.split()
        noisy_text_list = []
        for t in text_list:
            if len(t) >= 3 and random.random() <= args.noise_frequency_test:
                noisy_word = make_word_noise(phase_flag="inference", word=t, task_type=task_type)
            else:
                noisy_word = t
            noisy_text_list.append(noisy_word)
        noisy_text = " ".join(noisy_text_list)
    
    return noisy_text

# https://qiita.com/north_redwing/items/1e153139125d37829d2d
def prepare_data_loader(args, ds, batch_size, shuffle):

    fix_seed(args.random_seed)
    worker_seed = args.random_seed
    
    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(worker_seed)

    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=10, shuffle=shuffle, 
                             worker_init_fn=seed_worker, generator=g)
    
    return data_loader

def save_model(args, model, tokenizer, global_steps):
    trained_param_dir = get_trained_param_dir(args, global_steps)
    print(trained_param_dir)
    model.save_pretrained(trained_param_dir) #, safe_serialization=True)
    tokenizer.save_pretrained(trained_param_dir)
    print("save_model finished")

def save_optimizer(args, optimizer, global_steps):
    trained_param_dir = get_trained_param_dir(args, global_steps)
    optimizer_path = os.path.join(trained_param_dir, 'optimizer.pth')
    print(optimizer_path)
    torch.save({'optimizer_state_dict': optimizer.state_dict()}, optimizer_path)
    
def load_optimizer(args, model):
    
    params = [param for param in model.parameters() if param.requires_grad == True]
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.01)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)
    else:
        raise ValueError("error! optimizer is not properly defined!")
    
    if args.restore_steps is not None:
        trained_param_dir = get_trained_param_dir(args, args.restore_steps)
        optimizer_path = os.path.join(trained_param_dir, 'optimizer.pth')
        optimizer.load_state_dict(torch.load(optimizer_path)['optimizer_state_dict'])
        print("restoring optimizer state completed!")
    
    return optimizer

def get_trained_param_dir(args, global_steps):
    model_name = args.model_name
    model_name = model_name.replace("/", "_")
    trained_param_dir = os.path.join(args.disk_path, 'models', f'trained_param_{model_name}_{args.trained_param_file_suffix}_{global_steps}')    
    print(trained_param_dir)
    return trained_param_dir

def load_model(args, inference_mode=True):

    hf_cache_path = os.path.join(args.disk_path, 'huggingface')
    
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    
    # pip install flash-attn --no-build-isolation
    # https://github.com/Dao-AILab/flash-attention
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 torch_dtype=args.model_dtype,
                                                 cache_dir=hf_cache_path,
                                                 device_map = "balanced",
                                                 token=args.HUGGINGFACE_TOKEN,
                                                 attn_implementation="flash_attention_2",
                                                )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=hf_cache_path, token=args.HUGGINGFACE_TOKEN)
    tokenizer.padding_side = "right"
    
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=hf_cache_path, token=args.HUGGINGFACE_TOKEN)
    
    if inference_mode:
        print("inference mode")
        # Restore parameters from checkpoint ...
        if args.restore_steps is not None:
            assert args.trained_param_file_suffix is not None, f'args.trained_param_file_suffix is not defined!'
            from peft import PeftModel, PeftConfig
            trained_param_dir = get_trained_param_dir(args, args.restore_steps)
            model = PeftModel.from_pretrained(model, trained_param_dir, is_trainable=False)
            #model = model.merge_and_unload(safe_merge=True) # Warning! Performance degrades if this is enabled !!!!            
            model = model.to(args.model_dtype)
            print("loading peft completed!")

        # https://huggingface.co/docs/peft/developer_guides/troubleshooting
        # PEFT documents: Troubleshooting
        # If your model outputs are not exactly the same as previous runs, there could be an issue with random elements.
        # please ensure it is in .eval() mode, which is important, for instance, if the model uses dropout
        model.eval()
    else:
        print("training mode")
        #model.train()
        # Newly start trainig ...
        if args.restore_steps is None:
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                     inference_mode=False,
                                     r=args.lora_r,
                                     lora_alpha=args.lora_alpha,
                                     lora_dropout=args.lora_dropout,
                                     bias="none",
                                     target_modules=args.train_target_modules,
                                     )
            model = get_peft_model(model, peft_config)
            model = model.to(args.model_dtype)
        # Restore from checkpoint and continue training ...
        else:
            from peft import PeftModel, PeftConfig
            trained_param_dir = get_trained_param_dir(args, args.restore_steps)
            model = PeftModel.from_pretrained(model, trained_param_dir, is_trainable=True)
            model = model.to(args.model_dtype)
            print("lora parameter restoration completed!")

        # Dropout is automatically enabled even when loading from checkpoint.
        # inference_mode is also set as False when loading from checkpoint.
        print(model)
        print(model.peft_config)
        
        if args.bpe_dropout:
            print(tokenizer._tokenizer.model)
            tokenizer._tokenizer.model.dropout = 0.1
            print(tokenizer._tokenizer.model)
            text = "Iamacatwomanaaaaaaaaaaaaaaaaaaaaaaaaa."
            for _ in range(10):
                print(tokenizer.tokenize(text))
    
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    #print(tokenizer.eos_token_id)
    
    if inference_mode:
        model = torch.compile(model)
        print("torch.compile completed!")

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    print("load_model finished")    
    return model, tokenizer, config