from nltk import probability
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
import copy
import wandb
import re
import sys
import torch.utils.checkpoint

from utils import *
from eval_noise import *

def validation_loss(args, data_valid, model, tokenizer):

  loss_valid = 0
  num_samples_valid = len(data_valid)
  for k in range(num_samples_valid):

      encoded_input = tokenizer(data_valid[k], return_tensors='pt', padding=True, truncation=True, 
                                max_length=args.max_context_length).to(args.device)
      
      input_ids = encoded_input.input_ids
      attention_mask = encoded_input.attention_mask
      labels = copy.deepcopy(encoded_input.input_ids)
      labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
      
      with torch.no_grad():
          loss = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, use_cache=args.use_cache).loss
          loss = loss / num_samples_valid
          loss_valid += loss.detach().cpu().item()
          
  with torch.no_grad():
      args.eval_dataset_size = 200
      args.balance_label_flag = True
      args.noise_type = "multi_id"
      accuracy_id = float(eval_noise(args, model, tokenizer).split(";")[3].strip())
      print(f'accuracy_id: {accuracy_id}')
  
  return loss_valid

def train(args):

    assert args.trained_param_file_suffix is not None, f'args.trained_param_file_suffix is not defined!'

    global_step = 0
    
    #######################################
    # Setup for Training
    #######################################
    model, tokenizer, config = load_model(args, inference_mode=False)

    try:
        bos_token = tokenizer.bos_token
    except:
        bos_token = tokenizer.eos_token
    
    # To reduce memory...
    model.gradient_checkpointing_enable()

    model.print_trainable_parameters()
    
    for name, param in model.named_parameters():
          print(f"{name} : {param.size()} : {param.requires_grad} : {param.dtype}: device{param.get_device()}: {torch.sum(param)}")
    
    optimizer = load_optimizer(args, model)
    
    celoss_sum = torch.nn.CrossEntropyLoss(reduction='sum')
    
    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d%H%M%S')
    with wandb.init(project="charnoise", name=f"{args.model_name}_{args.trained_param_file_suffix}_{now}"):

      ds = load_dataset("HuggingFaceFW/fineweb", data_files="sample/10BT/000_00000.parquet", 
                        cache_dir=os.path.join(args.disk_path, "datasets"))
      print(ds)
      train_loader = prepare_data_loader(args, ds["train"]["text"], batch_size=1, shuffle=True)
      dataiter = iter(train_loader)
      
      # prepare validation datasets
      num_samples_valid = 20
      data_valid = []
      for k in range(num_samples_valid):
          data_valid.append(next(dataiter)[0])
      print(data_valid)

      if args.restore_steps is None:
          loss_valid = validation_loss(args, data_valid, model, tokenizer)
          wandb.log({"global_step": global_step, "loss_valid": loss_valid})
      
      total_token_count = 0
      loss_train = 0
      
      for j in range((args.total_global_step+100)*args.batch_size):

        # =======================
        # Mini-Batch related process
        # =======================
        if ((j+1) % args.mini_batch_size == 1) or (args.mini_batch_size == 1):
            batch_noisy_document_labels = []
            batch_clean_document_labels = []
            batch_input_ids_noisy = []
            batch_input_ids_clean = []
            batch_labels_noisy = []
        
        # =======================
        # Create clean document
        # =======================
        if args.append_bos_flag:
            clean_document = bos_token
        else:
            clean_document = ""
        context_flag = False
        while context_flag == False:
            data_train = next(dataiter)
            clean_doc = data_train[0]
            
            # Process clean document
            clean_doc = clean_doc.strip("\n").strip(" ")
            clean_doc = clean_doc + bos_token

            # Concatenate documents
            clean_document += clean_doc

            # Check context size of the documents            
            if len(tokenizer.tokenize(clean_document)) >= args.max_context_length + 500:
                context_flag = True

        # erase nobreak space
        # https://www.mediaprimestyle.jp/column_post/nbsp_text/
        clean_document = clean_document.replace("\xa0", "")
        
        #encode and decode to avoid trivial errors...
        clean_document = tokenizer.tokenize(clean_document)
        clean_document = tokenizer.convert_tokens_to_ids(clean_document)
        clean_document = tokenizer.decode(clean_document, add_special_tokens=False)
        
        clean_word_list = []
        #for doc in clean_document.split("<bos>"):
        for doc in clean_document.split(bos_token):
            for text in doc.split("\n"):
                for word in text.split(" "):
                    #if word != "":
                    clean_word_list.append(word)

        # =======================
        # Skip iteration for training restore
        # =======================
        if args.restore_steps is not None:
            if j < (args.batch_size * args.restore_steps):
                if (j + 1) % args.batch_size == 0:
                    global_step += 1
                    print(f"{global_step}: skip iteration")
                continue

        # =======================
        # Create noisy document
        # =======================
        noisy_document = []
        noisy_word_list = []
          
        #for doc in clean_document.split("<bos>"):
        for doc in clean_document.split(bos_token):
            if random.random() <= 0.5:
                noise_frequency = 0
            else:
                noise_frequency = random.random() * args.noise_frequency_train
            noisy_doc = []
            for text in doc.split("\n"):
                noisy_text = []
                for word in text.split(" "):
                    if random.random() < noise_frequency and len(word) >= 3 and word.isascii() == True:
                        noisy_word = make_word_noise(phase_flag="training", word=word)
                    else:
                        noisy_word = word
                    #if word != "":
                    noisy_word_list.append(noisy_word)
                    noisy_text.append(noisy_word)
                noisy_text = " ".join(noisy_text)
                noisy_doc.append(noisy_text)
            noisy_doc = "\n".join(noisy_doc)
            noisy_document.append(noisy_doc)
        noisy_document = bos_token.join(noisy_document)

        # =======================
        # Create word list
        # =======================
        assert len(clean_word_list) == len(noisy_word_list), f'num of total word error! {len(clean_word_list)}, {len(noisy_word_list)}'

        # =======================
        # Create noisy label
        # =======================
        noisy_document_tokens = tokenizer.tokenize(noisy_document)
        noisy_document_labels = [1] * len(noisy_document_tokens)       
        start_idx = 0
        end_idx = 1
        for noisy_word, clean_word in zip(noisy_word_list, clean_word_list):
          ok_flag = False
          while ok_flag == False:
            concat_word = tokenizer.convert_tokens_to_ids(noisy_document_tokens[start_idx:end_idx])
            concat_word = tokenizer.decode(concat_word, add_special_tokens=False)
            if noisy_word.replace(" ", "") in concat_word.replace(" ", ""):
              if noisy_word != clean_word:
                for idx in range(start_idx, end_idx):
                  noisy_document_labels[idx] = 0
                # to avoid error...
                if end_idx < len(noisy_document_tokens):
                  if noisy_document_tokens[end_idx] == "Ċ":
                    noisy_document_labels[end_idx] = 0
              ok_flag = True
            else:
              end_idx += 1
              if end_idx > len(noisy_document_tokens):
                raise ValueError(f"stop2!!\nnoisy_document_tokens: {noisy_document_tokens}\nnoisy_word_list:{noisy_word_list}\nnoisy_word: {noisy_word}\nconcat_word: {concat_word}\nstart_idx: {start_idx}, end_idx: {end_idx}")
          if noisy_word != "":
              start_idx = end_idx
              end_idx = start_idx + 1
        
        check_noisy = [t for t, l in zip(noisy_document_tokens, noisy_document_labels) if l == 1]
        
        input_ids_noisy = tokenizer.convert_tokens_to_ids(noisy_document_tokens)
        input_ids_noisy = torch.tensor(input_ids_noisy).to(args.device)
        input_ids_noisy = input_ids_noisy[:args.max_context_length]
        #input_ids_noisy = input_ids_noisy.unsqueeze(0)

        # for ours_celoss method...
        labels_noisy = torch.tensor(noisy_document_labels).to(args.device)
        labels_noisy = labels_noisy[:args.max_context_length]
        #labels_noisy = labels_noisy.unsqueeze(0)
        labels_noisy = input_ids_noisy * labels_noisy
        labels_noisy = torch.where(labels_noisy == 0, -100, labels_noisy)
          
        noisy_document_labels = noisy_document_labels[:args.max_context_length]
        noisy_document_labels = noisy_document_labels[1:]
        noisy_document_labels = [n for n, v in enumerate(noisy_document_labels) if v == 1]
        
        # =======================
        # Create clean label
        # =======================
        clean_document_tokens = tokenizer.tokenize(clean_document)
        clean_document_labels = [1] * len(clean_document_tokens)      
        start_idx = 0
        end_idx = 1
        for noisy_word, clean_word in zip(noisy_word_list, clean_word_list):
          ok_flag = False
          while ok_flag == False:
            concat_word = tokenizer.convert_tokens_to_ids(clean_document_tokens[start_idx:end_idx])
            concat_word = tokenizer.decode(concat_word, add_special_tokens=False)
            if clean_word.replace(" ", "") in concat_word.replace(" ", ""):
              if noisy_word != clean_word:
                for idx in range(start_idx, end_idx):
                  clean_document_labels[idx] = 0
                # to avoid error...
                if end_idx < len(clean_document_tokens):
                  if clean_document_tokens[end_idx] == "Ċ":
                    clean_document_labels[end_idx] = 0                  
              ok_flag = True
            else:
              end_idx += 1
              if end_idx > len(clean_document_tokens):
                raise ValueError(f"stop2!!\nclean_document: {clean_document}\nclean_document_tokens: {clean_document_tokens}\nclean_word_list:{clean_word_list}\nclean_word: {clean_word}\nconcat_word: {concat_word}\nstart_idx: {start_idx}, end_idx: {end_idx}")
          if clean_word != "":
              start_idx = end_idx
              end_idx = start_idx + 1
        
        check_clean = [t for t, l in zip(clean_document_tokens, clean_document_labels) if l == 1]

        input_ids_clean = tokenizer.convert_tokens_to_ids(clean_document_tokens)
        input_ids_clean = torch.tensor(input_ids_clean).to(args.device)
        input_ids_clean = input_ids_clean[:args.max_context_length]
        #input_ids_clean = input_ids_clean.unsqueeze(0)

        clean_document_labels = clean_document_labels[:args.max_context_length]
        clean_document_labels = clean_document_labels[1:]
        clean_document_labels = [n for n, v in enumerate(clean_document_labels) if v == 1]
        
        # =======================
        # Sample check
        # =======================
        if not args.bpe_dropout:
            #if (j <= 3) or (check_clean != check_noisy):
            if (j <= 3) or (len(check_clean) != len(check_noisy)):                
                print(j)
                print("====================")
                print(clean_document)
                print(clean_word_list)            
                print(clean_document_tokens)
                print(clean_document_labels)
                print(check_clean)
                print(input_ids_clean)
                print("====================")
                print(noisy_document)
                print(noisy_word_list)
                print(noisy_document_tokens)
                print(noisy_document_labels)
                print(check_noisy)
                print(input_ids_noisy)
                print(labels_noisy)            
                print("====================")
    
            #if check_clean != check_noisy:
            if len(check_clean) != len(check_noisy):                
                print("check_clean is not equal to check_noisy! Skip ...")
                continue
            #assert check_clean == check_noisy, f'target token error! See check_clean and check_noisy.'

        # =======================
        # Create Mini-Batch
        # =======================
        #noisy_document_labels -> list
        #clean_document_labels -> list
        #input_ids_noisy -> tensor
        #input_ids_clean -> tensor
        #labels_noisy -> tensor
        batch_noisy_document_labels.append(noisy_document_labels)
        batch_clean_document_labels.append(clean_document_labels)
        batch_input_ids_noisy.append(input_ids_noisy)
        batch_input_ids_clean.append(input_ids_clean)
        batch_labels_noisy.append(labels_noisy)

        if (j+1) % args.mini_batch_size == 0:
            batch_input_ids_noisy = torch.stack(batch_input_ids_noisy)
            batch_input_ids_clean = torch.stack(batch_input_ids_clean)
            batch_labels_noisy = torch.stack(batch_labels_noisy)
        else:
            continue

        # =======================
        # Calculate Loss
        # =======================
        if args.train_method == "ours":
            logits = model(input_ids=batch_input_ids_noisy, use_cache=args.use_cache).logits
            logits = logits.permute(0, 2, 1) # [b, t, v] -> [b, v, t]
            with torch.no_grad():
                model.disable_adapter_layers()
                logits2 = model(input_ids=batch_input_ids_clean, use_cache=args.use_cache).logits
                logits2 = logits2.permute(0, 2, 1) # [b, t, v] -> [b, v, t]
                model.enable_adapter_layers()
            loss = 0
            for k in range(len(batch_noisy_document_labels)):
                noisy_document_labels = batch_noisy_document_labels[k]
                clean_document_labels = batch_clean_document_labels[k]
                token_count = min(len(noisy_document_labels), len(clean_document_labels))
                total_token_count += token_count
                logit = logits[k:k+1, :, noisy_document_labels[:token_count]]
                logit2 = logits2[k:k+1, :, clean_document_labels[:token_count]]
                prob = torch.nn.Softmax(dim=1)(logit2)
                loss += celoss_sum(logit, prob)
        elif args.train_method == "baseline1":
            loss = model(input_ids=batch_input_ids_noisy, labels=batch_labels_noisy, use_cache=args.use_cache).loss
        elif args.train_method == "baseline2":
            loss = model(input_ids=batch_input_ids_noisy, labels=batch_input_ids_noisy, use_cache=args.use_cache).loss
        elif args.train_method == "baseline3":
            loss = model(input_ids=batch_input_ids_clean, labels=batch_input_ids_clean, use_cache=args.use_cache).loss            
        else:
            raise ValueError("error! train_method is not properly defined!")
        
        if args.train_method != "ours":
            loss = loss / (args.batch_size / args.mini_batch_size)
        
        loss.backward()
        loss_train += loss.detach().cpu().item()
            
        if (j + 1) % args.batch_size == 0:

          #####################################          
          # Adjust gradient of loss
          #####################################
          if args.train_method == "ours":
              print(total_token_count)
              for name, param in model.named_parameters():
                if param.requires_grad == True:
                  param.grad = param.grad / total_token_count

          nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
          optimizer.step()
          optimizer.zero_grad()
          
          global_step += 1
          
          if args.train_method == "ours":
              wandb.log({"global_step": global_step, "loss": loss_train/total_token_count})
              print(f'global step: {global_step}, loss: {loss_train/total_token_count}')
          else:
              wandb.log({"global_step": global_step, "loss": loss_train})
              print(f'global step: {global_step}, loss: {loss_train}')
                    
          if (global_step) % args.save_interval == 0:
              #####################################
              # Validation
              #####################################
              loss_valid = validation_loss(args, data_valid, model, tokenizer)
              print(f'global step: {global_step}, loss_valid: {loss_valid}')
              wandb.log({"global_step": global_step, "loss_valid": loss_valid})
              
          if (global_step % args.save_interval == 0) and (args.checkpointing == True):
              save_model(args, model, tokenizer, global_step)
              save_optimizer(args, optimizer, global_step)

          if global_step >= args.total_global_step:
              break
          
          total_token_count = 0
          loss_train = 0
    
    result = "Training completed!"
    print(result)
    return result