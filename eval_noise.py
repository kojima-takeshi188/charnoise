import random, string
import math
import collections
import sys
from datasets import load_from_disk

import numpy as np
from sklearn.metrics import f1_score

from utils import *

def evaluation(args, data_loader, model, tokenizer, train_loader):

  total = 0
  correct_list = []
  pred_list, gt_list = [], []
  pred_list2, gt_list2 = [], []
  print(len(data_loader))
  dataiter = iter(train_loader)

  for j, data in enumerate(data_loader):

    # Set an answer ...      
    output_idx = data[args.output_lable][0].item()

    # Balance labels ...
    if args.balance_label_flag:
        if (total % args.num_class) != output_idx:
            continue
      
    # Set a main question ...
    input_text = data[args.input_lable][0]
    input_text = tokenizer.tokenize(input_text)[:args.num_text_limit]
    input_text = tokenizer.convert_tokens_to_ids(input_text)
    input_text = tokenizer.decode(input_text)
    
    # Create in-context prompts ...
    try:
        sample_set = set_incontext_examples(args, dataiter, tokenizer)
    except StopIteration:
        dataiter = iter(train_loader)
        sample_set = set_incontext_examples(args, dataiter, tokenizer)
    
    samples = ""
    for sample in sample_set:
        samples += "".join(sample)
    if j < 3:
        print("*******************")
        print(samples)
        print("*******************")
    
    q = args.prefix1 + " " + input_text + "\n" + args.prefix2
    a = args.label_list[output_idx]

    # Concat demo + main question ...
    input_batch = args.instruction + samples + q
    encoded_input = tokenizer(input_batch, return_tensors='pt', padding=True, truncation=True).to(args.device)
    encoded_input_token_num = encoded_input['input_ids'].shape[-1]
    print(encoded_input_token_num)

    # Model inference ...
    with torch.no_grad():
        #print(encoded_input)
        output = model.generate(**encoded_input, max_new_tokens=10,
                                pad_token_id=tokenizer.eos_token_id,
                                return_dict_in_generate=True,
                                output_scores=True,
                                do_sample=False, top_p=1, temperature=0
                              )
    decoded_output = tokenizer.decode(output['sequences'][0][encoded_input_token_num:], skip_special_tokens=True)
    pred = decoded_output.split("\n")[0].strip("\n").strip(" ")

    if total % 50 == 0:
        if total == 0:
            print('************************************')
            print(tokenizer.decode(output['sequences'][0], skip_special_tokens=True))
        print(f'No: {j}, lable: {a}, pred: {pred}')

    pred_list.append(pred)
    gt_list.append(a)
    
    try:
      pred2 = args.label_list.index(pred)
    except:
      pred2 = len(args.label_list)
    
    pred_list2.append(pred2)
    gt_list2.append(output_idx)
    
    correct_list.append(pred == a)
    
    total += 1

    if total >= args.eval_dataset_size:
      print("evaluation break...")
      break

  print(f'pred_list: {pred_list}')
  print(f'gt_list: {gt_list}')
  print(f'pred_list2: {pred_list2}')
  print(f'gt_list2: {gt_list2}')

  f1_list = f1_score(gt_list2, pred_list2, average=None)
  f1_list = f1_list[:len(args.label_list)]
  f1_macro = 100 * sum(f1_list) / len(f1_list)
  f1_macro_check = 100 * f1_score(gt_list2, pred_list2, average='macro')
  print(f'f1_list: {f1_list}')
  print(f'f1_macro: {f1_macro}')
  print(f'f1_macro_check: {f1_macro_check}')

  accuracy = (np.sum(np.array(correct_list)) * 1.0 / total) * 100
  statistics_pred = collections.Counter(pred_list)
  statistics_gt = collections.Counter(gt_list)
  print(f'accuracy: {np.round(accuracy, decimals=1)}')
  print(f'statistics_pred: {statistics_pred}')
  print(f'statistics_gt: {statistics_gt}')  
  result = f"{args.eval_noise_dataset}; {args.noise_type}; {args.noise_frequency_test}; {accuracy}; {statistics_pred}"
  return result

# Set In-Context examples from Training dataset...
def set_incontext_examples(args, dataiter, tokenizer):
    sample_set = []
    for j in range(10000):
      data = next(dataiter)

      output_idx = data[args.output_lable][0].item()

      # Limit maximum word length in a text to fixed size ...
      input_text = data[args.input_lable][0]
      input_text = tokenizer.tokenize(input_text)[:args.num_text_limit]
      input_text = tokenizer.convert_tokens_to_ids(input_text)
      input_text = tokenizer.decode(input_text)
      
      sample = args.prefix1 + " " + input_text + "\n" + args.prefix2 + " " + args.label_list[output_idx] + "\n\n"

      # To balance the number of each class sample ...
      if len(sample_set) % args.num_class == output_idx:
          sample_set.append(sample)

      if args.incontex_sample_size == 0:
          sample_set = []
        
      if len(sample_set) >= args.incontex_sample_size:
          break
    
    random.shuffle(sample_set)
    return sample_set

class Typo_Injector:
    def __init__(self, args, threshold):
        self.typo_list = {}
        file_path = os.path.join(args.dataset_cache_dir, "custom_typo_injected", "spell-errors.txt")
        with open(file_path) as f:
            lines = [s.strip() for s in f.readlines()]
            for line in lines:
                key_value = line.split(": ")
                key = key_value[0].strip()
                if key == "?":
                    continue
                values = [v.split("*")[0].strip() for v in key_value[1].split(", ")]
                self.typo_list[key] = values
        self.typo_count = 0
        self.total_word_count = 0
        self.args = args
        self.threshold = threshold

    def typo_inject(self, example):
      input_text = example[self.args.input_lable]
      
      input_word_list = input_text.split(" ")
      output_word_list = copy.deepcopy(input_word_list)
      
      for i in range(len(input_word_list)):
          if random.random() <= self.threshold:
              try:
                  values = self.typo_list[input_word_list[i]]
                  value = random.choice(values)
                  output_word_list[i] = value
                  self.typo_count += 1
              except:
                  pass
                  
      output_text = " ".join(output_word_list)
        
      print(input_text)
      print(output_text)

      self.total_word_count += len(input_word_list)

      example[self.args.input_lable] = output_text
        
      return example

class Noise_Injector:
    def __init__(self, args):
        self.args = args

    def noise_inject(self, example):
      input_text = example[self.args.input_lable]                  
      output_text = make_text_noise(self.args, input_text)
      
      print(input_text)
      print(output_text)

      example[self.args.input_lable] = output_text
        
      return example

def eval_noise(args, model=None, tokenizer=None):

    fix_seed(args.random_seed)

    if model is None and tokenizer is None:
        model, tokenizer, _ = load_model(args)
    
    if tokenizer is not None:
      tokenizer.pad_token = tokenizer.eos_token

    if args.eval_noise_dataset == "google/jigsaw_toxicity_pred":
        data_dir = os.path.join("jigsaw-toxic-comment-classification-challenge")
    else:
        data_dir = None
    print(data_dir)
    
    print(args.dataset_cache_dir)

    ds = load_dataset(args.eval_noise_dataset,
                      data_dir=data_dir,
                      cache_dir=args.dataset_cache_dir, 
                      trust_remote_code=True)

    # https://huggingface.co/docs/datasets/v1.2.0/processing.html#processing-data-row-by-row
    ###########################
    # injection of typo ...
    ###########################
    if args.injection_typo_flag:
      
      ds_dir = os.path.join(args.dataset_cache_dir, "custom_typo_injected", 
                            args.eval_noise_dataset.replace("/", "_"))
      print(ds_dir)
            
      if not os.path.isdir(ds_dir):
          
          num_samples = min(10000, len(ds[args.ds_train_lable][args.input_lable]))
          ratio = num_samples / len(ds[args.ds_train_lable])
          if ratio < 1:
              ds[args.ds_train_lable] = ds[args.ds_train_lable].train_test_split(test_size=ratio)["test"]
          
          num_samples = min(10000, len(ds[args.ds_test_lable][args.input_lable]))
          ratio = num_samples / len(ds[args.ds_test_lable])
          if ratio < 1:
              ds[args.ds_test_lable] = ds[args.ds_test_lable].train_test_split(test_size=ratio)["test"]

          # Check typo_percentage upperbound.
          typo_injector = Typo_Injector(args, threshold=1.0)          
          _ = ds[args.ds_train_lable].map(typo_injector.typo_inject)
          _ = ds[args.ds_test_lable].map(typo_injector.typo_inject)
          typo_percentage = 100 * typo_injector.typo_count / typo_injector.total_word_count
          print(f"typo_percentage: {typo_percentage}")

          # for 50 percent replacement
          threshold = 50 / typo_percentage
          
          # Replacement.
          typo_injector = Typo_Injector(args, threshold=threshold)          
          ds[args.ds_train_lable] = ds[args.ds_train_lable].map(typo_injector.typo_inject)
          ds[args.ds_test_lable] = ds[args.ds_test_lable].map(typo_injector.typo_inject)
          typo_percentage = 100 * typo_injector.typo_count / typo_injector.total_word_count
          print(f"typo_percentage: {typo_percentage}")
          
          ds.save_to_disk(ds_dir)

          # to align random seed
          sys.exit()
    
    ###########################
    # injection of artificial noise ...
    ###########################
    else:
      ds_dir = os.path.join(args.dataset_cache_dir, "custom_noise_injected", 
                            args.eval_noise_dataset.replace("/", "_"), args.noise_type, str(args.noise_frequency_test))
      print(ds_dir)
            
      if not os.path.isdir(ds_dir):
          
          num_samples = min(10000, len(ds[args.ds_train_lable][args.input_lable]))
          ratio = num_samples / len(ds[args.ds_train_lable])
          if ratio < 1:
              ds[args.ds_train_lable] = ds[args.ds_train_lable].train_test_split(test_size=ratio)["test"]
          
          num_samples = min(10000, len(ds[args.ds_test_lable][args.input_lable]))
          ratio = num_samples / len(ds[args.ds_test_lable])
          if ratio < 1:
              ds[args.ds_test_lable] = ds[args.ds_test_lable].train_test_split(test_size=ratio)["test"]
          
          # Replacement.
          noise_injector = Noise_Injector(args)          
          ds[args.ds_train_lable] = ds[args.ds_train_lable].map(noise_injector.noise_inject)
          ds[args.ds_test_lable] = ds[args.ds_test_lable].map(noise_injector.noise_inject)
          
          ds.save_to_disk(ds_dir)
          
          # to align random seed
          sys.exit()
      
    ds = load_from_disk(ds_dir)
    
    train_loader = prepare_data_loader(args, ds[args.ds_train_lable], batch_size=1, shuffle=True)
    test_loader = prepare_data_loader(args, ds[args.ds_test_lable], batch_size=1, shuffle=True)
    
    result = evaluation(args, test_loader, model, tokenizer, train_loader)
    return result