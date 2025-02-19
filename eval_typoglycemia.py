import random, string
import math
import collections
from torch.utils.data import Dataset
import json
from utils import *

import Levenshtein

def evaluation(args, test_loader, model, tokenizer, train_loader):

  total = 0
  print(len(test_loader))
  train_dataiter = iter(train_loader)
  results1, results2 = [], []
  
  for j, (srs_text, tgt_text) in enumerate(test_loader):
    srs_text, tgt_text = srs_text[0], tgt_text[0]
    
    # Create in-context prompts ...
    if j % int(len(train_loader) / args.incontex_sample_size) == 0:
        train_dataiter = iter(train_loader)
    
    sample_set = set_incontext_examples(args, train_dataiter, tokenizer)
    samples = ""
    for sample in sample_set:
        samples += "".join(sample)
    if j < 3:
        print("*******************")
        print(samples)
        print("*******************")
    
    q = args.prefix1 + " " + srs_text + "\n" + args.prefix2
    a = tgt_text

    # Concat demo + main question ...
    input_batch = args.instruction + samples + q
    encoded_input = tokenizer(input_batch, return_tensors='pt', padding=True, truncation=True).to(args.device)
    encoded_input_token_num = encoded_input['input_ids'].shape[-1]
    #print(encoded_input_token_num)

    # Model inference ...
    with torch.no_grad():
        output = model.generate(**encoded_input, 
                                max_new_tokens=args.num_text_limit,
                                do_sample=False, top_p=1, temperature=0
                               )
    output = tokenizer.decode(output[0, encoded_input_token_num:], skip_special_tokens=True)
    pred = output.split("\n")[0].strip("\n").strip(" ")
    result1 = (pred==a)
    result2 = Levenshtein.distance(pred, a)

    results1.append(result1)
    results2.append(result2)
      
    if total % 10 == 0:
        print("check:")
        print(input_batch)
        print("###")
        print(output)
        print("###")
        print(pred)
        print(a)
        print(result1)
        print(result2)

    total += 1
    
    if total >= args.eval_dataset_size:
      print("evaluation break...")
      break

  accuracy = 100 * sum(results1) / len(results1)
  print(f"accuracy: {accuracy}")
  distance = sum(results2) / len(results2)
  print(f"distance: {distance}")
  result = f"{args.eval_noise_dataset}; {args.noise_type}; {accuracy}; {distance}"
  return result

# Set In-Context examples from Training dataset...
def set_incontext_examples(args, dataiter, tokenizer):
    sample_set = []
    for j in range(10000):
      srs_text, tgt_text = next(dataiter)
      srs_text, tgt_text = srs_text[0], tgt_text[0]
        
      sample = args.prefix1 + " " + srs_text + "\n" + args.prefix2 + " " + tgt_text + "\n\n"
      sample_set.append(sample)
      
      if len(sample_set) >= args.incontex_sample_size:
          break
    
    random.shuffle(sample_set)
    return sample_set

class MyDataset(Dataset):
    def __init__(self, data_dir, tokenizer, args):
        super().__init__()
        with open(data_dir) as f:
            self.data = [l.strip("\n").strip(" ") for l in f.readlines()]
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        tgt_text = self.data[index]
        tgt_text = self.tokenizer.tokenize(tgt_text)[:self.args.num_text_limit]
        tgt_text = self.tokenizer.convert_tokens_to_ids(tgt_text)
        tgt_text = self.tokenizer.decode(tgt_text)
        srs_text = make_text_noise(self.args, tgt_text) # ★★★★★        
        return srs_text, tgt_text

def eval_typoglycemia(args):

    fix_seed(args.random_seed)

    model, tokenizer, config = load_model(args)
    
    if tokenizer is not None:
      tokenizer.pad_token = tokenizer.eos_token

    data_dir = os.path.join("flores200_dataset", "devtest", "eng_Latn.devtest")
    print(data_dir)

    ds = MyDataset(data_dir=data_dir, tokenizer=tokenizer, args=args)

    n_samples = len(ds)
    train_size = int(len(ds) * 0.5)
    val_size = n_samples - train_size

    gen = torch.Generator()
    gen.manual_seed(args.random_seed)
    train_set, test_set = torch.utils.data.random_split(ds, [train_size, val_size], generator=gen)
    
    train_loader = prepare_data_loader(args, train_set, batch_size=1, shuffle=True)
    test_loader =  prepare_data_loader(args, test_set, batch_size=1, shuffle=True)
    
    result = evaluation(args, test_loader, model, tokenizer, train_loader)
    return result