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
  
  for j, (srs_text, tgt_text, prob_typo) in enumerate(test_loader):
    srs_text, tgt_text, prob_typo = srs_text[0], tgt_text[0], prob_typo[0]

    if len(tokenizer.tokenize(srs_text)) >= args.num_text_limit:
      continue

    if prob_typo < 0.95:
      continue

    if srs_text == tgt_text:
      continue
    
    # Create in-context prompts ...
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
      
    #if total % 10 == 0:
    if pred != a:
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
  result = f"{args.eval_noise_dataset}; {accuracy}; {distance}"
  return result

# Set In-Context examples from Training dataset...
def set_incontext_examples(args, dataiter, tokenizer):
    sample_set = []
    for j in range(10000):
      srs_text, tgt_text, prob_typo = next(dataiter)
      srs_text, tgt_text, prob_typo = srs_text[0], tgt_text[0], prob_typo[0]
        
      # Limit maximum word length in a text to fixed size ...
      if len(tokenizer.tokenize(srs_text)) >= args.num_text_limit:
        continue

      if prob_typo < 0.95:
        continue

      if srs_text == tgt_text:
        continue
        
      sample = args.prefix1 + " " + srs_text + "\n" + args.prefix2 + " " + tgt_text + "\n\n"
      sample_set.append(sample)
        
      if len(sample_set) >= args.incontex_sample_size:
          break
    
    random.shuffle(sample_set)
    return sample_set

class MyDataset_github(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        with open(data_dir) as f:
            self.data = [json.loads(l) for l in f.readlines()]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        srs_text = self.data[index]["edits"][0]["src"]["text"].strip()
        tgt_text = self.data[index]["edits"][0]["tgt"]["text"].strip()
        try:
            prob_typo = float(self.data[index]["edits"][0]["prob_typo"])
        except:
            prob_typo = -1
        return srs_text, tgt_text, prob_typo

class MyDataset_bea60k(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        with open(data_dir + ".noise") as f:
            self.data1 = [self.preprocess(l) for l in f.readlines()]
            #self.data1 = f.read().splitlines()
        with open(data_dir) as f:
            self.data2 = [self.preprocess(l) for l in f.readlines()]
            #self.data2 = f.read().splitlines()

    def preprocess(self, text):
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" 's", "'s")
        text = text.replace(" 'm", "'m")
        text = text.replace(" n't", "n't")
        #text = text.replace("( ", "(")
        #text = text.replace(" )", ")")
        text = text.replace("``", '"')
        text = text.replace("\n", "")
        return text
    
    def __len__(self):
        return len(self.data1)
    
    def __getitem__(self, index):
        srs_text = self.data1[index]
        tgt_text = self.data2[index]
        prob_typo = 1
        return srs_text, tgt_text, prob_typo

class MyDataset_jfleg(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        # https://github.com/keisks/jfleg
        with open(os.path.join(data_dir, "test.src.txt")) as f:
            data1_1 = [self.preprocess(l) for l in f.readlines()]
        with open(os.path.join(data_dir, "dev.src.txt")) as f:
            data1_2 = [self.preprocess(l) for l in f.readlines()]
        with open(os.path.join(data_dir, "test.spellchecked.src.txt")) as f:
            data2_1 = [self.preprocess(l) for l in f.readlines()]
        with open(os.path.join(data_dir, "dev.spellchecked.src.txt")) as f:
            data2_2 = [self.preprocess(l) for l in f.readlines()]
        self.data1 = data1_1 + data1_2
        self.data2 = data2_1 + data2_2
    
    def preprocess(self, text):
        text = text[0].upper() + text[1:]
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" 's", "'s")
        text = text.replace(" 'm", "'m")
        text = text.replace(" n't", "n't")
        text = text.replace("( ", "(")
        text = text.replace(" )", ")")
        text = text.replace("`` ", '"')
        text = text.replace(" ''", '"')        
        text = text.replace("\n", "")
        return text
    
    def __len__(self):
        return len(self.data1)
    
    def __getitem__(self, index):
        srs_text = self.data1[index]
        tgt_text = self.data2[index]
        prob_typo = 1
        return srs_text, tgt_text, prob_typo

def eval_typo(args):

    fix_seed(args.random_seed)

    model, tokenizer, config = load_model(args)
    
    if tokenizer is not None:
      tokenizer.pad_token = tokenizer.eos_token

    if args.eval_noise_dataset == "github-typo-corpus":
        data_dir = os.path.join("github-typo-corpus", "github-typo-corpus.v1.0.0.jsonl")
        ds = MyDataset_github(data_dir=data_dir)
    elif args.eval_noise_dataset == "bea60k":
        data_dir = os.path.join("bea60k", "test.bea60k")
        ds = MyDataset_bea60k(data_dir=data_dir)
    elif args.eval_noise_dataset == "jfleg":
        data_dir = "jfleg"
        ds = MyDataset_jfleg(data_dir=data_dir)
    
    print(data_dir)
    
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