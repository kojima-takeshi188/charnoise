from datasets import load_dataset
import math

from utils import *

def eval_ppl(args):

    cache_dir = args.dataset_cache_dir
    
    if args.eval_ppl_dataset == "fineweb":    
        ds = load_dataset("HuggingFaceFW/fineweb", data_files="sample/10BT/001_00000.parquet", cache_dir=cache_dir)
        ds = ds['train']['text']
    elif args.eval_ppl_dataset == "redpajama":
        ds = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample", cache_dir=cache_dir, trust_remote_code=True)
        ds = ds['train']['raw_content']
    else:
        raise ValueError("error! eval_ppl_dataset is not properly defined!")
    
    valid_loader = prepare_data_loader(args, ds, batch_size=1, shuffle=False)

    model, tokenizer, config = load_model(args)
    
    losses = []
    j = 0
    for data in valid_loader:
      
      if data[0] == "" and len(data[0].split(" ")) <= 10:
        continue
      else:
        j += 1
      
      encoded_input = tokenizer(data, return_tensors='pt', padding=True, 
                                truncation=True, max_length=args.max_context_length)['input_ids'].to(args.device)
      encoded_label = torch.where(encoded_input == tokenizer.pad_token_id, -100, encoded_input)
      
      with torch.no_grad():
        loss = model(input_ids=encoded_input, labels=encoded_label).loss
        ppl = math.exp(loss.cpu().item())
        losses.append(ppl)
      
      if j >= args.eval_dataset_size:
        break
    
    average_loss = sum(losses)/len(losses)
    print(average_loss)

    result = f"{args.eval_ppl_dataset}; {average_loss}"
    return result
