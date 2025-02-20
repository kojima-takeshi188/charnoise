import random, string
import math
import collections

from utils import *

from rouge_score import rouge_scorer

def evaluation(args, data_loader, model, tokenizer, train_loader):

  total = 0
  print(len(data_loader))
  dataiter = iter(train_loader)
  results1, results2, results3, results4 = [], [], [], []
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)
    
  for j, data in enumerate(data_loader):

    # Set an answer ...      
    output_text = data[args.output_lable][0]
    output_text = tokenizer.tokenize(output_text)[:args.max_new_tokens]
    output_text = tokenizer.convert_tokens_to_ids(output_text)
    output_text = tokenizer.decode(output_text)
    
    # Set a main question ...
    input_text = data[args.input_lable][0] #.split(" ")[:args.num_text_limit])
    input_text = make_text_noise(args, input_text)
    input_text = tokenizer.tokenize(input_text)[:args.num_text_limit]
    input_text = tokenizer.convert_tokens_to_ids(input_text)
    input_text = tokenizer.decode(input_text)
    
    # Create in-context prompts ...
    sample_set = set_incontext_examples(args, dataiter, tokenizer)
    samples = ""
    for sample in sample_set:
        samples += "".join(sample)
    if j < 3:
        print("*******************")
        print(samples)
        print("*******************")
    
    q = args.prefix1 + " " + input_text + "\n" + args.prefix2
    a = output_text

    # Concat demo + main question ...
    input_batch = args.instruction + samples + q
    encoded_input = tokenizer(input_batch, return_tensors='pt', padding=True, truncation=True).to(args.device)
    encoded_input_token_num = encoded_input['input_ids'].shape[-1]
    print(encoded_input_token_num)

    # Model inference ...
    with torch.no_grad():
        #print(encoded_input)
        output = model.generate(**encoded_input, max_new_tokens=args.max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id,
                                return_dict_in_generate=True,
                                output_scores=True,
                                do_sample=False, top_p=1, temperature=0
                              )
    pred_list = []
    decoded_output = tokenizer.decode(output['sequences'][0][encoded_input_token_num:], skip_special_tokens=True)
    pred = decoded_output.split("\n")[0].strip("\n").strip(" ")

    scores = scorer.score(a, pred)
    
    results1.append(scores['rouge1'][-1])
    results2.append(scores['rouge2'][-1])
    results3.append(scores['rouge3'][-1])
    results4.append(scores['rougeL'][-1])
    
    if total % 10 == 0:
        print("check:")
        print(input_batch)
        print("###")
        print(decoded_output)
        print("###")
        print(pred)
        print(a)
        print(results1[total])
        print(results2[total])
        print(results3[total])
        print(results4[total])
    
    total += 1
    
    if total >= 200:
      print("evaluation break...")
      break

  rouge1 = 100 * sum(results1) / len(results1)
  rouge2 = 100 * sum(results2) / len(results2)
  rouge3 = 100 * sum(results3) / len(results3)
  rougeL = 100 * sum(results4) / len(results4)    
  print(f"rouge1: {rouge1}")
  print(f"rouge2: {rouge2}")
  print(f"rouge3: {rouge3}")
  print(f"rougeL: {rougeL}")
  result = f"{args.eval_noise_dataset}; {args.noise_type}; {rouge1}; {rouge2}; {rouge3}; {rougeL}"
  return result

# Set In-Context examples from Training dataset...
def set_incontext_examples(args, dataiter, tokenizer):
    sample_set = []
    for j in range(10000):
      data = next(dataiter)

      output_text = data[args.output_lable][0]
      output_text = tokenizer.tokenize(output_text)[:args.max_new_tokens]
      output_text = tokenizer.convert_tokens_to_ids(output_text)
      output_text = tokenizer.decode(output_text)
        
      # Limit maximum word length in a text to fixed size ...
      input_text = data[args.input_lable][0] #.split(" ")[:args.num_text_limit])
      input_text = make_text_noise(args, input_text)
      input_text = tokenizer.tokenize(input_text)[:args.num_text_limit]
      input_text = tokenizer.convert_tokens_to_ids(input_text)
      input_text = tokenizer.decode(input_text)
      
      sample = args.prefix1 + " " + input_text + "\n" + args.prefix2 + " " + output_text + "\n\n"

      sample_set.append(sample)
        
      if args.incontex_sample_size == 0:
          sample_set = []
        
      if len(sample_set) >= args.incontex_sample_size:
          break
    
    random.shuffle(sample_set)
    return sample_set

def eval_noise_summary(args):

    fix_seed(args.random_seed)

    model, tokenizer, config = load_model(args)

    data_dir = None
    print(args.dataset_cache_dir)
    print(data_dir)
    
    ds = load_dataset(args.eval_noise_dataset, data_dir=data_dir, cache_dir=args.dataset_cache_dir)

    train_loader = prepare_data_loader(args, ds[args.ds_train_lable], batch_size=1, shuffle=True)
    
    # Evaluation
    test_loader = prepare_data_loader(args, ds[args.ds_test_lable], batch_size=1, shuffle=True)
    result = evaluation(args, test_loader, model, tokenizer, train_loader)
    return result