import subprocess
import torch

from utils import *

# https://github.com/EleutherAI/lm-evaluation-harness
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/glue/README.md
# !lm-eval --tasks list
def eval_lmevalharness(args):
    
    if args.trained_param_file_suffix is None:
        pretrained = args.model_name
        peft = ""
    else:
        pretrained = args.model_name
        peft = get_trained_param_dir(args, args.restore_steps)
    
    print(pretrained)
    print(peft)

    dtype = str(args.model_dtype).split(".")[-1]
    print(dtype)

    ################################    
    # lm-evaluation-harness/lm_eval/models/huggingface.py
    # To avoid pythia model loading error, comment-out the following line.
    # L646: self._model.resize_token_embeddings(len(self.tokenizer))
    ################################
    
    cmd = f"""
    lm_eval --model hf \
    --model_args pretrained={pretrained},peft={peft},tokenizer={peft},dtype={dtype},trust_remote_code=True \
    --tasks {args.lmeval_task} \
    --num_fewshot 4
    --batch_size auto
    --limit {args.eval_dataset_size}
    """
    
    subprocess.call(cmd.split())



