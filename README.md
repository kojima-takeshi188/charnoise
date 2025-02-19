# Continual Pre-training on Character-Level Noisy Texts Makes Decoder-based Language Models Robust Few-shot Learners

This is the official implementation of `Continual Pre-training on Character-Level Noisy Texts Makes Decoder-based Language Models Robust Few-shot Learners`.

## 1. Installation

#### 1-1. Create a virtual environment
```
cd <path_to_this_project>
conda create -n charnoise python=3.10.14
conda activate charnoise
```

#### 1-2. Install pytorch
```
# https://pytorch.org/get-started/locally/
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

#### 1-3. Install python libraries
```
pip3 install -r requirements.txt
```

#### 1-4. Install lm-evaluation harness
```
git clone https://github.com/EleutherAI/lm-evaluation-harness -b 0.4.7
cd lm-evaluation-harness
pip install -e .
cd ..
```

## 2. Training

#### 2-1. Set environment variables
```
# For Huggingface
export HF_TOKEN="XXXX"
# For Wandb
export WANDB_API_KEY="XXXX"
```

#### 2-2. Run training program
```
python main.py --model_name=gpt2-xl --scenario=train --checkpointing --trained_param_file_suffix=main --total_global_step=1000
```

## 3. Evaluation

#### 3-1. Download evaluation data

```
# bea60k
mkdir bea60k
cd bea60k
download test.bea60k and test.bea60k.noise from https://drive.google.com/drive/folders/1ejKSkiHNOlupxXVDMg67rPdqwowsTq1i

# github-typo-corpus
mkdir github-typo-corpus
cd github-typo-corpus
wget https://github-typo-corpus.s3.amazonaws.com/data/github-typo-corpus.v1.0.0.jsonl.gz

# flores200_dataset
mkdir flores200_dataset
cd flores200_dataset
wget https://tinyurl.com/flores200dataset

# spell-errors
mkdir datasets/custom_typo_injected
cd datasets/custom_typo_injected
wget https://norvig.com/ngrams/spell-errors.txt
```

#### 3-2. Evaluation (Artificial character-level noise - text classification)
```
# For base model
python main.py --scenario=eval_noise --noise_type=swap --model_name=gpt2-xl --eval_noise_dataset=sst2
# For contually pre-trained model (Ours)
python main.py --scenario=eval_noise --noise_type=swap --model_name=gpt2-xl --eval_noise_dataset=sst2 --trained_param_file_suffix=main --restore_steps=1000
```

#### 3-3. Evaluation (Artificial character-level noise - text summarization)
```
python main.py --scenario=eval_noise_summary --noise_type=nothing --model_name=gpt2-xl --eval_noise_dataset=EdinburghNLP/xsum --trained_param_file_suffix=main --restore_steps=1000
```

#### 3-4. Evaluation (Artificial character-level noise - text reconstruction from typoglycemia)
```
python main.py --scenario=eval_typoglycemia --eval_noise_dataset=typoglycemia --model_name=gpt2-xl --trained_param_file_suffix=main --restore_steps=1000
```

#### 3-5. Evaluation (Real World Typo - Typo Correction)
```
python main.py --scenario=eval_typo --model_name=gpt2-xl --eval_noise_dataset=github-typo-corpus --trained_param_file_suffix=main --restore_steps=1000
```

#### 3-6. Evaluation (Real World Typo - Classification)
```
python main.py --scenario=eval_noise --model_name=gpt2-xl --eval_noise_dataset=sst2 --injection_typo_flag --trained_param_file_suffix=main --restore_steps=1000
```

#### 3-7. Evaluation (English downstream task)
```
python main.py --scenario=eval_lmevalharness --model_name=gpt2-xl --lmeval_task=downstream --trained_param_file_suffix=main --restore_steps=1000
```

#### 3-8. Evaluation (Multilingual downstream task)
```
python main.py --scenario=eval_lmevalharness --model_name=gpt2-xl --lmeval_task=multilingual --trained_param_file_suffix=main --restore_steps=1000
```

#### 3-9. Evaluation (Perplexity)
```
python main.py --scenario=eval_ppl --model_name=gpt2-xl --eval_ppl_dataset=fineweb --trained_param_file_suffix=main --restore_steps=1000
```