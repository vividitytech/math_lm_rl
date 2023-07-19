# math_lm_rl
Technical report: A mixed policy to improve performance of language models on math problems
refer to https://arxiv.org/pdf/2307.08767v1.pdf

### Download the GSM8K dataset from url:

(1). https://github.com/openai/grade-school-math

(2). copy the data folder under the current repo

## train 
### run train.py to test CE method
python train.py

### run train_gpt_reinforce.py to test CE + RL method
python train_gpt_reinforce.y

### run train_gpt_rl2 to test CE+RL+operators 
python train_gpt_rl2

## evaluation
python evaluate.py
