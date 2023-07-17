# math_lm_rl
language models to solve math with reinforcement learning


### Download the GSM8K dataset from url:

(1). https://github.com/openai/grade-school-math

(2). And create data folder under the current repo

## train 
### run train.py to test CE method
python train.py

### run train_gpt_reinforce.py to test CE + RL method
python train_gpt_reinforce.y

### run train_gpt_rl2 to test CE+RL+operators 
python train_gpt_rl2

## evaluation
python evaluate.py
