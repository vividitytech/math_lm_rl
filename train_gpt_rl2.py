import torch as th
from torch import nn
import torch.nn.functional as F
from dataset import get_examples, GSMDataset, is_correct
from transformers import GPT2Tokenizer, GPT2LMHeadModel, HfArgumentParser
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from ppo_config import PPOConfig
from reinforce import PPOTrainer
from core import LengthSampler
from dataclasses import dataclass, field

from typing import Optional, TypeVar

from models import AutoModelForCausalLMWithValueHead

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def get_reward(texts, batch):
    reward = []
    for text, ans in zip(texts, batch):
        reward.append(True == is_correct(text,  ans))
    return reward

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = th.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    # logpy = th.gather(logits, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

def loss_fun(
        model,
        hidden_states,
        logits,
        rewards,
    ):
    """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`, `response_length`)
            logits (`torch.FloatTensor`):
                Logits of the model, shape (`batch_size`, `response_length`, `vocab_size`)
            v_pred (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
    """
    loss_base = 0.0
    loss_rl = 0.0
    advantages_reversed = []
    batch_size = len(hidden_states)
    for i in range(batch_size):
        seq_hidden = hidden_states[i]
        values = model.infer_value(seq_hidden)
        gen_len = seq_hidden.shape[0]
        returns = th.zeros_like(values)
        returns[-1] = rewards[i]
        for t in reversed(range(gen_len-1)):
            returns[t] = 0.99*returns[t+1]
        advantages = returns-values

        #advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()
        logp=F.log_softmax(logits[i], dim=-1)
        
        idx = th.argmax(logits[i], -1)
        logprobs = th.gather(logp, 1, idx.unsqueeze(1)).squeeze(-1)
        pg_losses = -advantages * logprobs
        pg_losses=th.where(pg_losses > 0, pg_losses, 0.)
        loss_rl +=pg_losses.mean()

        loss_base += ((returns-values) ** 2).mean()

    return loss_rl, loss_base


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="gpt2", #
                                       metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})

generation_kwargs = {
    "return_dict_in_generate": True,
    "output_scores": True,
    "output_hidden_states": True,
    "do_sample": True,
    'use_cache': True,
}
# 85 in totoal 1319
#
def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    o3 = tokenizer("x") 
    o4 = tokenizer("*") 
    o1 = tokenizer("+")
    o2 = tokenizer("-")
    o5 = tokenizer("/")
    r1 = tokenizer.decode(0)
    train_examples = get_examples("train")
    train_dset = GSMDataset(tokenizer, train_examples, loss_on_prefix=True)

    device = th.device("cuda")
    model_path ="/home/gangchen/Downloads/project/machinelearning/trl/examples/gsm8k2/"
    # config = GPT2Config.from_pretrained("gpt2")
    model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")#("model_ckpts2")#, config=config)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    pconfig = PPOConfig(
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        mini_batch_size=script_args.mini_batch_size,
        batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
    )

    model.to(device)
    model.train()
    ref_model.to(device)
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(pconfig, model, ref_model, tokenizer, dataset=train_dset)


    train_loader = DataLoader(train_dset, batch_size=script_args.batch_size, shuffle=True, collate_fn=collator)
    optim = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 20
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    tokenizer.pad_token = tokenizer.eos_token
    output_min_length = 300
    output_max_length = 350
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    use_mask = True # use mask, without question 
    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            #inputs = {k: v.to(device) for k, v in batch.items() if k=="input_ids" or k=="attention_mask"}
            
            query_tensors = batch["query"]#.to(device)
            query_tensors = [tensor.to(device) for tensor in query_tensors]
            response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False,gene_type="mixture", pad_token_id=tokenizer.eos_token_id, length_sampler=output_length_sampler, **generation_kwargs)
            batch["response"] = tokenizer.batch_decode(response_tensors["sequences"])
            scores = response_tensors["scores"]
            hidden_states = response_tensors['hidden_states']
            # Compute sentiment score
            texts = [q["question"] + r for q, r in zip(batch['data'], batch["response"])]
            pipe_outputs = get_reward(batch["response"], batch['data'])
            rewards = [th.tensor(output==True).to(th.float32) for output in pipe_outputs]

            loss_rl, loss_baseline = loss_fun(model, hidden_states, scores, rewards)
            # model_inputs = ppo_trainer.prepare_model_inputs(query_tensors, response_tensors)
            # input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            
            model_inputs = {k: th.stack(v).to(device) for k, v in batch.items() if k=="input_ids" or k=="attention_mask"}
            outputs = model(**model_inputs)
            lm_logits = outputs[0]
            labels=[ tensor[:lm_logits.shape[1]].to(device) for tensor in batch["input_ids"]]
            
            lm_action_logits = outputs[3]
            action_labels = [ tensor[:lm_logits.shape[1]].to(device) for tensor in batch["actions"]]
            ##loss_mask = batch["attention_mask"].to(device)
            loss_mask =[ tensor.to(device) for tensor in batch["loss_mask"]]
            if labels is not None:

                logprobs = logprobs_from_logits(lm_logits[:, :-1, :], th.stack(labels)[:, 1:])
                #logprobs = logprobs_from_logits(lm_logits[:, :-1, :], th.stack(labels)[:, 1:])
                # move labels to correct device to enable model parallelism
                '''
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                '''
                if use_mask:
                    mask_probs = th.stack(loss_mask)[:, 1:]*logprobs
                    loss = -mask_probs.mean()
                else:
                    loss = -logprobs.mean() # -th.mean(th.sum(mask_probs, -1)/th.sum(loss_mask[:, 1:], -1))
                action_logprobs = logprobs_from_logits(lm_action_logits[:, :-1, :], th.stack(action_labels)[:, 1:])
                tmp = th.stack(loss_mask)[:, 1:]*action_logprobs
                loss = loss - tmp.mean()
            #loss = outputs[0]

            loss = loss + 0.2*loss_rl
            loss = loss + 0.1*loss_baseline
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}, loss_rl: {loss_rl}, loss_baseline: {loss_baseline}")

        model.save_pretrained("model_rl_02/")


if __name__ == "__main__":
    main()
