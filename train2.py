import torch as th
from torch import nn
import torch.nn.functional as F
from dataset import get_examples, GSMDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = th.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    # logpy = th.gather(logits, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

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
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True)
    optim = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 20
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k=="input_ids" or k=="attention_mask"}

            outputs = model(**inputs)

            labels=batch["input_ids"].to(device)
            #loss_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            if labels is not None:

                lm_logits = outputs['logits']
                logprobs = logprobs_from_logits(lm_logits[:, :-1, :], labels[:, 1:])
                # move labels to correct device to enable model parallelism
                mask_probs = loss_mask[:, 1:]*logprobs
                '''
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                '''
                loss = -mask_probs.mean()
                # loss = -logprobs.mean() # -th.mean(th.sum(mask_probs, -1)/th.sum(loss_mask[:, 1:], -1))
            #loss = outputs[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")

    model.save_pretrained("model_ckpts2/")


if __name__ == "__main__":
    main()
