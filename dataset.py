import json
import os
import re
import torch as th


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] +"\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct2(model_completion, gt_answer):
    gt_answer = extract_answer(gt_answer)
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class GSMDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

        self.op2tid = {"+": 10, "-":12, "/": 15, "x": 87, "*":9}
        self.tid2op = dict((v,k) for k,v in self.op2tid.items())
        self.label2op = {1:"+", 2:"-", 3:"/", 4: "x", 5:"*", 0: "other"}
        self.op2label = dict((v,k) for k,v in self.label2op.items())

        self.tid2label = dict((k,self.op2label[v]) for k,v in self.tid2op.items())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [50256] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)

        mask2 = (
            ([int(False)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )

        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        action_tokens = [self.tid2label[tid] if tid in self.tid2label else 0 for tid in ans_tokens]
        # action_tokens = [1 if tid in self.tid2label else 2 for tid in ans_tokens]
        actions = [0]*len(qn_tokens) + action_tokens + pad_tokens
        query_pad_tokens = [0] * (self.max_len - len(qn_tokens) )
        return dict(input_ids=tokens, attention_mask=mask, loss_mask=th.tensor(mask2), query=th.tensor(qn_tokens),actions=th.tensor(actions), data=self.examples[idx])
