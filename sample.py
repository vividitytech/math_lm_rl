import torch as th
from dataset import get_examples, GSMDataset, is_correct
from calculator import sample, greedy_sample
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def main():
    device = th.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("model_ckpts")
    model.to(device)
    print("Model Loaded")

    test_examples = get_examples("test")
    qn = test_examples[1]["question"]
    sample_len = 100

    output_ids = greedy_sample(model,qn, tokenizer,device, sample_len)
    qnans2 = tokenizer.decode(output_ids[0])
    print(qn.strip())
    qnans = sample(model, qn, tokenizer, device, sample_len)
    print(qnans)

    res = is_correct(qnans,  test_examples[1])
    print(res)


if __name__ == "__main__":
    main()
