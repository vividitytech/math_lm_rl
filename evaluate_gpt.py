import torch as th
from dataset import get_examples, GSMDataset, is_correct
from calculator import sample2
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from models import AutoModelForCausalLMWithValueHead

def main():
    device = th.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLMWithValueHead.from_pretrained("model_ckpts2")#("model_rl_08")#
    #("/home/gangchen/Downloads/project/machinelearning/trl/examples/gsm8k/")#
    model.to(device)
    print("Model Loaded")

    test_examples = get_examples("test")
    tsize = len(test_examples)
    count = 0
    sample_len = 100

    fd= open('original.txt', 'a')

    for i in range(len(test_examples)):
        qn = test_examples[i]["question"]
    
        #print(qn.strip())
        qnans = sample2(model, qn, tokenizer, device, sample_len)
        print(qnans)

        res = is_correct(qnans,  test_examples[i])
        count = count + (res==True)

        if(res==True):
            fd.write(f'{qnans}\n')
    print(f"{count} in totoal {tsize}")
    fd.close()
    # 83 in totoal 1319

if __name__ == "__main__":
    main()
