import torch as th
from dataset import get_examples, GSMDataset, is_correct
from calculator import sample
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def main():
    device = th.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = 'left'
    model = GPT2LMHeadModel.from_pretrained("model_ckpts")
    #("/home/gangchen/Downloads/project/machinelearning/trl/examples/gsm8k/")#
    model.to(device)
    print("Model Loaded")

    test_examples = get_examples("test")
    tsize = len(test_examples)
    count = 0
    sample_len = 100
    for i in range(len(test_examples)):
        qn = test_examples[i]["question"]
    
        #print(qn.strip())
        qnans = sample(model, qn, tokenizer, device, sample_len)
        print(qnans)

        res = is_correct(qnans,  test_examples[i])
        count = count + (res==True)
    print(f"{count} in totoal {tsize}")
    # 83 in totoal 1319
    '''
    with open('original.txt', 'a') as fd:
        fd.write(f'{count}\n')
        fd.close()
    '''
if __name__ == "__main__":
    main()
