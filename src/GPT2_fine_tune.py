from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ChatData import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch


def train(chatData, model, optim):
    epochs = 3
    for i in tqdm.tqdm(range(epochs)):
        for X, a in chatData:
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
        torch.save(model.state_dict(), "../models/model_state1.pt")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """Load tokenizer and Gpt2 Model"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<pad>",
                                  "bos_token": "<startofstring>",
                                  "eos_token": "<endofstring>"})
    tokenizer.add_tokens(["<bot>:"])

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    """Load training data"""
    chatData = ChatData("../data/Dataset1.json", tokenizer)
    chatData = DataLoader(chatData, batch_size=64)

    model.train()

    optim = Adam(model.parameters(), lr=2e-4)

    print("training...")
    train(chatData, model, optim)
