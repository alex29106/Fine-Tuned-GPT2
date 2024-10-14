import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

"""Function to use the model"""
def run(inputs):
    """Turning input text into model readable form"""
    inputs = "<startofstring> " + inputs + " <bot>: "
    inputs = tokenizer(inputs, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    """Generate output using the loaded model"""
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=200)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    return output


if __name__ == "__main__":
    """Set the device (GPU/CPU)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """Load the tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<pad>",
                                  "bos_token": "<startofstring>",
                                  "eos_token": "<endofstring>"})
    tokenizer.add_tokens(["<bot>:"])

    """Load gpt2"""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    """Load fine tuned model weights"""
    model.load_state_dict(torch.load("models/model_state1.pt", map_location=device))

    model = model.to(device)

    """Set the model to evaluation mode"""
    model.eval()

    """Run the model"""
    print("Running the loaded model...")
    while True:
        text = input()
        print(run(text))
