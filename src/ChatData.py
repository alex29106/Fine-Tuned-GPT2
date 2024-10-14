from torch.utils.data import Dataset
import json


class ChatData(Dataset):
    def __init__(self, path: str, tokenizer):
        """Load data"""
        self.data = json.load(open(path, "r"))
        self.data = self.data[:]
        self.formated_data = []

        """Convert loaded data into formated data"""
        for i in self.data:
            self.formated_data.append("<startofstring> " + i['text'] + " <bot>: " + i['output'] + " <endofstring>")

        """Encode formated data"""
        self.formated_data_encoded = tokenizer(self.formated_data,
                                               max_length=40,
                                               truncation=True,
                                               padding="max_length",
                                               return_tensors="pt")
        self.input_ids = self.formated_data_encoded['input_ids']
        self.attention_mask = self.formated_data_encoded['attention_mask']

    def __len__(self):
        return len(self.formated_data)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]
