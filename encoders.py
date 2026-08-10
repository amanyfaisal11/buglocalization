
from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

UNIXCODER_MODEL = "microsoft/unixcoder-base"
GRAPHCODEBERT_MODEL = "microsoft/graphcodebert-base"


class CodeEncoder(nn.Module):

    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 512):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name)

        self.device = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
        self.model.to(self.device)

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.train()

    def encode(self, texts: List[str], max_length: int = None) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.model.config.hidden_size, device=self.device)
        max_length = max_length or self.max_length
        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def encode_one(self, text: str, max_length: int = None) -> torch.Tensor:
        return self.encode([text], max_length=max_length)[0]

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, map_location=None):
        state = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(state)


def load_unixcoder(device: str = "cuda", max_length: int = 512) -> CodeEncoder:
    return CodeEncoder(UNIXCODER_MODEL, device=device, max_length=max_length)


def load_graphcodebert(device: str = "cuda", max_length: int = 512) -> CodeEncoder:
    return CodeEncoder(GRAPHCODEBERT_MODEL, device=device, max_length=max_length)
