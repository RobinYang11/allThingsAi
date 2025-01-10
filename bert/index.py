# Use a pipeline as a high-level helper
from transformers import pipeline


from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
import os
import pandas as pd

pipe = pipeline("fill-mask", model="bert-base-chinese")

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')

# Load the OpenOrca-Chinese dataset from Hugging Face
dataset = load_dataset("yys/OpenOrca-Chinese")


