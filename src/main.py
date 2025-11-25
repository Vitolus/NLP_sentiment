#%% import libraries
import gc
import os
import re
from collections import deque
import cv2
import lmdb
import pickle
import shutil
from tqdm.notebook import tqdm
import optuna
from optuna.trial import TrialState
import torch
from torch import nn, optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.models as models
from torchinfo import summary
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, DatasetDict
#%%
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_NAME = "stanfordnlp/imdb"
np.random.seed(SEED)
torch.manual_seed(SEED) # if using CPU
torch.cuda.manual_seed(SEED) # if using single-GPU
torch.cuda.manual_seed_all(SEED) # if using multi-GPU
torch.backends.cudnn.deterministic = True # deterministic mode
torch.backends.cudnn.benchmark = False # disable auto-tuner to find the best algorithm to use for your hardware
torch.backends.cuda.matmul.allow_tf32 = True # allow TensorFloat-32 on matmul operations
torch.backends.cudnn.allow_tf32  = True # allow TensorFloat-32 on convolution operations
torch.autograd.set_detect_anomaly(True)
print("Using device: ", DEVICE)
#%% Dataset loading
dataset = load_dataset(DATASET_NAME)
print(dataset)
#%%
# Just take the first 50 tokens for speed/running on cpu
def truncate(example):
    return {
        'text': " ".join(example['text'].split()[:50]),
        'label': example['label']
    }

small_dataset = DatasetDict(
    train=dataset['train'].shuffle(seed=1111).select(range(128)).map(truncate),
    val=dataset['train'].shuffle(seed=1111).select(range(128, 160)).map(truncate),
)
print(small_dataset)
print(small_dataset['train'][:10])
#%%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(tokenizer)
#%% Dataset preprocessing
small_tokenized_dataset = small_dataset.map(
    lambda example: tokenizer(example['text'], padding=True, truncation=True), # https://huggingface.co/docs/transformers/pad_truncation
    batched=True,
    batch_size=16
)
small_tokenized_dataset = small_tokenized_dataset.remove_columns(["text"])
small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
small_tokenized_dataset.set_format("torch")
print(small_tokenized_dataset['train'][0:2])
#%%
