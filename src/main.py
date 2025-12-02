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
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, 
Trainer, TrainingArguments,TrainerCallback, EarlyStoppingCallback
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
    train=dataset['train'].shuffle(seed=SEED).select(range(128)).map(truncate),
    val=dataset['train'].shuffle(seed=SEED).select(range(128, 160)).map(truncate),
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
trainloader = DataLoader(small_tokenized_dataset['train'], batch_size=16, shuffle=True)
valloader = DataLoader(small_tokenized_dataset['val'], batch_size=16, shuffle=False)
#%% Model definition
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
summary(model, input_size=(16, 50), col_names=('input_size', 'output_size', 'num_params', 'trainable'))
#%%
def compute_metrics(pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    return {"accuracy": np.mean(predictions == labels)}

arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    evaluation_strategy="epoch", # run validation at the end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    seed=SEED
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['val'], # change to test when you do your final evaluation!
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
    # will call on_log on each logging step, specified by TrainerArguement. (i.e TrainerArguement.logginng_step)
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
trainer.add_callback(LoggingCallback("./results/log.jsonl"))
#%%
trainer.train()
#%%
# just gets evaluation metrics
# results = trainer.evaluate()
# also gives you predictions
results = trainer.predict(small_tokenized_dataset['val'])
#%%
# To load our saved model, we can pass the path to the checkpoint into the `from_pretrained` method:
test_str = "I enjoyed the movie!"
finetuned_model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-???")
model_inputs = tokenizer(test_str, return_tensors="pt")
prediction = torch.argmax(finetuned_model(**model_inputs).logits)
print(["NEGATIVE", "POSITIVE"][prediction])