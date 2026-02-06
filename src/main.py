#%% import libraries
import gc
import os
import json
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments,TrainerCallback, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
#%%
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "stanfordnlp/imdb"
np.random.seed(SEED)
torch.manual_seed(SEED) # if using CPU
torch.cuda.manual_seed(SEED) # if using single-GPU
torch.cuda.manual_seed_all(SEED) # if using multi-GPU
torch.backends.cudnn.deterministic = True # deterministic mode
torch.backends.cudnn.benchmark = False # disable auto-tuner to find the best algorithm to use for your hardware
torch.backends.cuda.matmul.allow_tf32 = True # allow TensorFloat-32 on matmul operations
torch.backends.cudnn.allow_tf32  = True # allow TensorFloat-32 on convolution operations
# torch.autograd.set_detect_anomaly(True) # keep this commented out for speed unless debugging NaN
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
print(f"Train size: {len(small_dataset['train'])}")
print(f"Val size: {len(small_dataset['val'])}")
#%% Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(tokenizer)
#%% Dataset preprocessing
small_tokenized_dataset = small_dataset.map(
    lambda example: tokenizer(example['text'], truncation=True),
    batched=True,
    batch_size=16
)
small_tokenized_dataset = small_tokenized_dataset.remove_columns(["text"])
small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
small_tokenized_dataset.set_format("torch")
print(small_tokenized_dataset['train'][0:2])
#%% Model definition
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def compute_metrics(pred):
    """Called at the end of validation. Gives accuracy"""
    logits = pred.predictions
    labels = pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": np.mean(predictions == labels),
        "f1": f1_score(labels, predictions, average='weighted')
    }

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 5e-5, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.1]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 3),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
    }

arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    eval_strategy="epoch", # run validation at the end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    seed=SEED
)
trainer = Trainer(
    model_init=model_init,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['val'], # change to test when you do your final evaluation!
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
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
trainer.add_callback(LoggingCallback("./results/log_classic.jsonl"))
#%%
best_run = trainer.hyperparameter_search(
    direction="maximize", 
    backend="optuna", 
    hp_space=hp_space, 
    n_trials=5,
    compute_objective=lambda metrics: metrics['eval_accuracy']
)
# Update trainer with best run hyperparameters and train final model
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)
print(best_run)
#%%
trainer.train()
#%%
# just gets evaluation metrics
# results = trainer.evaluate()
# also gives you predictions
results = trainer.predict(small_tokenized_dataset['val'])
# Report metrics and inference time
print("--- Evaluation Results ---")
print(f"Accuracy: {results.metrics['test_accuracy']:.4f}")
print(f"F1 Score: {results.metrics['test_f1']:.4f}")
print(f"Inference Time: {results.metrics['test_runtime']:.4f} seconds")
print(f"Inference Speed: {results.metrics['test_samples_per_second']:.2f} samples/sec")
#%%
#TODO: missing the saving of the checkpoint

# To load our saved model, we can pass the path to the checkpoint into the `from_pretrained` method:
test_str = "I enjoyed the movie!"
finetuned_model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-???")
model_inputs = tokenizer(test_str, return_tensors="pt")
prediction = torch.argmax(finetuned_model(**model_inputs).logits)
print(["NEGATIVE", "POSITIVE"][prediction])