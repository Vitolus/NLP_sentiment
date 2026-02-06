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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments,TrainerCallback, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
#%%
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
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
#%% Prompt Strategies
def truncate_few_shot(example):
    # Few-shot prompt design: providing examples to the LLM
    # Take the first 50 words of the review to keep the prompt short
    review_segment = " ".join(example['text'].split()[:50])
    prompt = (
        "You are a sentiment classifier. Determine if the following movie reviews are POSITIVE or NEGATIVE.\n\n"
        "Review: The movie was terrible, boring and too long.\n"
        "Sentiment: NEGATIVE\n\n"
        "Review: Absolutely fantastic! I loved every minute of it.\n"
        "Sentiment: POSITIVE\n\n"
        f"Review: {review_segment}\n"
        "Sentiment:"
    )
    return {'text': prompt, 'label': example['label']}

small_few_shot = DatasetDict(
    train=dataset['train'].shuffle(seed=SEED).select(range(128)).map(truncate_few_shot),
    val=dataset['train'].shuffle(seed=SEED).select(range(128, 160)).map(truncate_few_shot),
)
print(small_few_shot)
print(small_few_shot['train'][:10])
print(f"Train size: {len(small_few_shot['train'])}")
print(f"Val size: {len(small_few_shot['val'])}")
#%%
def truncate_zero_shot(example):
    # Zero-shot prompt design: No examples, just instruction
    review_segment = " ".join(example['text'].split()[:50])
    prompt = (
        "You are a sentiment classifier. Determine if the following movie reviews are POSITIVE or NEGATIVE.\n\n"
        f"Review: {review_segment}\n"
        "Sentiment:"
    )
    return {'text': prompt, 'label': example['label']}

small_zero_shot = DatasetDict(
    train=dataset['train'].shuffle(seed=SEED).select(range(128)).map(truncate_zero_shot),
    val=dataset['train'].shuffle(seed=SEED).select(range(128, 160)).map(truncate_zero_shot),
)
print(small_zero_shot)
print(small_zero_shot['train'][:10])
print(f"Train size: {len(small_zero_shot['train'])}")
print(f"Val size: {len(small_zero_shot['val'])}")
#%% Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
print(tokenizer)
#%% Dataset preprocessing
small_tokenized_few_shot = small_few_shot.map(
    lambda example: tokenizer(example['text'], truncation=True),
    batched=True,
    batch_size=16
)
small_tokenized_few_shot = small_tokenized_few_shot.remove_columns(["text"])
small_tokenized_few_shot = small_tokenized_few_shot.rename_column("label", "labels")
small_tokenized_few_shot.set_format("torch")
print(small_tokenized_few_shot['train'][0:2])
#%% Dataset preprocessing
small_tokenized_zero_shot = small_zero_shot.map(
    lambda example: tokenizer(example['text'], truncation=True),
    batched=True,
    batch_size=16
)
small_tokenized_zero_shot = small_tokenized_zero_shot.remove_columns(["text"])
small_tokenized_zero_shot = small_tokenized_zero_shot.rename_column("label", "labels")
small_tokenized_zero_shot.set_format("torch")
print(small_tokenized_zero_shot['train'][0:2])
#%% Model Definition
def model_init():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=False)
    # IMPORTANT: Do not set rope_scaling to None on Phi-3 configs with transformers>=5.
    # Doing so will set rope_parameters=None internally and crash the native Phi3 implementation.
    # Leave the default as-is; if needed, adjust rope_type explicitly (e.g., to 'linear').
    # Example of safe adjustment (commented out):
    # if isinstance(config.rope_scaling, dict) and config.rope_scaling.get("rope_type") == "default":
    #     config.rope_scaling["rope_type"] = "linear"
    
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config, num_labels=2, trust_remote_code=False)

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
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),
    }

arguments_few_shot = TrainingArguments(
    output_dir="./results/few_shot",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    eval_strategy="epoch", # run validation at the end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=torch.cuda.is_available()
)
arguments_zero_shot = TrainingArguments(
    output_dir="./results/zero_shot",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    eval_strategy="epoch", # run validation at the end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=torch.cuda.is_available()
)

trainer_few_shot = Trainer(
    model_init=model_init,
    args=arguments_few_shot,
    train_dataset=small_tokenized_few_shot['train'],
    eval_dataset=small_tokenized_few_shot['val'], # change to test when you do your final evaluation!
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)
trainer_zero_shot = Trainer(
    model_init=model_init,
    args=arguments_zero_shot,
    train_dataset=small_tokenized_zero_shot['train'],
    eval_dataset=small_tokenized_zero_shot['val'], # change to test when you do your final evaluation!
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
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")

class FreeMemoryCallback(TrainerCallback):
    """
    Frees Python + CUDA memory at the end of *each* training run.
    This matters a lot during hyperparameter_search, which runs multiple train() calls.
    """
    def on_train_end(self, args, state, control, **kwargs):
        # Try to release as much as possible between Optuna trials
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

trainer_few_shot.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
trainer_few_shot.add_callback(LoggingCallback("./results/log_few_shot.jsonl"))
trainer_few_shot.add_callback(FreeMemoryCallback())
trainer_zero_shot.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
trainer_zero_shot.add_callback(LoggingCallback("./results/log_zero_shot.jsonl"))
trainer_zero_shot.add_callback(FreeMemoryCallback())
#%% Few Shot Experiment
best_run_few_shot = trainer_few_shot.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=5,
    compute_objective=lambda metrics: metrics['eval_accuracy'],
    gc_after_trial=True
)
# Update trainer with best run hyperparameters and train final model
for n, v in best_run_few_shot.hyperparameters.items():
    setattr(trainer_few_shot.args, n, v)
print(best_run_few_shot)
#%% Zero Shot Experiment
best_run_zero_shot = trainer_zero_shot.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=5,
    compute_objective=lambda metrics: metrics['eval_accuracy'],
    gc_after_trial=True
)
# Update trainer with best run hyperparameters and train final model
for n, v in best_run_zero_shot.hyperparameters.items():
    setattr(trainer_zero_shot.args, n, v)
print(best_run_zero_shot)
#%% Execution
trainer_few_shot.train()
#%%
results_few_shot = trainer_few_shot.predict(small_tokenized_few_shot['val'])
#%%
trainer_zero_shot.train()
#%%
results_zero_shot = trainer_zero_shot.predict(small_tokenized_zero_shot['val'])
#%% Comparison
print("--- FINAL COMPARISON ---")
print(f"{'Metric':<20} | {'Few-Shot':<15} | {'Zero-Shot':<15}")
print("-" * 56)
print(f"{'Accuracy':<20} | {results_few_shot.metrics['test_accuracy']:.4f}  | {results_zero_shot.metrics['test_accuracy']:.4f}")
print(f"{'F1 Score':<20} | {results_few_shot.metrics['test_f1']:.4f}  | {results_zero_shot.metrics['test_f1']:.4f}")
print(f"{'Inference Time (s)':<20} | {results_few_shot.metrics['test_runtime']:.4f}  | {results_zero_shot.metrics['test_runtime']:.4f}")
print(f"{'Inference Speed (samples/s)':<20} | {results_few_shot.metrics['test_samples_per_second']:.4f}  | {results_zero_shot.metrics['test_samples_per_second']:.4f}")
