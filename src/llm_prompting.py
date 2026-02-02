#%% import libraries
import gc
import os
import json
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments,TrainerCallback, EarlyStoppingCallback
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
torch.autograd.set_detect_anomaly(True)
print("Using device: ", DEVICE)
#%% Dataset loading
dataset = load_dataset(DATASET_NAME)
print(dataset)
#%% Prompt Strategies
def truncate_few_shot(example):
    # Few-shot prompt design: providing examples to the LLM
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

def truncate_zero_shot(example):
    # Zero-shot prompt design: No examples, just instruction
    review_segment = " ".join(example['text'].split()[:50])
    prompt = (
        "You are a sentiment classifier. Determine if the following movie reviews are POSITIVE or NEGATIVE.\n\n"
        f"Review: {review_segment}\n"
        "Sentiment:"
    )
    return {'text': prompt, 'label': example['label']}
#%% Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Ensure pad token is set for Phi-3 (often missing in LLMs)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
print(tokenizer)
#%% Model Definition
def model_init():
    # Load Phi-3 for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    # Freeze the backbone to perform Linear Probing
    # This keeps the LLM knowledge intact and only trains the classification head
    # This is crucial to fit training on standard GPUs without LoRA
    for param in model.base_model.parameters():
        param.requires_grad = False
    return model

model = model_init() # Create one instance for summary
try:
    # Adjusted input size for summary to match prompt length roughly
    summary(model, input_size=(1, 128), col_names=('input_size', 'output_size', 'num_params', 'trainable'), dtypes=[torch.IntTensor])
except Exception as e:
    print(f"Summary skipped: {e}")
#%%
def compute_metrics(pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": np.mean(predictions == labels),
        "f1": f1_score(labels, predictions, average='weighted')
    }

def hp_space(trial):
    return {
        # Higher LR for head tuning
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.01]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 3),
        # Smaller batch sizes for LLM memory constraints
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [1, 2, 4]),
    }

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
#%% Experiment Runner
def run_experiment(name, map_fn):
    print(f"\n\n{'='*20} Running Experiment: {name} {'='*20}")
    # Prepare Data
    current_dataset = DatasetDict(
        train=dataset['train'].shuffle(seed=SEED).select(range(128)).map(map_fn),
        val=dataset['train'].shuffle(seed=SEED).select(range(128, 160)).map(map_fn),
    )
    tokenized_dataset = current_dataset.map(
        lambda example: tokenizer(example['text'], padding=True, truncation=True),
        batched=True,
        batch_size=16
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    # Setup Trainer
    arguments = TrainingArguments(
        output_dir=f"./results/{name}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        load_best_model_at_end=True,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        logging_dir=f"./results/{name}/logs"
    )
    trainer = Trainer(
        model_init=model_init,
        args=arguments,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['val'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
    trainer.add_callback(LoggingCallback(f"./results/{name}/log.jsonl"))
    # Hyperparameter Search
    print(f"--- Tuning Hyperparameters for {name} ---")
    best_run = trainer.hyperparameter_search(
        direction="maximize", 
        backend="optuna", 
        hp_space=hp_space, 
        n_trials=5,
        compute_objective=lambda metrics: metrics['eval_accuracy']
    )
    # Train Best Model
    print(f"--- Training Best Model for {name} ---")
    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)
    trainer.train()
    # Evaluate
    results = trainer.predict(tokenized_dataset['val'])
    # Cleanup to free VRAM for next run
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return results.metrics

#%% Execution
# Run Few-Shot Training
fs_metrics = run_experiment("Few_Shot", truncate_few_shot)
# Run Zero-Shot Training
zs_metrics = run_experiment("Zero_Shot", truncate_zero_shot)
#%% Comparison
print("\n\n=== FINAL COMPARISON ===")
print(f"{'Metric':<20} | {'Few-Shot':<15} | {'Zero-Shot':<15}")
print("-" * 56)
print(f"{'Accuracy':<20} | {fs_metrics['test_accuracy']:.4f}          | {zs_metrics['test_accuracy']:.4f}")
print(f"{'F1 Score':<20} | {fs_metrics['test_f1']:.4f}          | {zs_metrics['test_f1']:.4f}")
print(f"{'Inference Time (s)':<20} | {fs_metrics['test_runtime']:.4f}          | {zs_metrics['test_runtime']:.4f}")