#%% import libraries
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

#%% Configuration
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Using Phi-3-mini as a modern, efficient open-weight LLM
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" 
DATASET_NAME = "stanfordnlp/imdb"

np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")

#%% Dataset loading (Mimicking main.py exactly)
dataset = load_dataset(DATASET_NAME)

# Just take the first 50 tokens for speed/consistency with main.py
def truncate(example):
    return {
        'text': " ".join(example['text'].split()[:50]),
        'label': example['label']
    }

# Exact same split logic as main.py
small_dataset = DatasetDict(
    train=dataset['train'].shuffle(seed=SEED).select(range(128)).map(truncate),
    val=dataset['train'].shuffle(seed=SEED).select(range(128, 160)).map(truncate),
)

print("Dataset loaded.")
print(f"Train size: {len(small_dataset['train'])}")
print(f"Val size: {len(small_dataset['val'])}")

#%% Model Loading
print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
    trust_remote_code=True
)

# Create a generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10, # We only need a short answer (POSITIVE/NEGATIVE)
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)

#%% Prompt Engineering
def create_few_shot_prompt(target_text, examples):
    """
    Creates a few-shot prompt using examples from the training set.
    """
    prompt = "Classify the sentiment of the following movie reviews as POSITIVE or NEGATIVE.\n\n"
    
    # Add few-shot examples (3 examples)
    for ex in examples:
        label_str = "POSITIVE" if ex['label'] == 1 else "NEGATIVE"
        prompt += f"Review: {ex['text']}\nSentiment: {label_str}\n\n"
    
    # Add target
    prompt += f"Review: {target_text}\nSentiment:"
    return prompt

# Select 3 fixed examples for few-shot to keep it deterministic
few_shot_examples = [small_dataset['train'][i] for i in [0, 10, 20]]

#%% Inference Loop
print("Starting inference...")
predictions = []
ground_truth = []
latencies = []

# Iterate over validation set
for i, example in tqdm(enumerate(small_dataset['val']), total=len(small_dataset['val'])):
    text = example['text']
    label = example['label']
    ground_truth.append(label)
    
    # Construct prompt
    prompt = create_few_shot_prompt(text, few_shot_examples)
    
    # Measure inference time
    start_time = time.time()
    
    # Generate
    # Note: We use a low temperature for deterministic outputs
    output = pipe(prompt, temperature=0.1, do_sample=False)
    generated_text = output[0]['generated_text'].strip().upper()
    
    end_time = time.time()
    latencies.append(end_time - start_time)
    
    # Parse prediction
    if "POSITIVE" in generated_text:
        pred_label = 1
    elif "NEGATIVE" in generated_text:
        pred_label = 0
    else:
        # Fallback or "Neutral" - mapping to -1 or random could be options. 
        # Here we default to 0 (Negative) but print a warning for qualitative analysis.
        # print(f"Warning: Could not parse output: '{generated_text}'")
        pred_label = 0 
        
    predictions.append(pred_label)

#%% Evaluation
accuracy = accuracy_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions, average='weighted')
avg_latency = np.mean(latencies)
samples_per_sec = 1.0 / avg_latency

print("\n--- Evaluation Results (LLM Few-Shot) ---")
print(f"Model: {MODEL_NAME}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Avg Inference Time: {avg_latency:.4f} seconds/sample")
print(f"Inference Speed: {samples_per_sec:.2f} samples/sec")

#%% Qualitative Analysis (Optional)
print("\n--- Qualitative Errors ---")
count = 0
for i, (true, pred) in enumerate(zip(ground_truth, predictions)):
    if true != pred and count < 3:
        print(f"Example {i}:")
        print# filepath: /home/vito/NLP_sentiment/src/llm_prompting.py
#%% import libraries
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

#%% Configuration
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Using Phi-3-mini as a modern, efficient open-weight LLM
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" 
DATASET_NAME = "stanfordnlp/imdb"

np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")

#%% Dataset loading (Mimicking main.py exactly)
dataset = load_dataset(DATASET_NAME)

# Just take the first 50 tokens for speed/consistency with main.py
def truncate(example):
    return {
        'text': " ".join(example['text'].split()[:50]),
        'label': example['label']
    }

# Exact same split logic as main.py
small_dataset = DatasetDict(
    train=dataset['train'].shuffle(seed=SEED).select(range(128)).map(truncate),
    val=dataset['train'].shuffle(seed=SEED).select(range(128, 160)).map(truncate),
)

print("Dataset loaded.")
print(f"Train size: {len(small_dataset['train'])}")
print(f"Val size: {len(small_dataset['val'])}")

#%% Model Loading
print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
    trust_remote_code=True
)

# Create a generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10, # We only need a short answer (POSITIVE/NEGATIVE)
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)

#%% Prompt Engineering
def create_few_shot_prompt(target_text, examples):
    """
    Creates a few-shot prompt using examples from the training set.
    """
    prompt = "Classify the sentiment of the following movie reviews as POSITIVE or NEGATIVE.\n\n"
    
    # Add few-shot examples (3 examples)
    for ex in examples:
        label_str = "POSITIVE" if ex['label'] == 1 else "NEGATIVE"
        prompt += f"Review: {ex['text']}\nSentiment: {label_str}\n\n"
    
    # Add target
    prompt += f"Review: {target_text}\nSentiment:"
    return prompt

# Select 3 fixed examples for few-shot to keep it deterministic
few_shot_examples = [small_dataset['train'][i] for i in [0, 10, 20]]

#%% Inference Loop
print("Starting inference...")
predictions = []
ground_truth = []
latencies = []

# Iterate over validation set
for i, example in tqdm(enumerate(small_dataset['val']), total=len(small_dataset['val'])):
    text = example['text']
    label = example['label']
    ground_truth.append(label)
    
    # Construct prompt
    prompt = create_few_shot_prompt(text, few_shot_examples)
    
    # Measure inference time
    start_time = time.time()
    
    # Generate
    # Note: We use a low temperature for deterministic outputs
    output = pipe(prompt, temperature=0.1, do_sample=False)
    generated_text = output[0]['generated_text'].strip().upper()
    
    end_time = time.time()
    latencies.append(end_time - start_time)
    
    # Parse prediction
    if "POSITIVE" in generated_text:
        pred_label = 1
    elif "NEGATIVE" in generated_text:
        pred_label = 0
    else:
        # Fallback or "Neutral" - mapping to -1 or random could be options. 
        # Here we default to 0 (Negative) but print a warning for qualitative analysis.
        # print(f"Warning: Could not parse output: '{generated_text}'")
        pred_label = 0 
        
    predictions.append(pred_label)

#%% Evaluation
accuracy = accuracy_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions, average='weighted')
avg_latency = np.mean(latencies)
samples_per_sec = 1.0 / avg_latency

print("\n--- Evaluation Results (LLM Few-Shot) ---")
print(f"Model: {MODEL_NAME}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Avg Inference Time: {avg_latency:.4f} seconds/sample")
print(f"Inference Speed: {samples_per_sec:.2f} samples/sec")

#%% Qualitative Analysis (Optional)
print("\n--- Qualitative Errors ---")
count = 0
for i, (true, pred) in enumerate(zip(ground_truth, predictions)):
    if true != pred and count < 3:
        print(f"Example {i}:")
        print(f"Text: {small_dataset['val'][i]['text']}")
        print(f"True: {'POSITIVE' if true==1 else 'NEGATIVE'}, Pred: {'POSITIVE' if pred==1 else 'NEGATIVE'}")
        print("-" * 20)
        count += 1