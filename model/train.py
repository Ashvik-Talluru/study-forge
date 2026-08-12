import os
import sys
import time

# Auto-fix Python path so running directly never throws ModuleNotFoundError
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from model.transformer import StudyForgeTransformer

# ==============================================================================
# UPGRADED HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 32
BLOCK_SIZE = 512       # UPGRADED: Increased from 128 to 512 so it can hold longer paragraphs
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 20
N_EMBD = 256
N_HEAD = 8  
N_LAYER = 6
DROPOUT = 0.2

# Paths pointing directly to textbook corpus and model saving
CORPUS_DIR = os.path.join(project_root, "data", "inputs", "textbook_corpus")
SAVE_PATH = os.path.join(project_root, "out", "studyforge_model.pth")

print("=" * 70, flush=True)
print(f"STARTING STUDYFORGE MODEL TRAINING PIPELINE [{str(DEVICE).upper()}]", flush=True)
print("=" * 70, flush=True)


def load_corpus(directory_path):
    """Reads all .txt files from target corpus directory with real-time progress."""
    raw_text = ""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"[!] Target corpus directory did not exist. Created: {directory_path}", flush=True)
        return raw_text

    txt_files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
    total_files = len(txt_files)
    print(f"[*] Discovered {total_files} text file(s) inside '{directory_path}'.", flush=True)

    for i, filename in enumerate(txt_files, 1):
        filepath = os.path.join(directory_path, filename)
        print(f"    -> Reading file {i}/{total_files}: {filename} ...", flush=True)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw_text += f.read() + "\n"

    return raw_text


# ------------------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------------------
print("\n[*] Loading training data...", flush=True)
text = load_corpus(CORPUS_DIR)

if not text.strip():
    raise ValueError(f"No text data found in '{CORPUS_DIR}'. Please add your .txt files there!")

print(f"[+] Corpus successfully loaded. Total characters: {len(text):,}", flush=True)

# ------------------------------------------------------------------------------
# 2. Vocabulary & Encoding
# ------------------------------------------------------------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"[+] Unique vocabulary size: {vocab_size} characters", flush=True)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]  # Added safety check for missing characters
decode = lambda l: "".join([itos[i] for i in l])

print("[*] Encoding corpus into PyTorch Tensors...", flush=True)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"[+] Train/Val Split complete: {len(train_data):,} train tokens, {len(val_data):,} val tokens", flush=True)


def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_split[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss(model_instance):
    out = {}
    model_instance.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model_instance(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model_instance.train()
    return out


# ------------------------------------------------------------------------------
# 3. Instantiate Model & Optimizer
# ------------------------------------------------------------------------------
print("\n[*] Initializing StudyForge Transformer Model...", flush=True)
model = StudyForgeTransformer(
    vocab_size=vocab_size,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("[+] Model setup complete. Beginning training loop...\n", flush=True)

# ------------------------------------------------------------------------------
# 4. Training Loop
# ------------------------------------------------------------------------------
start_time = time.time()
for iter_num in range(1, MAX_ITERS + 1):
    if iter_num % EVAL_INTERVAL == 0 or iter_num == 1:
        losses = estimate_loss(model)
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"Step {iter_num:4d}/{MAX_ITERS} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Elapsed: {elapsed}", flush=True)

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save Checkpoint
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"\n[+] Training complete! Model weights saved to '{SAVE_PATH}'", flush=True)


# ==============================================================================
# NEW DETAILED TESTING LOOP 
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("SWITCHING TO INTERACTIVE MODEL TESTING MODE", flush=True)
print("=" * 70, flush=True)

# Set model to evaluation mode for testing
model.eval()

print("\nType your textbook question below. Type 'exit' to stop.", flush=True)
while True:
    user_question = input("\n[Question]: ")
    if user_question.strip().lower() == 'exit':
        print("[*] Exiting test mode. Goodbye!", flush=True)
        break
        
    if not user_question.strip():
        continue

    # 1. Format the question cleanly as text context
    prompt_text = f"\nQuestion: {user_question}\nAnswer:"
    
    # 2. Trim prompt if it exceeds our model's capacity limit
    if len(prompt_text) > BLOCK_SIZE:
        prompt_text = prompt_text[-BLOCK_SIZE:]
        
    # 3. Convert character string into token integers and move to GPU/CPU
    encoded_prompt = encode(prompt_text)
    context_tensor = torch.tensor([encoded_prompt], dtype=torch.long, device=DEVICE)
    
    print("[Answering] Generating response...", end="", flush=True)
    
    # 4. Generate tokens using a high count limit (1500 characters) so it never truncates early
    with torch.no_grad():
        # Ensure your model.transformer file features a .generate(context, max_new_tokens) method
        generated_tokens = model.generate(context_tensor, max_new_tokens=1500)[0].tolist()
        
    # 5. Decode back into characters and clean out the input prompt text from the display
    full_output_text = decode(generated_tokens)
    final_answer = full_output_text[len(prompt_text):]
    
    print("\n" + "-" * 40)
    print(final_answer.strip())
    print("-" * 40, flush=True)
