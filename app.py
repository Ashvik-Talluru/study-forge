import base64
import hashlib
import json
import os

import streamlit as st
import torch

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from model.transformer import StudyForgeTransformer
from uploads.file_parser import extract_text_from_file

# Streamlit Page Config
st.set_page_config(
    page_title="StudyForge",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# 1. RAG & Prompt Formatting Helpers (Solutions 1 & 2)
# -----------------------------------------------------------------------------
def find_relevant_context(user_query, knowledge_text, max_chars=100):
    """Solution 2: Searches uploaded knowledge base for keywords matching query."""
    if not knowledge_text:
        return ""

    keywords = [w.lower() for w in user_query.split() if len(w) > 3]
    sentences = [
        s.strip()
        for s in knowledge_text.replace("\n", " ").split(".")
        if s.strip()
    ]

    best_sentence = ""
    max_matches = 0

    for sentence in sentences:
        matches = sum(1 for kw in keywords if kw in sentence.lower())
        if matches > max_matches:
            max_matches = matches
            best_sentence = sentence

    if max_matches > 0:
        return best_sentence[:max_chars]

    return ""


def prepare_model_input(prompt, knowledge_base):
    """Combines RAG context retrieval and auto-transformation rules."""
    # Try Solution 2 (Knowledge Base Match)
    retrieved_snippet = find_relevant_context(prompt, knowledge_base)
    if retrieved_snippet:
        return f"Context: {retrieved_snippet}\nExplanation:"

    # Fallback to Solution 1 (Auto-transform natural user prompts)
    clean_prompt = prompt.strip().lower()

    if clean_prompt.startswith(("how to", "how do you", "how can i")):
        concept = (
            prompt.lower()
            .replace("how to", "")
            .replace("how do you", "")
            .replace("how can i", "")
            .strip(" ?.")
        )
        return f"To {concept}, the standard procedure is"

    elif clean_prompt.startswith(("what is", "what are", "define")):
        concept = (
            prompt.lower()
            .replace("what is", "")
            .replace("what are", "")
            .replace("define", "")
            .strip(" ?.")
        )
        return f"In academic study, {concept} is defined as"

    elif clean_prompt.startswith(("explain", "describe")):
        concept = (
            prompt.lower()
            .replace("explain", "")
            .replace("describe", "")
            .strip(" ?.")
        )
        return f"An overview of {concept}:"

    else:
        return f"Q: {prompt}\nA:"


# -----------------------------------------------------------------------------
# 2. Cached Model & Vocabulary Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_studyforge_model():
    checkpoint_path = os.path.join("out", "studyforge_model.pth")
    corpus_dir = os.path.join("data", "inputs", "textbook_corpus")

    raw_text = ""
    if os.path.exists(corpus_dir):
        for filename in os.listdir(corpus_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(corpus_dir, filename)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text += f.read() + "\n"

    if not raw_text:
        raw_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_\n"

    chars = sorted(list(set(raw_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = StudyForgeTransformer(
        vocab_size=len(chars),
        n_embd=256,
        n_head=8,
        n_layer=6,
        block_size=128,
    ).to(device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        model.eval()

    return model, stoi, itos, device


# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if "auth_key" not in st.session_state:
    st.session_state.auth_key = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = ""


# -----------------------------------------------------------------------------
# 4. Authentication View
# -----------------------------------------------------------------------------
if not st.session_state.auth_user:
    st.title("Welcome to StudyForge")
    st.subheader("Please sign in to access your personal agent.")

    email_input = st.text_input("Email Address", placeholder="yourname@gmail.com")
    password_input = st.text_input("Password", type="password")

    if st.button("Sign In"):
        if email_input and password_input:
            salt = hashlib.sha256(email_input.encode()).digest()[:16]
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password_input.encode()))
            st.session_state.auth_key = key.decode()
            st.session_state.auth_user = email_input
            st.rerun()
        else:
            st.error("Please fill in both email and password fields.")

    st.stop()


# -----------------------------------------------------------------------------
# 5. Main App Setup & State Recovery
# -----------------------------------------------------------------------------
user_identity = st.session_state.auth_user
st.title("Welcome to StudyForge!")
st.write(f"Logged in as: **{user_identity}**")

hashed_user = hashlib.sha256(user_identity.encode()).hexdigest()
SAVE_FILE = f"history_{hashed_user}.json"

if not st.session_state.messages:
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            encrypted_data = json.load(f)
        try:
            f_obj = Fernet(st.session_state.auth_key.encode())
            decrypted_bytes = f_obj.decrypt(encrypted_data.encode())
            st.session_state.messages = json.loads(decrypted_bytes.decode())
        except Exception:
            st.error("Failed to safely decode conversation history.")

model, stoi, itos, device = load_studyforge_model()


# -----------------------------------------------------------------------------
# 6. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Chatbot controls")
    st.caption("Customize how you want the chatbot to run")
    st.divider()

    if st.button("Sign Out"):
        st.session_state.auth_user = None
        st.session_state.auth_key = None
        st.session_state.messages = []
        st.session_state.knowledge_base = ""
        st.rerun()

    st.divider()
    st.subheader("Chatbot Precision")

    precision_setting = st.slider(
        "Select precision (Higher = More Focused / Deterministic)",
        min_value=1,
        max_value=10,
        value=8,
    )

    temperature = max(0.1, round((11 - precision_setting) / 10.0, 2))
    st.caption(f"Sampling Temperature: {temperature}")

    st.divider()
    st.subheader("Study Material Upload")

    uploaded_files = st.file_uploader(
        "Upload your study material here",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"Successfully uploaded {len(uploaded_files)} file(s).")
        combined_text = ""
        for file in uploaded_files:
            file_text = extract_text_from_file(file)
            combined_text += file_text + "\n"
        st.session_state.knowledge_base = combined_text

    st.divider()
    st.subheader("Save the chat")

    if st.button("Save chat"):
        f_obj = Fernet(st.session_state.auth_key.encode())
        raw_json_str = json.dumps(st.session_state.messages)
        encrypted_str = f_obj.encrypt(raw_json_str.encode()).decode()

        with open(SAVE_FILE, "w") as f:
            json.dump(encrypted_str, f)

        st.success("Chat saved successfully!")


# -----------------------------------------------------------------------------
# 7. Chat Interface & Inference Execution
# -----------------------------------------------------------------------------
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("Hello! How can I help you with your schoolwork today?")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything about your schoolwork!")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Transform input prompt using Solutions 1 & 2
    model_input = prepare_model_input(prompt, st.session_state.knowledge_base)

    # Ensure input fits context window (128 chars)
    context_text = model_input[-120:] if len(model_input) > 120 else model_input
    encoded_tokens = [stoi[c] for c in context_text if c in stoi]

    if encoded_tokens:
        context_tensor = torch.tensor(
            encoded_tokens, dtype=torch.long, device=device
        ).unsqueeze(0)

        with st.spinner("Generating response..."):
            with torch.no_grad():
                generated = model.generate(
                    context_tensor,
                    max_new_tokens=250,
                    temperature=temperature,
                    top_k=40,
                )

        generated_ids = generated[0].tolist()
        new_ids = generated_ids[len(encoded_tokens) :]
        response_text = "".join([itos[i] for i in new_ids if i in itos])

        if not response_text.strip():
            response_text = "".join([itos[i] for i in generated_ids if i in itos])
    else:
        response_text = "I couldn't process those characters based on my training vocabulary."

    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})