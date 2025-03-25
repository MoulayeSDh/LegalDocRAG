# ===================== 📦 Imports =====================
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient, models
from babyagi import TaskManager
from llama_index.llama_automated_task_solver import LATS

# ===================== ⚙️ Configuration =====================
docs_dir = "legal_docs"  # Directory containing legal PDFs or text files
MODEL = "deepseek-ai/deepseek-7b"
qdrant_url = "http://localhost:6333"
collection = "legalrag_index"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# ===================== 📄 Check Document Directory =====================
if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
    raise FileNotFoundError(f"Directory '{docs_dir}' is missing or empty. Please add legal documents to proceed.")

# ===================== 🤗 Load Model & Tokenizer =====================
print("🔧 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config
)

# Optional: freeze base model weights
for param in model.base_model.parameters():
    param.requires_grad = False

# Apply QLoRA with PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    bias="none"
)
model = get_peft_model(model, lora_config)

# ===================== 📚 Load & Embed Documents =====================
print("📚 Reading and embedding documents...")
documents = SimpleDirectoryReader(docs_dir).load_data()
embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# ===================== 💾 Setup Qdrant =====================
client = QdrantClient(qdrant_url)

if collection not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )

for doc in documents:
    vector = embed_model.get_text_embedding(doc.text)
    client.upsert(
        collection_name=collection,
        points=[models.PointStruct(
            id=str(hash(doc.text)),
            payload={"text": doc.text},
            vector=vector
        )]
    )

# ===================== 🧾 Fine-Tuning Corpus Prep =====================
def prepare_finetune_corpus(docs, output_file="legal_ft.jsonl"):
    """
    Generate a fine-tuning corpus using a naive instruction/response split
    based on sentence pairs.
    """
    print(f"🛠️ Preparing fine-tuning corpus → {output_file}")
    from nltk.tokenize import sent_tokenize

    data = []
    for doc in docs:
        sentences = sent_tokenize(doc.text)
        for i in range(0, len(sentences) - 1, 2):
            inst, resp = sentences[i], sentences[i + 1]
            if len(inst.split()) > 5 and len(resp.split()) > 5:
                data.append({"instruction": inst.strip(), "response": resp.strip()})

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    print(f"✅ Fine-tuning data saved to {output_file}")

# Uncomment this to prepare fine-tuning corpus
prepare_finetune_corpus(documents)

# ===================== 🔍 Context Retrieval =====================
def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve the most relevant document chunks using Qdrant vector similarity.
    """
    try:
        vector = embed_model.get_text_embedding(query)
        results = client.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k
        )
        return "\n---\n".join([r.payload["text"] for r in results])
    except Exception as e:
        print("Error retrieving context:", e)
        return "Context unavailable due to error."

# ===================== ✍️ Prompt Template =====================
PROMPT_TEMPLATE = """
You are a legal expert assistant. Given the legal context and a user's question, provide a clear and accurate answer.

Context:
{context}

User Question:
{query}

Answer:
"""

# ===================== 🤖 Legal Agent =====================
def legal_agent(query: str) -> str:
    """
    Generate an answer to a legal query based on retrieved context using a LLM.
    """
    context = retrieve_context(query)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.5,
            do_sample=False
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ===================== 🧠 Agentic Execution =====================
tm = TaskManager()
lats = LATS()

# ===================== ▶️ Main Execution Loop =====================
if __name__ == "__main__":
    print("🔎 LegalDocRAG is running. Ask your legal questions.")
    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ["exit", "quit"]:
            print("👋 Exiting LegalDocRAG.")
            break
        try:
            if len(q.split()) < 12:
                print("LegalAgent:", tm.run_task(lambda: legal_agent(q), task_description="Short legal query"))
            else:
                print("LegalAgent:", lats.execute_task(lambda: legal_agent(q), task_description="Complex legal analysis"))
        except Exception as err:
            print("⚠️ Error:", err)
