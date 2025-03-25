# ===================== 📦 Importation =====================
import os, json, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from qdrant_client import QdrantClient, models
from babyagi import TaskManager
from llama_index.llama_automated_task_solver import LATS

# ===================== ⚙️ Config =====================
docs_dir = "legal_docs"  # Contient les contrats / lois
MODEL = "deepseek-ai/deepseek-7b"
qdrant_url = "http://localhost:6333"
collection = "legalrag_index"

# ===================== 🤖 Chargement du modèle =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype="auto", load_in_4bit=True, device_map="auto"
)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, task_type="CAUSAL_LM", bias="none"
))

# ===================== 🧠 Qdrant & Lecture PDF =====================
client = QdrantClient(qdrant_url)
documents = SimpleDirectoryReader(docs_dir).load_data()
index = VectorStoreIndex.from_documents(documents)

client.recreate_collection(
    collection_name=collection,
    vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE)
)
for doc in documents:
    client.upsert(
        collection_name=collection,
        points=[models.PointStruct(
            id=str(hash(doc.text)),
            payload={"text": doc.text},
            vector=index.index_struct.get_text_embedding(doc.text)
        )]
    )

# ===================== 📄 Fine-Tuning (optionnel) =====================
def prepare_ft(corpus):
    data = []
    for text in corpus:
        pairs = text.split(". ")
        for i in range(0, len(pairs)-1, 2):
            data.append({"instruction": pairs[i], "response": pairs[i+1]})
    with open("legal_ft.jsonl", "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")

# ===================== 🔍 Récupération & RAG =====================
def retrieve_context(query):
    results = client.search(
        collection_name=collection,
        query_vector=index.index_struct.get_text_embedding(query),
        limit=3
    )
    return "\n".join([r.payload["text"] for r in results]) if results else "No match found."

def legal_agent(query):
    context = retrieve_context(query)
    full_prompt = f"Context:\n{context}\n\nUser Question:\n{query}"
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_length=1024, temperature=0.5)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ===================== 🧠 Agentic Execution =====================
tm = TaskManager()
lats = LATS()

if __name__ == "__main__":
    print("🔎 LegalDocRAG is running. Ask your legal questions.")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]: break
        if len(q.split()) < 12:
            print("LegalAgent:", tm.run_task(lambda: legal_agent(q)))
        else:
            print("LegalAgent:", lats.execute_task(lambda: legal_agent(q)))