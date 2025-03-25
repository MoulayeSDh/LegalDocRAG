
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Built by Moulaye Sidi Dahi](https://img.shields.io/badge/Built%20by-Moulaye%20Sidi%20Dahi-brightgreen)](https://www.linkedin.com/in/moulayesididahi)


# ⚖️ LegalDocRAG — Agentic RAG AI for Legal Document Analysis

LegalDocRAG is a fully local, Retrieval-Augmented Generation (RAG)-powered assistant fine-tuned on legal documents.  
It supports complex legal question-answering by combining DeepSeek-7B (QLoRA), vector search (Qdrant), and hybrid autonomous orchestration (BabyAGI + LATS).



## 🚀 Key Features

✅ **RAG Pipeline**: Qdrant + LlamaIndex for high-performance vector retrieval  
✅ **Fine-Tuned LLM**: QLoRA adaptation of DeepSeek-7B on your legal corpus  
✅ **Local-Only**: 100% offline, GPU-efficient, and privacy-friendly  
✅ **Agentic Layer**: BabyAGI for sequential tasks & LATS for complex planning  
✅ **Extendable**: Swap the legal dataset to adapt for finance, policy, etc.

---

## ⚙️ Tech Stack

- **LLM Backbone**: `deepseek-ai/deepseek-7b` (4-bit quantized, fine-tuned via QLoRA)
- **Fine-Tuning**: `PEFT + LoRAConfig` on legal Q&A pairs
- **Vector Database**: `Qdrant` for fast retrieval
- **RAG Framework**: `LlamaIndex` for document parsing & embedding
- **Orchestration**: `BabyAGI` for task sequencing, `LATS` for structured task solving
- **Web Scraping (optional)**: `Crawl4AI` can be added

---

## 🛠️ Setup & Run

### 1️⃣ Install Required Libraries

```
pip install torch transformers accelerate peft datasets llama-index qdrant-client pypdf
```
2️⃣ Launch Qdrant Locally
```
docker run -p 6333:6333 qdrant/qdrant
```
3️⃣ Run the Agent
```
python legal_agent.py
```
You’ll be prompted with:

🔎 LegalDocRAG is running. Ask your legal questions.
You:


💡 Example Use Cases

🧾 Contract Analysis: “What clauses are missing from this NDA?”

📜 Law Retrieval: “What changed in tax law since 2022?”

📚 Regulation Comparison: “Difference between GDPR and CCPA?”


##📌 Conditions of Use

This project is 100% open-source, but attribution is required.

🔹 📢 Mandatory Attribution

If you use or modify LegalDocRAG, you must credit:

Author: Moulaye Sidi Dahi

GitHub Repo: https://github.com/MoulayeSDh


Violation of this condition may result in a DMCA takedown.



🔐 License - Apache 2.0 + Attribution Clause

This project uses a modified Apache 2.0 License:

✅ You can use, modify, and distribute the project

❌ You must not remove author attribution

❌ No commercial reuse without author permission



🧭 ## What's Next?

[ ] Modular version with FastAPI interface

[ ] Auto-RAG injection with document classification

[ ] CI/CD & Hugging Face Demo

