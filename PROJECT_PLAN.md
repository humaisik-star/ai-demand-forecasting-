# One-Month Project Plan: Local RAG AI Assistant with Microsoft Foundry Local

This plan guides beginner computer science students through a full-time, one-month summer program. The final project is a local, offline Q&A chatbot built with Retrieval-Augmented Generation (RAG) and Microsoft Foundry Local.

## Project Overview

**Goal:** Build a local document Q&A assistant that answers questions from a private document collection entirely offline.

**Key features:**
- Document ingestion and local knowledge base creation
- Embedding-based semantic search
- SQLite-backed vector store for local retrieval
- Offline LLM inference with Microsoft Foundry Local
- Prompt engineering for grounded, source-aware answers

**Why this matters:** students learn how modern AI apps combine retrieval and generation to produce accurate, trustable answers without network dependency.

---

## Phase 1: Foundational Learning (Weeks 1–2)

### Week 1: RAG Concepts & Foundry Local Setup

**Learning objectives:**
- Understand Retrieval-Augmented Generation
- Install and verify Foundry Local
- Build a minimal Python project structure

**Topics & activities:**
- RAG overview: retrieve, augment, generate
- Why RAG reduces hallucinations
- Foundry Local introduction: offline model runtime, local inference, CPU/NPU support
- Python project basics: `main.py`, functions, `requirements.txt`

**Hands-on exercises:**
- Read the Microsoft Tech Community example: "Building Your First Local RAG Application with Foundry Local"
- Install Foundry Local SDK and run a small test inference
- Create a skeleton Python project and print a greeting

**Milestone:**
- Students have Foundry Local installed and working
- A basic Python app can call the local LLM successfully

### Week 2: Embeddings, Vector Search & SQLite

**Learning objectives:**
- Learn how embeddings represent text meaning
- Understand cosine similarity and semantic retrieval
- Use SQLite for local data storage
- Practice prompt-engineering basics

**Topics & activities:**
- Text embeddings and vector similarity
- Building a lightweight vector store with SQLite
- System vs user prompts for grounded Q&A
- Responsible instructions: say "I don’t know" when context is missing

**Hands-on exercises:**
- Generate embeddings for sample sentences and query similarity
- Create a SQLite database with `documents`, `chunks`, and `embeddings`
- Write a retrieval prototype that finds top relevant chunks
- Experiment with prompt context and answer quality

**Milestone:**
- Students can generate and compare embeddings
- A SQLite-backed retrieval workflow is understood and tested

---

## Phase 2: Project Implementation (Weeks 3–4)

### Week 3: Data Ingestion & Retrieval Pipeline

**Learning objectives:**
- Build the knowledge base ingestion pipeline
- Split documents into chunks for retrieval
- Store chunk embeddings in SQLite
- Retrieve relevant content for a query

**Topics & activities:**
- Document selection and chunking strategy
- Embedding each chunk with Foundry Local
- Storing embeddings and metadata in SQLite
- Implementing a top-K retrieval function

**Hands-on exercises:**
- Choose a small document set (notes, FAQs, manuals)
- Write ingestion code for chunking, embedding, and storing
- Build and test `get_top_chunks(query)`
- Validate retrieval results with sample questions

**Milestone:**
- A populated local knowledge base exists
- The app can retrieve relevant document chunks for a query

### Week 4: LLM Integration & Application Assembly

**Learning objectives:**
- Connect retrieved context to the local LLM
- Build a complete question-answering pipeline
- Create a usable interface for the chatbot

**Topics & activities:**
- Loading a Foundry Local model for chat-based generation
- Composing prompts with retrieved context
- Option A: console-based Q&A interface
- Option B: Streamlit or Gradio web UI (optional)

**Hands-on exercises:**
- Implement `answer_query(user_question)` using retrieved chunks
- Add a system prompt instructing the model to base answers on context
- Build a CLI or lightweight web front end
- Test end-to-end with multiple user queries

**Stretch goals:**
- Add source citation labels to retrieved chunks
- Improve prompt wording to reduce hallucinations
- Display the retrieved context for debugging

**Milestone:**
- Students have a working offline RAG chatbot
- The app answers questions from local documents accurately

---

## Phase 3: Testing, Evaluation & Documentation (Week 5–6)

### Week 5: System Testing & Evaluation

**Learning objectives:**
- Validate QA behavior across real queries
- Measure answer quality and robustness
- Identify and fix common failure modes

**Topics & activities:**
- Functional testing with answerable and unanswerable questions
- Response time and performance evaluation
- Iterating on prompt templates and retrieval settings

**Hands-on exercises:**
- Create a test set of questions and expected results
- Run the assistant against the test set and record outcomes
- Compare answers to source documents and assess correctness
- Tune retrieval count and prompt phrasing as needed

**Milestone:**
- Teams document test results and improvements
- The assistant reliably returns grounded answers or safe fallback responses

### Week 6: Documentation & Final Presentation

**Learning objectives:**
- Produce clear project documentation
- Prepare a concise demo and reflection
- Communicate technical decisions and lessons learned

**Topics & activities:**
- Final README/project report: purpose, setup, architecture, limitations
- Code cleanup, comments, and dependency management
- Demo preparation: problem statement, features, live Q&A, lessons learned

**Hands-on exercises:**
- Write a project summary and usage instructions
- Polish the interface and remove debug output
- Practice a short demo showing the chatbot answering questions
- Share challenges and next steps

**Milestone:**
- Teams deliver a polished project with documentation
- Students present a working local RAG assistant and explain their approach

---

## Recommended Project Architecture

**Core components:**
- `data/` – source documents and ingestion scripts
- `models/` – Foundry Local model configuration (local runtime)
- `db/` – SQLite file with chunks and embeddings
- `app/` – retrieval and LLM orchestration logic
- `ui/` – CLI or web interface for Q&A

**Pipeline flow:**
1. User asks a question
2. The app embeds the query
3. The app retrieves top relevant chunks from SQLite
4. The app sends the query + retrieved context to the local LLM
5. The model returns a grounded answer
6. The answer is displayed to the user

---

## Resources

- Microsoft Tech Community blog: Building Your First Local RAG Application with Foundry Local
- Microsoft Learn: Foundry Local overview and Python quickstart
- Microsoft Learn: Build a RAG application tutorial
- SQLite documentation for local data storage
- Prompt engineering guides for system and user messages

---

## Notes for ML Topic Adaptation

If your project topic is different but still ML-focused, keep the same structure:
- Week 1–2: learn the core concepts and tools
- Week 3–4: implement the data pipeline and model integration
- Week 5–6: test, refine, document, and present

### Choosing your ML topic

For a different ML domain, keep the same workflow while changing the content and data source:
- Select a focused dataset or document collection relevant to your topic
- Define the user problem clearly (e.g., FAQ assistant, study guide helper, troubleshooting support)
- Use the same local retrieval + model pipeline, but replace the documents and prompts with your domain-specific material

This plan is flexible: the final app can be adapted to any offline ML-driven assistant, knowledge discovery tool, or small local AI product.
