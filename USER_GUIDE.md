# RAG PDF Chatbot: Comprehensive User Guide 📖

Welcome to the **RAG PDF Chatbot**! This guide walks you through setting up, running, and getting the most out of your document intelligence assistant.

---

## 1. Quick Start

### 1.1 Launching the Application
Make sure your virtual environment is active, then run:

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

The application will start automatically in your default browser at:
👉 **[http://localhost:8501](http://localhost:8501)**

### 1.2 Supplying Your API Key
* **Via `.env` file (Recommended)**: Set your `GEMINI_API_KEY` inside `.env`.
* **Via the UI Sidebar**: If no key is set in `.env`, a secure password input box appears in the left sidebar where you can paste your key during your session.

---

## 2. Using the Application

### 2.1 Uploading Documents
1. Locate the **"📄 Upload Documents"** box in the left sidebar.
2. Drag and drop one or more `.pdf` files (or click **Browse files**).
3. The system will automatically:
   - Verify the file is a valid PDF and under the 20 MB size limit.
   - Compute its unique **SHA-256 content hash** to prevent duplicate work.
   - Extract selectable text page by page.
   - Split pages into **800-character chunks** with **150-character overlap**.
   - Embed and store vectors into the persistent Chroma database.

> [!NOTE]
> **Scanned / Image-only PDFs**: The current engine extracts selectable digital text. If you upload a scanned image without selectable text, the system displays a clear warning advising that OCR is required.

---

### 2.2 Selecting Query Target (Document Filter)
Once documents are indexed, the sidebar displays an active inventory of your knowledge base with page and chunk counts.

* **"All Documents" (Default)**: Searches across every document in your knowledge base simultaneously.
* **Specific Document**: Select an individual document from the dropdown to constrain your questions to that specific file (filtered securely by `file_hash`).

---

### 2.3 Asking Questions & Conversational Memory
* Type your question in the chat bar at the bottom and hit **Enter**.
* **Multi-Turn Follow-Ups**: The chatbot maintains context across the last 6 messages. You can ask:
  > **User:** *"What is the meal allowance during business trips?"*
  >
  > **Assistant:** *"Daily meal reimbursement is capped at $75 per day..."*
  >
  > **User:** *"Are there any exceptions or excluded items for it?"*
  
  The system automatically contextualizes *"it"* into *"business travel meal allowances"* before performing vector retrieval.

---

### 2.4 Inspecting Citations & Sources
Beneath every assistant response, click the **"📚 View Source Citation(s)"** expander to inspect:
* **Source Document**: The exact filename.
* **Page Number**: The precise 1-indexed page where the information lives.
* **Verbatim Snippet**: The source passage from the document that informed the answer.

---

### 2.5 Grounding & Refusal Behavior
If you ask a question that is **not answered in your documents** (e.g. *"What is the distance from Earth to Jupiter?"* or *"Who won the 1998 World Cup?"*):
* The system calculates cosine similarity against your chunks.
* Because the similarity falls below the calibrated `RELEVANCE_THRESHOLD` (default `0.34`), the system **bypasses the LLM entirely**.
* It immediately responds:
  > *"I could not find information about that in the uploaded document(s)."*
* No source citations are shown, eliminating hallucinations and saving API token costs.

---

### 2.6 Management Controls
In the sidebar under the document inventory:
* **Clear Chat**: Clears current conversation history on screen while keeping your indexed documents intact.
* **Reset KB**: Completely wipes the ChromaDB vector database and resets the document registry, allowing you to start completely fresh.

---

## 3. Hands-On Tutorial: Testing the Bundled Sample Policy

A test corporate handbook is pre-bundled at `data/sample_docs/sample_policy.pdf`.

### Try Uploading `sample_policy.pdf` and Testing These Questions:

| Page | Topic | Try Asking |
| :--- | :--- | :--- |
| **Page 1** | Remote Work | *"What are the core collaboration hours for remote team members?"* |
| **Page 1** | Home Office | *"What is the home internet speed requirement and office stipend?"* |
| **Page 2** | Travel & Meals | *"What is the daily meal per diem and what purchases are excluded?"* |
| **Page 2** | Hotels | *"What is the maximum allowed hotel rate per night?"* |
| **Page 3** | PTO | *"How many days of paid time off do employees accrue and can they rollover?"* |
| **Page 3** | Parental Leave | *"What are the parental leave benefits for primary vs secondary caregivers?"* |
| **Page 4** | Security | *"Within what timeframe must lost hardware be reported to IT?"* |
| **Negative Test** | Off-Topic | *"How do I bake chocolate chip cookies?"* *(Should refuse without citations)* |

---

## 4. Running the Evaluation Benchmark

To verify retrieval quality or recalibrate the relevance threshold after changing chunking settings or embedding models:

```powershell
.\venv\Scripts\python.exe tests/sample_eval.py
```

### Benchmark Metrics:
* **Retrieval Top-4 Hit Rate**: Verifies that the ground-truth page is retrieved in top-4 results (achieves **100.0%** on sample dataset).
* **Threshold Calibration**: Compares similarity score distributions between in-document and off-topic queries to provide a recommended `RELEVANCE_THRESHOLD`.

---

## 5. Troubleshooting & FAQs

### Q: Why do I see "Embedding Model Mismatch"?
* **Cause**: You switched `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL` in `.env` after having already indexed documents with a different model.
* **Fix**: Click the red **"Reset Vector Store to Fix Mismatch"** button in the sidebar (or delete `./data/chroma_db/`), then re-upload your documents.

### Q: Why is my question refused with "I could not find information..."?
* **Cause 1**: The answer is genuinely not in the uploaded documents.
* **Cause 2**: If you have selected a specific document in the **Query Target** dropdown, the question might be located in a *different* uploaded document. Switch the target to **"All Documents"**.
* **Cause 3**: Your `RELEVANCE_THRESHOLD` might be set too high in `.env`. Run `python tests/sample_eval.py` to see your model's score distribution.

### Q: Are my documents sent to third-party services?
* If `EMBEDDING_PROVIDER=local` (default), embeddings are generated 100% locally on your computer CPU for $0 cost.
* Only the retrieved relevant text snippets are sent to Google Gemini (or OpenAI) to formulate the final grounded natural language answer.
