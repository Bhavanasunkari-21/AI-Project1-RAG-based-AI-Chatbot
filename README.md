***Multi-Document RAG Chatbot***

Ask questions across an entire document library — not just one file.

Upload up to 10 PDFs at once. Ask one question. Get a single answer synthesised across all of them.
Built with Google Gemini, ChromaDB, HuggingFace embeddings, and Streamlit.

##Why This Is Different from a Regular Chatbot
Most AI chat tools — including ChatGPT with file uploads — let you upload one document at a time and ask questions about it. This system is built specifically for working across a library of documents simultaneously. Here is what that makes possible:

Scenario	Regular chatbot	This system
Compare two resumes against a job spec	Upload one at a time, ask three separate questions, manually compare answers	Upload all three at once, ask once, get a single synthesised comparison
Find contradictions across policy docs	Not possible in one session	Retrieves relevant chunks from all docs, surfaces conflicts in one answer
Research across 10 papers	10 separate sessions or copy-paste	One session, cross-document retrieval, cited page numbers for every claim
Contract vs regulation compliance check	Switch between files manually	Ask the gap question once, system finds the relevant clause from each file
Onboarding: HR handbook + role guide + org chart	Three uploads, three chats	Single upload batch, ask any onboarding question, get one clear answer

Every answer includes a Sources panel showing exactly which file and page number each piece of information came from — so you can verify every claim, not just trust it.

##What It Does

•	Accepts up to 10 PDFs simultaneously in a single upload batch.

•	Validates file sizes before processing — 10 MB per file, 50 MB total.

•	Splits all documents into overlapping chunks and indexes them together into one searchable vector database.

•	At query time, retrieves the most relevant chunks from across the entire document set — not just one file.

•	Generates a single grounded answer synthesised from multiple sources.

•	Cites the exact file name and page number for every piece of retrieved content.

•	Preserves full chat history so you can ask follow-up questions across sessions.

•	Automatically handles API rate limits with retry logic and model fallback

##How It Works
The system runs two phases. The indexing phase runs once when you upload your documents. The query phase runs on every question.

Indexing phase  (runs once per document batch)
All PDFs uploaded simultaneously
      |
      v
PyPDFLoader  --  each file parsed page by page
      |
      v
RecursiveCharacterTextSplitter
      1000-char chunks  |  150-char overlap  |  source file + page tagged on every chunk
      |
      v
all-MiniLM-L6-v2  --  384-dim embeddings  --  runs locally, no API cost
      |
      v
ChromaDB  --  all chunks from all files indexed into one vector store

Query phase  (runs on every question)
User question
      |
      v
Embed question with same MiniLM model
      |
      v
Cosine similarity search  --  top 4 chunks retrieved from across ALL uploaded files
      |
      v
Prompt:  "Use only this context. Synthesise across documents if needed."
      + retrieved chunks with [Source: filename, Page N] labels
      + user question
      |
      v
Google Gemini 2.0 Flash  --  grounded answer with cross-document synthesis
      |
      v
Answer + collapsible Sources panel (file name + page number for every chunk used)

##Real-World Use Cases

Recruitment
Upload: 5 candidate resumes + 1 job description.  Ask: "Rank these candidates by fit and explain the gaps for each."  The system reads all six files and returns a ranked comparison in one response, citing which resume page each assessment came from.

Legal and compliance
Upload: A supplier contract + your internal procurement policy + relevant regulation.  Ask: "Where does this contract conflict with our policy or the regulation?"  The system finds the relevant clause from each document and surfaces the conflict directly.

Academic research
Upload: 10 research papers on a topic.  Ask: "What do these papers disagree on, and which methodology appears most often?"  Cross-paper synthesis that would take hours manually is returned in seconds, with citations.

Employee onboarding
Upload: HR handbook + role-specific guide + org chart + benefits document.  Ask: "What is the process for requesting leave and who do I report to?"  One answer drawn from multiple internal documents — no more hunting through four separate files.

Financial analysis
Upload: Annual reports from three competitor companies.  Ask: "Compare R&D spending as a percentage of revenue across all three."  The system extracts and compares the relevant figures from all three reports simultaneously.


##Technology Stack

Component	Technology and detail
Embedding model	sentence-transformers/all-MiniLM-L6-v2  —  384-dim, runs locally, no API cost
Language model	Google Gemini 2.0 Flash via google.genai SDK  —  auto-fallback across model tiers
Vector store	ChromaDB  —  local persistent index, cosine similarity, unique dir per session
PDF parsing	PyPDFLoader (LangChain)  —  page-by-page with source metadata tagging
Text splitting	RecursiveCharacterTextSplitter  —  1000-char chunks, 150-char overlap
Interface	Streamlit  —  session-aware chat UI with live upload progress

##Rate Limits and Automatic Fallback
The system handles Google API quota errors automatically. When a 429 error is returned, it waits the exact delay suggested in the error response and retries up to 3 times. If the quota is exhausted, it silently switches to the next model in the chain:

gemini-2.0-flash-lite  ->  gemini-2.0-flash  ->  gemini-2.0-flash-001  ->  gemini-2.5-flash  ->  gemini-2.5-pro

Model	Free requests/min	Free requests/day
gemini-2.0-flash-lite	30 RPM	1,500
gemini-2.0-flash	15 RPM	1,500
gemini-2.5-flash	10 RPM	500


##Setup and Installation

1.  Clone the repository
git clone https://github.com/Bhavanasunkari-21/AI-Project1-RAG-based-AI-Chatbot.git
cd AI-Project1-RAG-based-AI-Chatbot

2.  Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate      # macOS / Linux

3.  Install dependencies
pip install -r requirements.txt

# Or manually:

pip install streamlit python-dotenv google-genai \

            langchain langchain-core langchain-community \
            
            langchain-huggingface sentence-transformers \
            chromadb pypdf

4.  Add your Google API key
   
5.  Create a .env file in the project root:
GOOGLE_API_KEY=AIza...your_key_here

6. Get a free key at https://aistudio.google.com/app/apikey

8.  Run the app
streamlit run app.py





Project Structure


AI-Project1-RAG-based-AI-Chatbot/

|
|-- app.py                # Main application  --  upload, index, chat, retry logic

|-- requirements.txt      # All dependencies

|-- .env                  # Your API key  (never commit this)

|-- .gitignore            # Excludes venv/, chroma_db_*/, tempDir/, .env

L-- README.md




Security Notes

•	Never commit the .env file — it is excluded by .gitignore

•	Uploaded PDFs are written to tempDir/ for processing and never stored permanently

•	ChromaDB indexes are local to your machine and not transmitted anywhere

•	This project is intended for local development and experimentation
