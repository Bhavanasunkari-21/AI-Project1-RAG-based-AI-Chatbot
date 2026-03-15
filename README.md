

***RAG-Based PDF Q&A Chatbot***

This project is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload PDF documents and ask questions about their content. The system retrieves relevant information from the uploaded PDFs and generates answers grounded in the document context.

***Overview***
The application extracts text from uploaded PDFs, converts the text into embeddings, stores them in a vector database, and retrieves the most relevant chunks when a user asks a question. The retrieved context is then sent to a language model to generate an accurate answer.

##Features
•	Upload multiple PDF files
•	Ask questions in natural language
•	Answers generated using document context
•	Source citations showing file and page number
•	Vector search using ChromaDB
•	Fast responses using Google Gemini models
•	Interactive chat interface built with Streamlit


## Technologies Used
•	Python
•	Streamlit
•	LangChain
•	ChromaDB
•	HuggingFace Embeddings
•	Google Gemini LLM
•	PyPDF

***Project Workflow***
1.	Upload PDF documents
2.	Extract text from PDFs
3.	Split text into smaller chunks
4.	Convert chunks into embeddings
5.	Store embeddings in ChromaDB
6.	Retrieve relevant chunks for the question
7.	Send context and question to the LLM
8.	Generate an answer with references



***Project Structure***

app.py – Main Streamlit application
tempDir/ – Temporary storage for uploaded PDFs
chroma_db/ – Vector database storing embeddings
.env – API keys and environment variables
README.md – Project documentation


***How to Run***
1. Clone the repository
git clone https://github.com/your-username/rag-pdf-chatbot.git
2. Create virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install streamlit langchain chromadb sentence-transformers pypdf python-dotenv google-genai
4. Add API key in .env file
GOOGLE_API_KEY=your_api_key
5. Run the application
streamlit run app.py
