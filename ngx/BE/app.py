import os
import time
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for specific frontend origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError("API key not found in environment variables")

# Initialize LLM (Language Model)
llm = ChatGroq(api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Initialize embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors = None

# Store a list of courses and their embeddings
courses = []

class QueryRequest(BaseModel):
    query: str

@app.post("/create-embedding")
async def create_embedding(file: UploadFile = File(...)):
    global vectors, courses

    # Save the uploaded PDF file to a temporary location
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Load and process the PDF file
    loader = PyPDFLoader(file_location)
    data = loader.load()

    # Split the text into manageable chunks for embedding
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(data[:20])  # Process first 20 pages for now

    # Create and store embeddings in FAISS
    vectors = FAISS.from_documents(final_documents, embeddings)

    # Add the course (filename without extension) to the course list
    course_name = file.filename.replace(".pdf", "")
    courses.append(course_name)

    return {
        "message": "Embedding created successfully",
        "course_name": course_name,
        "total_courses": len(courses)
    }

@app.post("/query")
def query_bot(request: QueryRequest):
    global vectors

    if vectors is None:
        raise HTTPException(status_code=400, detail="Please create embeddings first")

    query = request.query

    # Build the document retrieval and response chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Process the query and measure time
    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': query})
    response_time = time.process_time() - start_time

    return {
        "answer": response['answer'],
        "response_time": response_time,
        "context": response.get("context", "No context available")
    }

@app.get("/courses")
def get_courses():
    return {"courses": courses}

@app.get("/")
def root():
    return {"message": "Welcome to the Chatbot API"}
