from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import transformers

print(f"Transformers version: {transformers.__version__}")

# Load environment variables
load_dotenv()
groq_api_key = "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"

# Initialize the chatbot
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

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

def vector_embedding(file_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(data[:20])
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

def create_embeddings(file_path):
    vectors = vector_embedding(file_path)
    print("Vector Store DB Is Ready")
    return vectors

def handle_query(vectors, query):
    if vectors:
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({'input': query})
        return response['answer'], response['context']
    else:
        raise ValueError("Please create the embedding first by providing a valid file path.")

@app.route('/api/summarize', methods=['POST'])
def summarize():

    summaries = []
    files = request.files.getlist('files')
    for file in files:
        file_path = f"./dl/{file.filename}"
        file.save(file_path)
        
        vectors = create_embeddings(file_path)
        query = "Summarize this course"
        answer, _ = handle_query(vectors, query)
        summaries.append({file.filename: answer})
    
    print(summaries)
    return jsonify({"summaries": summaries})

if __name__ == "__main__":

    app.run(debug=True)
