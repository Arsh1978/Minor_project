from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

app = Flask(__name__)

# Load model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create question-answering pipeline
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Create FAISS index
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_texts([""], embeddings)

def process_pdfs(company_path, job_path):
    # Load PDFs
    company_loader = PyPDFLoader(company_path)
    job_loader = PyPDFLoader(job_path)

    company_data = company_loader.load()
    job_data = job_loader.load()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    company_texts = text_splitter.split_documents(company_data)
    job_texts = text_splitter.split_documents(job_data)

    # Combine texts
    all_texts = company_texts + job_texts

    # Update FAISS index
    global db
    db = FAISS.from_documents(all_texts, embeddings)

@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    if 'company_description' not in request.files or 'job_role' not in request.files:
        return jsonify({"error": "Please upload both PDFs"}), 400

    company_pdf = request.files['company_description']
    job_pdf = request.files['job_role']

    # Save the uploaded PDFs temporarily
    company_path = 'temp_company.pdf'
    job_path = 'temp_job.pdf'
    company_pdf.save(company_path)
    job_pdf.save(job_path)

    # Process the PDFs
    process_pdfs(company_path, job_path)

    return jsonify({"message": "PDFs processed successfully"}), 200

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    question = data['question']

    # Create RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(pipeline=qa_pipeline),
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    # Get answer
    result = qa.run(question)

    return jsonify({"answer": result})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
