from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # needed for flash messages
app.config['UPLOAD_FOLDER'] = 'uploads/'  # create this folder in your project directory

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def process_pdfs(company_path, job_path):
    # Extract text from the PDFs
    company_text = get_pdf_text([company_path])
    job_text = get_pdf_text([job_path])

    # Combine texts
    all_texts = company_text + "\n" + job_text

    # Split text into chunks and create FAISS index
    text_chunks = get_text_chunks(all_texts)
    get_vector_store(text_chunks)

jobs = [
    {
        'id': 1,
        'title': 'Software Engineer',
        'company': 'Tech Co.',
        'location': 'San Francisco, CA',
        'description': 'We are looking for a talented software engineer to join our team.',
        'responsibilities': [
            'Develop and maintain web applications',
            'Collaborate with cross-functional teams',
            'Write clean, efficient, and maintainable code'
        ],
        'skills': ['Python', 'JavaScript', 'SQL', 'Git']
    },
    {
        'id': 2,
        'title': 'Data Scientist',
        'company': 'Data Insights Inc.',
        'location': 'New York, NY',
        'description': 'Join our data science team to solve complex problems using machine learning.',
        'responsibilities': [
            'Develop and implement machine learning models',
            'Analyze large datasets to extract insights',
            'Present findings to stakeholders'
        ],
        'skills': ['Python', 'R', 'Machine Learning', 'Statistics']
    }
]

@app.route('/')
def job_list():
    return render_template('job_list.html', jobs=jobs)

@app.route('/job/<int:job_id>')
def job_detail(job_id):
    job = next((job for job in jobs if job['id'] == job_id), None)
    if job:
        return render_template('job_detail.html', job=job)
    return 'Job not found', 404

@app.route('/apply/<int:job_id>')
def apply_form(job_id):
    job = next((job for job in jobs if job['id'] == job_id), None)
    if job:
        return render_template('apply_form.html', job=job)
    return 'Job not found', 404

@app.route('/submit_application/<int:job_id>', methods=['POST'])
def submit_application(job_id):
    if request.method == 'POST':

        full_name = request.form['full_name']
        email = request.form['email']
        phone = request.form['phone']
        cover_letter = request.form['cover_letter']
        
        # Handle resume upload
        if 'resume' in request.files:
            resume = request.files['resume']
            if resume.filename != '':
                filename = secure_filename(resume.filename)
                resume.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Here you would typically save this information to a database
        # For now, we'll just print it and show a success message
        print(f"Application received for job {job_id}:")
        print(f"Name: {full_name}, Email: {email}, Phone: {phone}")
        print(f"Cover Letter: {cover_letter}")
        print(f"Resume filename: {filename}")
        
        flash('Your application has been submitted successfully!', 'success')
        return redirect(url_for('job_detail', job_id=job_id))

@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    if 'company_description' not in request.files or 'job_role' not in request.files:
        return jsonify({"error": "Please upload both PDFs"}), 400

    company_pdf = request.files['company_description']
    job_pdf = request.files['job_role']

    # Save the uploaded PDFs temporarily
    company_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(company_pdf.filename))
    job_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(job_pdf.filename))
    company_pdf.save(company_path)
    job_pdf.save(job_path)

    # Process the PDFs
    process_pdfs(company_path, job_path)

    return jsonify({"message": "PDFs processed successfully"}), 200

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_question = data['question']

    # Load the vector store with dangerous deserialization allowed
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return jsonify({"answer": response["output_text"]})

@app.route('/chat')
def chat():
    return render_template('chat.html')  # Chat page

if __name__ == '__main__':
    app.run(debug=True)
