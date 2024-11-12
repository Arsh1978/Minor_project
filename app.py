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
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("google-api-key")
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # needed for flash messages
app.config['UPLOAD_FOLDER'] = 'uploads/'  # create this folder in your project directory

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def calculate_similarity_score(job_text, resume_text):
    # Using a lightweight Hugging Face model for similarity scoring
    model_name = 'distilbert-base-nli-stsb-mean-tokens'  # Fast and efficient model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    job_embedding = embeddings.embed_query(job_text)
    resume_embedding = embeddings.embed_query(resume_text)
    score = cosine_similarity([job_embedding], [resume_embedding])[0][0]
    return score

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
    company_text = get_pdf_text([company_path])
    job_text = get_pdf_text([job_path])
    all_texts = company_text + "\n" + job_text
    text_chunks = get_text_chunks(all_texts)
    get_vector_store(text_chunks)

# Mock job data
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
    },
    {
        'id': 3,
        'title': 'Generative AI Intern',
        'company': 'Extuent',
        'location': 'FL, USA',
        'description': 'We are seeking a Machine learning intern. This internship offers an exciting opportunity to work on cutting-edge AI and blockchain intersection projects.',
        'responsibilities': [
            'Assist in design and implementation cutting-edge machine learning models for complex tabular and time series data problems',
            'Collaborate on end-to-end ML projects, from problem definition to deployment and monitoring',
            ' Plan, execute, and successfully deploy a suite of cutting-edge AI-powered solutions, revolutionizing userexperience and operational efficiency: Elevator pitch summary generator, Resume parsing service, Profilesummary creator, Content moderation system',
            'Proven experience in developing and deploying ML models for tabular'
        ],
        'skills': [ 'Amazon Web Services, AWS SageMaker, Docker, Git, GitHub Actions, Python, SQL, TensorFlow, Keras, Pytorch']
    }
]

# Routes
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
    score = request.args.get('score', None)
    if score is not None:
        score = float(score)
    if job:
        return render_template('apply_form.html', job=job, score=score)
    return 'Job not found', 404

@app.route('/submit_application/<int:job_id>', methods=['POST'])
def submit_application(job_id):
    if request.method == 'POST':
        # Extract job description
        job = next((job for job in jobs if job['id'] == job_id), None)
        if not job:
            return "Job not found", 404

        job_text = job['description']

        # Get resume file and candidate's email
        if 'resume' in request.files and 'email' in request.form:
            resume = request.files['resume']
            candidate_email = request.form['email']

            if resume.filename != '':
                filename = secure_filename(resume.filename)
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                resume.save(resume_path)

                # Extract resume text
                resume_text = get_pdf_text([resume_path])

                # Calculate similarity score
                similarity_score = calculate_similarity_score(job_text, resume_text)

                # Send email based on similarity score
                sender_email = "namanhr823@gmail.com"
                receiver_email = candidate_email
                email_password = "--password--"

                if similarity_score >= 0.5:
                    subject = "Application Submitted - Interview Invitation"
                    body = f"Your application has been submitted successfully!\n\nAfter reviewing your resume, we found a strong alignment with our job description, with a similarity score of {round(similarity_score, 2)}. We are excited to invite you to the next stage in our hiring process!. Please join the Google Meet link at [https://meet.google.com/hej-trnt-rjd] on [insert_date_and_time].Please confirm your availability by replying to this email. We look forward to speaking with you and learning more about your background and qualifications."
                else:
                    subject = "Application Submitted - Rejection Notice"
                    body = f"Your application has been submitted successfully!\n\nAfter careful review, we found that your resume's similarity score with our job description was {round(similarity_score, 2)}, which is below the threshold required to advance to the next stage of our hiring process. Unfortunately, We encourage you to explore other opportunities with us in the future, and we wish you success in your job search."

                message = MIMEMultipart()
                message["From"] = sender_email
                message["To"] = receiver_email
                message["Subject"] = subject
                message.attach(MIMEText(body, "plain"))

                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login(sender_email, email_password)
                    server.sendmail(sender_email, receiver_email, message.as_string())

                # Flash message and redirect to job detail
                flash('Your application has been submitted successfully!', 'success')

                # Return a JSON response with the similarity score
                return jsonify({'Application Status': 'success', 'score': similarity_score})

        return "Invalid request", 400

@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    if 'company_description' not in request.files or 'job_role' not in request.files:
        return jsonify({"error": "Please upload both PDFs"}), 400

    company_pdf = request.files['company_description']
    job_pdf = request.files['job_role']

    company_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(company_pdf.filename))
    job_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(job_pdf.filename))
    company_pdf.save(company_path)
    job_pdf.save(job_path)

    process_pdfs(company_path, job_path)

    return jsonify({"message": "PDFs processed successfully"}), 200

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_question = data['question']
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
