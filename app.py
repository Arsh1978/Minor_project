from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # needed for flash messages
app.config['UPLOAD_FOLDER'] = 'uploads/'  # create this folder in your project directory

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Sample job data (replace this with a database in a real application)
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

if __name__ == '__main__':
    app.run(debug=True)