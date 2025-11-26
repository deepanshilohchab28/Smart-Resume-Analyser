import re, sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Database setup ---
connection = sqlite3.connect('resume_data.db', check_same_thread=False)
cursor = connection.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    phone TEXT,
    filename TEXT,
    job_category TEXT,
    role TEXT,
    similarity REAL,
    matched_skills TEXT,
    skill_gaps TEXT
)""")
connection.commit()

def save_to_db(data):
    cursor.execute("""INSERT INTO candidates 
        (name, email, phone, filename, job_category, role, similarity, matched_skills, skill_gaps)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (data["name"], data["email"], data["phone"], data["filename"],
         data["job_category"], data["role"], data["similarity"],
         data["matched_skills"], data["skill_gaps"]))
    connection.commit()

def load_all_data():
    return pd.read_sql("SELECT * FROM candidates", connection)

def extract_text(file):
    import docx2txt, PyPDF2
    if file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        return "".join(page.extract_text() for page in reader.pages)
    elif file.name.endswith('.docx'):
        return docx2txt.process(file)
    else:
        return file.read().decode('utf-8', errors='ignore')

def extract_candidate_details(text):
    # Clean text to remove excessive newlines/spaces
    cleaned = " ".join(text.split())

    # NAME: Capture uppercase names or normal names
    name_match = re.search(r'\b([A-Z][A-Z]+(?:\s[A-Z][A-Z]+)+)\b', cleaned)
    if not name_match:
        # fallback for normal names
        name_match = re.search(r'\b[A-Z][a-zA-Z]+\s[A-Z][a-zA-Z]+\b', cleaned)

    # EMAIL
    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', cleaned)

    # PHONE
    phone_match = re.search(r'\+?\d[\d\s\-]{8,}\d', cleaned)

    # SKILLS
    skills = re.findall(r'\b[A-Za-z\#\+]{2,20}\b', text.lower())

    return {
        "Name": name_match.group(0) if name_match else "Unknown",
        "Email": email_match.group(0) if email_match else "Not Found",
        "Phone": phone_match.group(0) if phone_match else "Not Found",
        "Skills": list(set(skills))
    }

def extract_skills_from_jd(jd_text):
    return [s.lower().strip() for s in re.findall(r'\b[A-Za-z\#\+]{2,15}\b', jd_text)]

def detect_skill_gaps(jd_skills, resume_skills):
    matched = [s for s in jd_skills if s in resume_skills]
    gaps = [s for s in jd_skills if s not in resume_skills]
    return matched, gaps

JOB_CATEGORIES = {
    "Software Development": ["Frontend Developer", "Backend Developer", "Full Stack Developer"],
    "Data & Analytics": ["Data Scientist", "Data Analyst"],
    "Cloud & DevOps": ["DevOps Engineer", "Cloud Engineer"]
}

ROLE_SKILLS = {
    "Frontend Developer": "HTML CSS JavaScript React UI UX",
    "Backend Developer": "Python Django Flask SQL APIs",
    "Full Stack Developer": "HTML CSS JS Node React Python SQL",
    "Data Scientist": "Python SQL Machine Learning Statistics Pandas Numpy",
    "Data Analyst": "Python SQL Excel Tableau PowerBI Pandas Numpy",
    "DevOps Engineer": "AWS Docker Kubernetes Linux CI/CD",
    "Cloud Engineer": "AWS Azure GCP Terraform DevOps"
}


