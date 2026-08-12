
from pypdf import PdfReader
from docx import Document

# 1. Individual PDF Extractor
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    master_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            master_text += page_text + "\n"
    return master_text 

# 2. Individual Plain Text Extractor
def extract_text_from_txt(uploaded_file):
    master_text = uploaded_file.read().decode("utf-8")
    return master_text

# 3. Individual Word Document Extractor
def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    master_text = ""
    for paragraph in doc.paragraphs:
        master_text += paragraph.text + "\n"
    return master_text

# 4. The Master Router Function (The only one app.py needs to talk to)
def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name
    
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif filename.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        return "Unsupported file format."
