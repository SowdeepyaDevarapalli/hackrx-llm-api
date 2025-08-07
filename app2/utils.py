import PyPDF2
import docx2txt

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path: str) -> str:
    return docx2txt.process(file_path)
import spacy
nlp = spacy.load("en_core_web_sm")

def chunk_text(text: str, max_chunk_tokens=500):
    """
    Splits text into smaller chunks (~max_chunk_tokens) on sentence boundaries
    """
    doc = nlp(text)
    chunks = []
    chunk = ""
    for sent in doc.sents:
        if len(chunk.split()) + len(sent.text.split()) > max_chunk_tokens:
            chunks.append(chunk.strip())
            chunk = sent.text
        else:
            chunk += " " + sent.text
    if chunk:
        chunks.append(chunk.strip())
    return chunks
