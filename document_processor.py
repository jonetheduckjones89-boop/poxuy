from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    # Placeholder for DOCX support
    return "DOCX extraction not implemented yet."
