import pytesseract
from PIL import Image
import pandas as pd
from PyPDF2 import PdfReader

# OCR for images
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# PDF loader
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# CSV loader
def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    return df.to_string(index=False)
