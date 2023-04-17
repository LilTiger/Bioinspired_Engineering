import PyPDF2

def extract_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


pdf_path = "science.aap8987.pdf"  # 用您的PDF文件路径替换
text = extract_text(pdf_path)
print(text)
