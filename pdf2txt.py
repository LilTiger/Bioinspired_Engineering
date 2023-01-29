import pdfplumber
import docx
import glob

path = 'pdf_folder'  # 存储PDF的文件夹 根据此程序的相对路径来更改path
doc = docx.Document()
list = glob.glob(path + '/*.pdf')

for a in list:
    with pdfplumber.open(a) as pdf:
        for page in pdf.pages:
            print(page.extract_text())
            text = page.extract_text()

            with open(a + '.txt', 'a', encoding='utf8') as f:
                f.write(text)  # 提取pdf中的文字后写入到同名txt文件中


