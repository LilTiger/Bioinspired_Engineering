# import pdfplumber
# import docx
# import glob
#
# path = 'pdf_folder'  # 存储PDF的文件夹 根据此程序的相对路径来更改path
# doc = docx.Document()
# list = glob.glob(path + '/*.pdf')
#
# for a in list:
#     with pdfplumber.open(a) as pdf:
#         for page in pdf.pages:
#             print(page.extract_text())
#             text = page.extract_text()
#
#             with open(a + '.txt', 'a', encoding='utf-8-sig') as f:
#                 f.write(text)  # 提取pdf中的文字后写入到同名txt文件中

import fitz
from tqdm import tqdm
from bs4 import BeautifulSoup
import re

#将pdf转成html
def pdf2html(input_path,html_path):
    doc = fitz.open(input_path)
    print(doc)
    html_content =''
    for page in tqdm(doc):
        html_content += page.get_text('html')
    print('开始输出html文件')
    html_content +="</body></html>"
    with open(html_path, 'w', encoding = 'utf-8', newline='')as fp:
        fp.write(html_content)

#使用Beautifulsoup解析本地html
def html2txt(html_path):
    html_file = open(html_path, 'r', encoding = 'utf-8')
    htmlhandle = html_file.read()
    soup = BeautifulSoup(htmlhandle, "html.parser")
    for div in soup.find_all('div'):
        for p in div:
            text = str()
            for span in p:
                p_info = '<span .*?>(.*?)</span>'   #提取规则
                res = re.findall(p_info,str(span))  #findall函数
                if len(res) ==0:
                    pass
                else:
                    text+= res[0]  #将列表中的字符串内容合并加到行字符串中
            print(text)
            with open("test.txt",'a',encoding = 'utf-8')as text_file:
                text_file.write(text)
                text_file.write('\n')


#主函数
input_path = r'pdf_folder/1.pdf'
html_path = r'pdf_folder/1.html'
pdf2html(input_path,html_path )  #pdf转html
html2txt(html_path)
