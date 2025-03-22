# from pix2tex.cli import LatexOCR
# from pdf2image import convert_from_path
#
# model = LatexOCR()  # Загрузка модели
# images = convert_from_path("Integral.pdf")
#
# latex_output = []
# for img in images:
#     # Распознаем формулы и текст
#     latex = model(img)
#     latex_output.append(latex)
#
# # Сохраняем результат
# with open("Integral.pdf", "w") as f:
#     f.write("\n".join(latex_output))

"""from pypdf import PdfReader
import re

reader = PdfReader('SPLINE.pdf')
pages = reader.pages
text = ""
for page in reader.pages:
    text += page.extract_text()


print(text)"""

# from Preprocessing import pdf2tex
# from transformers import AutoTokenizer, AutoModel
# import torch
#
# tex = pdf2tex("Integral.pdf")
# text=tex.textLatex
# print(text[0])

# model_name = "tbs17/MathBERT"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
#
# inputs = tokenizer(text[0:350], return_tensors="pt")
#
# # получение эмбеддингов
# with torch.no_grad():
#     outputs = model(**inputs)
#
# embeddings = outputs.last_hidden_state.mean(dim=1) # вектор текста
# print(embeddings)

from Preprocessing import pdf2tex
from transformers import AutoTokenizer, AutoModel
import torch

tex = pdf2tex("SPLINE.pdf")
text=tex.textLatex
#print(text)

model_name = "tbs17/MathBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

inputs = tokenizer(text[0:350], return_tensors="pt")

# получение эмбеддингов
with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state.mean(dim=1) # вектор текста
print(embeddings)