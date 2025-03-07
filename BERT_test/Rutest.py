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