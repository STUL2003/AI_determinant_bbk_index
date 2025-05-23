
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pdfplumber
from pathlib import Path
import RBERTTEST.ml.rbert as rb
import tempfile
from fastapi.responses import PlainTextResponse
import os

UPLOAD_DIR = Path() / 'uploads'

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")# подключение цсс из папки
templates = Jinja2Templates(directory="templates")# шаблоны из Jinja


def extract_text( pdf_path, start_page=0):
    """Извлечение текста из PDF с обработкой ошибок"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_pages = []
            for i, page in enumerate(pdf.pages[start_page:], start=start_page):
                text = page.extract_text()
                if text:
                    text_pages.append(text)
            result = " ".join(text_pages) if text_pages else ""
            return result
    except Exception as e:
        return ""


@app.post("/extract-text")
async def extract_text_from_pdf(file: UploadFile):
    # delete=False – файл не будет автоматически удаляться после закрытия
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    text = extract_text(tmp_path)
    os.unlink(tmp_path)

    return PlainTextResponse(text)


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})# рендерим html шаблон и передаем объект request


@app.post("/files")
async def classificator(request: Request, upload_file: UploadFile):
    try:
        data = await upload_file.read()
        save_to = UPLOAD_DIR / upload_file.filename
        with open(save_to, 'wb') as f:
            f.write(data)
        processor = rb.DocumentProcessor()
        book_text = extract_text(save_to)
        processor.analyze_document(book_text)
        print("Всё")
        with open("res.txt", "r",  encoding='utf-8') as f:
            res = f.readlines()
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "book_text": book_text, "res_text": res}
        )
    except Exception as e:
        return {"message": f"There was an error uploading the file: {str(e)}"}
    finally:
        upload_file.file.close()
if __name__ == "__main__":
    uvicorn.run("main:app", reload = True)