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
import json
import psycopg2
from RBERTTEST.ml.Training import train

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'

app = FastAPI()

app.mount( "/static",StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")# подключение цсс из папки
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))# шаблоны из Jinja

CURRENT_BOOK_TEXT = ""

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

@app.get("/files", response_class=HTMLResponse)
async def get_files(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "book_text": "", "res_text": ""}
    )
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
        global CURRENT_BOOK_TEXT
        CURRENT_BOOK_TEXT = book_text

        processor.analyze_document(book_text)

        with open(BASE_DIR / 'static' / 'data' / 'results.json', encoding='utf-8') as fh:
            res = json.load(fh)["final_predictions"]

        res_ = ""
        for r in res:
            res_ += r['code'] + ' ' + r['name'] + "<br>"

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "book_text": book_text, "res_text": res_}
        )
    except Exception as e:
        return {"status": "error", "message": f"There was an error uploading the file: {str(e)}"}
    finally:
        upload_file.file.close()


@app.post("/trainindexes")
async def train_model(request: Request, real_index: str = Form(...)):
    db_config = {
        'host': 'host.docker.internal',
        'database': 'BBK_index',
        'user': 'postgres',
        'password': 'Dima2003',
        'port': 5432
    }

    global CURRENT_BOOK_TEXT

    try:
        with psycopg2.connect(**db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT EXISTS (SELECT 1 FROM index_bbk WHERE path = %s);", (real_index,))
                exists = cursor.fetchone()[0]

                if not exists:
                    return {"status": "error", "message": f"Индекс ББК '{real_index}' не найден в базе данных"}

        import threading
        threading.Thread(
            target=train,
            kwargs={
                "mode": "incremental",
                "text": CURRENT_BOOK_TEXT,
                "bbk_id": real_index
            }
        ).start()

        return {"status": "success", "message": "Данные отправлены на дообучение модели"}

    except Exception as e:
        return {"status": "error", "message": f"Произошла ошибка: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("main:app", reload = True)