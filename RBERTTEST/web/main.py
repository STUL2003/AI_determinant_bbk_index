from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import RBERTTEST.ml.rbert

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")# подключение цсс из папки
templates = Jinja2Templates(directory="templates")# шаблоны из Jinja

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})# рендерим html шаблон и передаем объект request



@app.post("/files")# загрузка файла в текущую дирректорию
async def upload_books(upload_file: UploadFile):
    file = upload_file.file
    filename = upload_file.filename
    with open(filename, "wb") as f:
        f.write(file.read())

if __name__ == "__main__":
    uvicorn.run("main:app", reload = True)