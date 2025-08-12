from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import aiofiles
import uuid, mimetypes

app = FastAPI(title="Vocalist")

UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
templates = Jinja2Templates(directory="app/templates")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    allowed = {"audio/mpeg","audio/wav","audio/x-wav","audio/mp4",
               "audio/aac","audio/webm","audio/ogg","audio/x-m4a"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported type: {file.content_type}")
    ext = mimetypes.guess_extension(file.content_type) or Path(file.filename).suffix
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(1_048_576):
            await out.write(chunk)
    return templates.TemplateResponse("_upload_result.html",
                                      {"request": None, "filename": dest.name, "bytes": dest.stat().st_size})
