# import WHAM
# from WHAM import wham_api

# print(dir(WHAM))i
from fastapi import FastAPI, UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import os
import uuid

UPLOAD_DIR = "tmp"        # 你现在的设置
os.makedirs(UPLOAD_DIR, exist_ok=True)   # ← 加这一行，不存在就建

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/api/upload")
async def upload(file:UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    save_path = os.getcwd() +  f"/tmp/{file.filename}"
    async with aiofiles.open(save_path,"wb") as out:
        while chunk := await file.read(1024*1024):
            await out.write(chunk)
    return {"task_id":task_id,
            "filename":file.filename,
             "path":save_path}

@app.get("/api/ping")
async def ping():
    return {"ok":True}