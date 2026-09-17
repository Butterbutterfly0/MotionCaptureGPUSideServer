# import WHAM
# from WHAM import wham_api

# print(dir(WHAM))i
from fastapi import FastAPI, UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse
import aiofiles
import os
import uuid
import shutil
from wham_service import run
from anime_gen import anime_video_gen

UPLOAD_DIR = "tmp/uploads"
OUTPUT_DIR = "tmp/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173","http://123.56.101.120:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

tasks: dict[str, dict] = {}
anime_tasks: dict[str,dict] = {}


def fake_process(task_id: str, video_path: str):
    tasks[task_id]["status"] = "running"
    total = 5
    for i in range(total):
        import time
        time.sleep(0.1)              # 假装处理 1 秒
        tasks[task_id]["progress"] = int((i + 1) / total * 100)

    output_path = os.path.join(OUTPUT_DIR,f'{task_id}')
    run(video_path,output_dir=output_path,visualize=True)
    shutil.move(os.path.join(output_path,'output.mp4'),output_path+'.mp4')

    tasks[task_id]["status"] = "done"
    tasks[task_id]["result"] = output_path+'.mp4'
    tasks[task_id]["progress"] = 100
    
    anime_tasks[task_id] = {"status":"pending", "progress":0, "result":None}
    anime_tasks[task_id]["status"] = "running"
    anime_video_gen(output_path, output_path+'.mp4',task_id)
    anime_tasks[task_id]["status"] = "done"
    anime_tasks[task_id]["result"] = output_path+'_anime.mp4'
    anime_tasks[task_id]["progress"] = 100
    

@app.post("/api/upload")
async def upload(file:UploadFile = File(...), bg: BackgroundTasks = None):
    task_id = str(uuid.uuid4())
    save_path = os.getcwd() +  f"/tmp/{task_id}.mp4"
    async with aiofiles.open(save_path,"wb") as out:
        while chunk := await file.read(1024*1024):
            await out.write(chunk)
    tasks[task_id] = {"status": "pending", "progress": 0, "result": None}
    bg.add_task(fake_process, task_id, save_path)
    return {"task_id":task_id,
            "filename":file.filename,
             "path":save_path}

@app.get("/api/ping")
async def ping():
    return {"ok":True}

@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task Not Found")
    t = tasks[task_id]
    return {
        "status":t["status"],
        "progress":t["progress"],
        "result":t["result"]
    }

@app.get("/api/anime_task/{task_id}")
async def get_task(task_id: str):
    if task_id not in anime_tasks:
        raise HTTPException(404, "Task Not Found")
    t = anime_tasks[task_id]
    return {
        "status":t["status"],
        "progress":t["progress"],
        "result":t["result"]
    }


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task Not Found")
    t = tasks[task_id]
    if t["status"] != "done" or not t["result"]:
        raise HTTPException(400, "Result not ready")
    return FileResponse(t["result"], media_type="video/mp4")

@app.get("/api/anime_result/{task_id}")
async def get_result(task_id: str):
    if task_id not in anime_tasks:
        raise HTTPException(404, "Task Not Found")
    t = anime_tasks[task_id]
    if t["status"] != "done" or not t["result"]:
        raise HTTPException(400, "Result not ready")
    return FileResponse(t["result"], media_type="video/mp4")