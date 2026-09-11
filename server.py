# import WHAM
# from WHAM import wham_api

# print(dir(WHAM))i
from fastapi import FastAPI, UploadFile,File
import aiofiles


app = FastAPI()

@app.post("/api/upload")
async def upload(file:UploadFile = File(...)):
    save_path = f"/tmp/{file.filename}"
    async with aiofiles.open(save_path,"wb") as out:
        while chunk := await file.read(1024*1024):
            await out.write(chunk)
    return {"filename":file.filename, "path":save_path}
