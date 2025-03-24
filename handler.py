from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
from wan.text2video import Text2Video
from wan.image2video import Image2Video
import torch
import os
from typing import Optional
from celery import Celery
import uuid

app = FastAPI()

# Настройка Celery
celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# Инициализация моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text2video_model = Text2Video(device=device)
image2video_model = Image2Video(device=device)

# Модели данных
class Text2VideoRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 576
    height: Optional[int] = 576
    num_frames: Optional[int] = 16
    fps: Optional[int] = 8

class Image2VideoRequest(BaseModel):
    image: str  # base64 encoded image
    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: Optional[int] = 40
    guidance_scale: Optional[float] = 7.5
    num_frames: Optional[int] = 16
    fps: Optional[int] = 8

# Celery tasks
@celery_app.task
def generate_text2video(task_id: str, params: dict):
    try:
        video = text2video_model.generate(**params)
        output_path = f"/tmp/output_{task_id}.mp4"
        video.save(output_path)
        return {
            "status": "success",
            "output": {
                "video_path": output_path,
                "task_type": "text2video",
                "prompt": params["prompt"]
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@celery_app.task
def generate_image2video(task_id: str, params: dict):
    try:
        video = image2video_model.generate(**params)
        output_path = f"/tmp/output_{task_id}.mp4"
        video.save(output_path)
        return {
            "status": "success",
            "output": {
                "video_path": output_path,
                "task_type": "image2video",
                "prompt": params["prompt"]
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# FastAPI endpoints
@app.post("/text2video/create")
async def create_text2video_task(request: Text2VideoRequest):
    task_id = str(uuid.uuid4())
    params = request.dict()

    # Запуск задачи асинхронно
    task = generate_text2video.delay(task_id, params)

    return {
        "task_id": task_id,
        "status": "processing"
    }

@app.post("/image2video/create")
async def create_image2video_task(request: Image2VideoRequest):
    task_id = str(uuid.uuid4())
    params = request.dict()

    # Запуск задачи асинхронно
    task = generate_image2video.delay(task_id, params)

    return {
        "task_id": task_id,
        "status": "processing"
    }

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    # Проверяем статус задачи
    task = celery_app.AsyncResult(task_id)

    if task.ready():
        result = task.get()
        return result
    else:
        return {
            "task_id": task_id,
            "status": "processing"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)