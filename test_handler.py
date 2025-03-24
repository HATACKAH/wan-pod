from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
import uuid
import time
from typing import Optional

app = FastAPI()

# Имитация хранилища задач
tasks_store = {}

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
    image: str
    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: Optional[int] = 40
    guidance_scale: Optional[float] = 7.5
    num_frames: Optional[int] = 16
    fps: Optional[int] = 8

async def mock_video_generation(task_id: str, params: dict, task_type: str):
    # Имитация длительной генерации
    await asyncio.sleep(30)  # Имитация 30 секунд генерации

    tasks_store[task_id] = {
        "status": "success",
        "output": {
            "video_path": f"/tmp/mock_output_{task_id}.mp4",
            "task_type": task_type,
            "prompt": params["prompt"]
        }
    }

@app.post("/text2video/create")
async def create_text2video_task(request: Text2VideoRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    params = request.dict()

    # Сохраняем начальный статус
    tasks_store[task_id] = {"status": "processing"}

    # Запускаем "генерацию" в фоне
    background_tasks.add_task(mock_video_generation, task_id, params, "text2video")

    return {
        "task_id": task_id,
        "status": "processing"
    }

@app.post("/image2video/create")
async def create_image2video_task(request: Image2VideoRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    params = request.dict()

    tasks_store[task_id] = {"status": "processing"}
    background_tasks.add_task(mock_video_generation, task_id, params, "image2video")

    return {
        "task_id": task_id,
        "status": "processing"
    }

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_store:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks_store[task_id]

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")  # Добавим отладочный вывод
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"  # Добавим логирование
    )