from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
from wan.text2video import WanT2V
import torch
import os
from typing import Optional, Dict
import uuid
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
import logging
import time
from torchvision.utils import save_image
from wan.utils.utils import cache_video
from concurrent.futures import ThreadPoolExecutor, Future
from model_singleton import ModelSingleton
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI()

# Добавляем CORS middleware сразу после создания приложения
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем запросы с любых источников
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все HTTP методы
    allow_headers=["*"],  # Разрешаем все заголовки
)

tasks_store: Dict[str, dict] = {}
futures_store: Dict[str, Future] = {}

text2video_model = ModelSingleton.get_model()

class Text2VideoRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 832
    height: Optional[int] = 480
    num_frames: Optional[int] = 16
    fps: Optional[int] = 8

video_generator = ThreadPoolExecutor(max_workers=1)

def generate_video(task_id: str, params: dict):
    start_time = time.time()
    try:
        logger.info(f"Начало генерации видео для task_id: {task_id}")
        logger.debug(f"Параметры запроса: {params}")

        size = SIZE_CONFIGS["832*480"]
        logger.info(f"Установлен размер видео: {size}")

        generation_start = time.time()
        logger.info("Запуск генерации видео...")
        video = text2video_model.generate(
            input_prompt=params["prompt"],
            size=size,
            sampling_steps=params["num_inference_steps"],
            guide_scale=params["guidance_scale"],
            n_prompt=params["negative_prompt"],
            frame_num=params["num_frames"]
        )
        generation_time = time.time() - generation_start
        logger.info(f"Генерация видео завершена за {generation_time:.2f} секунд")

        save_start = time.time()
        output_path = os.path.join(RESULTS_DIR, f"output_{task_id}.mp4")
        logger.info(f"Сохранение видео в {output_path}")

        cache_video(
            tensor=video[None],
            save_file=output_path,
            fps=params["fps"],
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

        save_time = time.time() - save_start
        logger.info(f"Видео успешно сохранено за {save_time:.2f} секунд")

        total_time = time.time() - start_time
        tasks_store[task_id] = {
            "status": "success",
            "output": {
                "video_path": output_path,
                "task_type": "text2video",
                "prompt": params["prompt"],
                "generation_time": generation_time,
                "save_time": save_time,
                "total_time": total_time
            }
        }
        logger.info(f"Задача {task_id} успешно завершена. Общее время выполнения: {total_time:.2f} секунд")
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Ошибка при генерации видео после {total_time:.2f} секунд работы: {str(e)}"
        logger.error(error_msg, exc_info=True)
        tasks_store[task_id] = {
            "status": "error",
            "error": str(e),
            "execution_time": total_time
        }

@app.post("/text2video/create")
async def create_text2video_task(request: Text2VideoRequest):
    from fastapi.responses import JSONResponse

    logger.info("Получен новый запрос на создание видео")
    task_id = str(uuid.uuid4())
    params = request.model_dump()

    tasks_store[task_id] = {"status": "processing"}

    future = video_generator.submit(generate_video, task_id, params)
    futures_store[task_id] = future

    return JSONResponse(
        content={
            "task_id": task_id,
            "status": "processing"
        }
    )

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    logger.info(f"Запрос статуса для task_id: {task_id}")

    if task_id not in tasks_store:
        logger.warning(f"Задача {task_id} не найдена")
        raise HTTPException(status_code=404, detail="Task not found")

    # Добавляем информацию о статусе потока
    response = tasks_store[task_id].copy()
    if task_id in futures_store:
        future = futures_store[task_id]
        response["thread_status"] = {
            "running": future.running(),
            "done": future.done(),
            "cancelled": future.cancelled()
        }

    logger.info(f"Возвращаем статус для task_id {task_id}: {response}")
    return response

@app.get("/tasks")
async def get_all_tasks():
    active_tasks = {}
    for task_id in tasks_store:
        task_info = tasks_store[task_id].copy()
        if task_id in futures_store:
            future = futures_store[task_id]
            task_info["thread_status"] = {
                "running": future.running(),
                "done": future.done(),
                "cancelled": future.cancelled()
            }
        active_tasks[task_id] = task_info
    return active_tasks

if __name__ == "__main__":
    import uvicorn
    logger.info("Запуск FastAPI сервера...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        access_log=True
    )