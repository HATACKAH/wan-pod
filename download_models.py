import os
import sys
from huggingface_hub import snapshot_download

def download_models():
    # Проверяем наличие токена при сборке образа
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ОШИБКА: Не найден HF_TOKEN. Укажите его при сборке образа: --build-arg HF_TOKEN=xxx")
        sys.exit(1)

    models = {
        "Wan2.1-T2V-14B": "Wan-AI/Wan2.1-T2V-14B",
        "Wan2.1-I2V-14B-720P": "Wan-AI/Wan2.1-I2V-14B-720P",
    }

    # Используем переменную окружения для пути к моделям
    MODELS_PATH = os.getenv('MODELS_PATH', '/models')  # fallback to /models if not set
    base_path = MODELS_PATH
    os.makedirs(base_path, exist_ok=True)

    for model_name, repo_id in models.items():
        model_path = os.path.join(base_path, model_name)

        print(f"Загрузка модели {model_name}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                token=hf_token,
                resume_download=True
            )
            print(f"Модель {model_name} успешно загружена")
        except Exception as e:
            print(f"Ошибка при загрузке модели {model_name}: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    download_models()