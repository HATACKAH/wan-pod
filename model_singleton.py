import torch
from wan.text2video import WanT2V
from wan.configs import WAN_CONFIGS
import logging

logger = logging.getLogger(__name__)

class ModelSingleton:
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_model(cls):
        if cls._model is None:
            logger.info("Инициализация модели...")
            device_id = 0 if torch.cuda.is_available() else "cpu"
            config = WAN_CONFIGS["t2v-14B"]
            cls._model = WanT2V(
                config=config,
                checkpoint_dir="./Wan2.1-T2V-14B",
                device_id=device_id,
                rank=0
            )
            logger.info("Модель инициализирована")
        return cls._model