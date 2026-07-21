from transformers import Wav2Vec2Processor, Wav2Vec2Model
from app.core.config import MODEL_NAME
import torch


class ModelLoader:

    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.processor = None
        self.model = None
        self.loaded = False

    def load(self):

        if self.loaded:
            return

        print(f"Loading {MODEL_NAME}...")

        self.processor = Wav2Vec2Processor.from_pretrained(
            MODEL_NAME
        )

        self.model = Wav2Vec2Model.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        self.loaded = True

        print("Model Loaded Successfully")


model_loader = ModelLoader()