from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        self._loaded = False
        self._metadata = {
            "name": "Unknown",
            "description": "No description.",
            "input_spec": "Unknown",
            "version": 0
        }

    def get_metadata(self):
        return self._metadata

    @abstractmethod
    def load(self):
        pass

    def ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Model must be loaded before inference.")

    @abstractmethod
    def convert(self, image_path):
        pass