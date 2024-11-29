# models/base_model.py

from abc import ABC, abstractmethod

class BaseModelRegister(ABC):
    def __init__(self, config, helper):
        self.config = config
        self.helper = helper

    @abstractmethod
    def register_model(self):
        pass