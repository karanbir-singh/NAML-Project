from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
        An abstract base class for all the models we develop.
    """
    
    #Up to now, this method only pass: implment it in the child class
    def __init__(self):
        pass

    #Up to now, this method only pass: implment it in the child class
    @abstractmethod
    def train(self):
        pass

    #Up to now, this method only pass: implment it in the child class
    @abstractmethod
    def predict(self):
        pass