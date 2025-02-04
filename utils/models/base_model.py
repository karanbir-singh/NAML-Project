from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
        An abstract base class for all the models we develop.
    """
    
    #Up to now, this method only pass: implement it in the child class
    def __init__(self):
        pass

    #Up to now, this method only pass: implement it in the child class
    @abstractmethod
    def fit(self, X = None, y = None):
        pass

    #Up to now, this method only pass: implement it in the child class
    @abstractmethod
    def predict(self, X = None):
        pass