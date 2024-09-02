from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def apply(self,netwotk,sources):
        pass

class LinearStrategy(Strategy):
    def apply(self,network,sources):
        print("solution for linear component")
