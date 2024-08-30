from abc import ABC, abstractmethod


class CallbackI(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.ippo = None


class UpdateCallback(CallbackI):
    def __init__(self, ippo):
        self.ippo = ippo
        self.update_metrics = None

    @abstractmethod
    def after_update(self):
        pass

    @abstractmethod
    def before_update(self):
        pass