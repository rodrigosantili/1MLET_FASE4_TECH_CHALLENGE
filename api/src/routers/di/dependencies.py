from di import Injector
from routers.interfaces import PredictionService
from services import PredictionServiceImpl

injector = Injector()
injector.register(PredictionService, PredictionServiceImpl())
