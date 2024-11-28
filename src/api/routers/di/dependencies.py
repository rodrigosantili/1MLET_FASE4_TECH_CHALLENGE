from ...di import Injector
from ...routers.interfaces import PredictionService, HistoricalDataService
from ...services import PredictionServiceImpl, HistoricalDataServiceImpl

injector = Injector()
injector.register(HistoricalDataService, HistoricalDataServiceImpl())
injector.register(PredictionService, PredictionServiceImpl())
