from models.xgboost_model import XGBoostModel
from models.cnn_model import CNNModel
from models.bigru_model import BiGRUModel
from models.rf_model import RFModel
from models.end2race_model import End2RaceModel

MODEL_REGISTRY = {
    "xgboost": XGBoostModel,
    "rf": RFModel,
    "cnn": CNNModel,
    "bigru": BiGRUModel,
    "end2race": End2RaceModel,
}
