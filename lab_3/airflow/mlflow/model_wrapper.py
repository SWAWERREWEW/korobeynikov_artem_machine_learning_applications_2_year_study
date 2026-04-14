# model_wrapper.py
import numpy as np
import mlflow.pyfunc

class PricePredictor(mlflow.pyfunc.PythonModel):
    """
    Оборачивает обученный SGDRegressor и PowerTransformer,
    чтобы inverse_transform применялся автоматически при predict.
    Наследуется от PythonModel, что требуется для mlflow.pyfunc.
    """
    def __init__(self, model, power_transformer):
        self.model = model
        self.power_transformer = power_transformer

    def predict(self, context, model_input):
        """
        context: игнорируется (нужен для совместимости с MLflow)
        model_input: входные данные (numpy array или DataFrame)
        """
        y_scaled = self.model.predict(model_input)
        y_original = self.power_transformer.inverse_transform(
            y_scaled.reshape(-1, 1)
        )
        return y_original.flatten()
