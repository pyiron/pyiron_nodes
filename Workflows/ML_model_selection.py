from pyiron_nodes.dataframe import ReadDataFrame
from pyiron_nodes.ml_nodes import ChooseBestModel, MLDataSplitter
from pyiron_nodes.sklearn_nodes import (
    EvaluateRegressionModelSklearn_model,
    LinearRegression_model,
    PredictRegression_model,
    SupportVectorRegression_model,
)
from core import Workflow
from core import group_node

wf = Workflow("simple_inference")

wf.ReadDataFrame = ReadDataFrame(filename="sim_equilibrium_pot.csv", file_format="csv")

wf.MLDataSplitter = MLDataSplitter(df=wf.ReadDataFrame, y_name="volume")

wf.LinearRegression_model = LinearRegression_model(
    X_train=wf.MLDataSplitter.outputs.X_train, y_train=wf.MLDataSplitter.outputs.y_train
)

wf.SupportVectorRegression_model = SupportVectorRegression_model(
    X_train=wf.MLDataSplitter.outputs.X_train, y_train=wf.MLDataSplitter.outputs.y_train
)

wf.EvaluateRegressionModelSklearn_model = EvaluateRegressionModelSklearn_model(
    model=wf.LinearRegression_model,
    X_test=wf.MLDataSplitter.outputs.X_test,
    y_test=wf.MLDataSplitter.outputs.y_test,
)

wf.EvaluateRegressionModelSklearn_model_1 = EvaluateRegressionModelSklearn_model(
    model=wf.SupportVectorRegression_model,
    X_test=wf.MLDataSplitter.outputs.X_test,
    y_test=wf.MLDataSplitter.outputs.y_test,
)

wf.ChooseBestModel = ChooseBestModel(
    model_1=wf.LinearRegression_model,
    model_2=wf.SupportVectorRegression_model,
    X_validation=wf.MLDataSplitter.outputs.X_validation,
    y_validation=wf.MLDataSplitter.outputs.y_validation,
)

wf.PredictRegression_model = PredictRegression_model(
    model=wf.ChooseBestModel.outputs.best_model
)
