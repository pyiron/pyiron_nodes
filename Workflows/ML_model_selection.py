from core import Workflow
from pyiron_nodes.dataframe import ReadDataFrame
from pyiron_nodes.machine_learning.models import (
    EvaluateRegressionModelSklearn,
    LinearRegressionModel,
    PredictRegressionModel,
    SupportVectorRegressionModel,
)
from pyiron_nodes.machine_learning.pipeline import ChooseBestModel, MLDataSplitter

wf = Workflow("simple_inference")

wf.ReadDataFrame = ReadDataFrame(
    filename="share/sim_equilibrium_pot.csv", file_format="csv"
)

wf.MLDataSplitter = MLDataSplitter(df=wf.ReadDataFrame, y_name="volume")

wf.LinearRegressionModel = LinearRegressionModel(
    X_train=wf.MLDataSplitter.outputs.X_train, y_train=wf.MLDataSplitter.outputs.y_train
)

wf.SupportVectorRegressionModel = SupportVectorRegressionModel(
    X_train=wf.MLDataSplitter.outputs.X_train, y_train=wf.MLDataSplitter.outputs.y_train
)

wf.EvaluateRegressionModelSklearn = EvaluateRegressionModelSklearn(
    model=wf.LinearRegressionModel,
    X_test=wf.MLDataSplitter.outputs.X_test,
    y_test=wf.MLDataSplitter.outputs.y_test,
)

wf.EvaluateRegressionModelSklearn_1 = EvaluateRegressionModelSklearn(
    model=wf.SupportVectorRegressionModel,
    X_test=wf.MLDataSplitter.outputs.X_test,
    y_test=wf.MLDataSplitter.outputs.y_test,
)

wf.ChooseBestModel = ChooseBestModel(
    model_1=wf.LinearRegressionModel,
    model_2=wf.SupportVectorRegressionModel,
    X_validation=wf.MLDataSplitter.outputs.X_validation,
    y_validation=wf.MLDataSplitter.outputs.y_validation,
)

wf.PredictRegressionModel = PredictRegressionModel(
    model=wf.ChooseBestModel.outputs.best_model,
    X=wf.MLDataSplitter.outputs.X_test,
)
