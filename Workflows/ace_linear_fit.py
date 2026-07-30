from core import Workflow, group_node
from pyiron_nodes.atomistic.ml_potentials.fitting.linearfit import (
    ParameterizePotentialConfig,
    PlotEnergyFittingCurve,
    PredictEnergiesAndForces,
    ReadPickledDatasetAsDataframe,
    RunLinearFit,
    SplitTrainingAndTesting,
)

wf = Workflow("ace_linear_fit")

wf.ParameterizePotentialConfig = ParameterizePotentialConfig()

wf.ReadPickledDatasetAsDataframe = ReadPickledDatasetAsDataframe(
    file_path="data/mgca.pckl.tgz"
)

wf.SplitTrainingAndTesting = SplitTrainingAndTesting(
    data_df=wf.ReadPickledDatasetAsDataframe
)

wf.RunLinearFit = RunLinearFit(
    potential_config=wf.ParameterizePotentialConfig,
    df_train=wf.SplitTrainingAndTesting.outputs.df_training,
    df_test=wf.SplitTrainingAndTesting.outputs.df_testing,
)
wf.RunLinearFit.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.PredictEnergiesAndForces = PredictEnergiesAndForces(
    basis=wf.RunLinearFit,
    df_train=wf.SplitTrainingAndTesting.outputs.df_training,
    df_test=wf.SplitTrainingAndTesting.outputs.df_testing,
)
wf.PredictEnergiesAndForces.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.PlotEnergyFittingCurve = PlotEnergyFittingCurve(
    data_dict=wf.PredictEnergiesAndForces
)
