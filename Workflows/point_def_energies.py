from pyiron_nodes.atomistic.thermodynamics.defect_phases import AddDefectConcentrationColumns, AddElementCountColumns, ComputeChemicalPotentials, ComputeDefectFormationEnergy, PlotDefectFormationEnergy
from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.dataframe import GetColumnFromDataFrame
from pyiron_nodes.math_utils import Linspace
from core import Workflow
from core import group_node

# ── Group node factories ─────────────────────────────

@group_node("table")
def CreateStructures(name, element, cubic=False, repeat_scalar=1):
    from pyiron_nodes.atomistic.structure.build import Bulk
    from pyiron_nodes.atomistic.structure.container import AddPristine, CreateSubstitutional, CreateVacancy, GetStructureTable
    from pyiron_nodes.atomistic.structure.transform import Repeat
    from core import Workflow
    
    inner_wf = Workflow("CreateStructures")
    inner_wf.Bulk = Bulk(name=name, cubic=cubic)
    inner_wf.Repeat = Repeat(structure=inner_wf.Bulk, repeat_scalar=repeat_scalar)
    inner_wf.AddPristine = AddPristine(pristine_structure=inner_wf.Repeat)
    inner_wf.CreateVacancy = CreateVacancy(structure_container=inner_wf.AddPristine)
    inner_wf.CreateVacancy_1 = CreateVacancy(structure_container=inner_wf.CreateVacancy, parent_defect_index=-1)
    inner_wf.CreateSubstitutional = CreateSubstitutional(element=element, structure_container=inner_wf.CreateVacancy_1)
    inner_wf.GetStructureTable = GetStructureTable(structure_container=inner_wf.CreateSubstitutional)
    return inner_wf.GetStructureTable

@group_node("energy")
def group_GRACE_StaticEnergy(structure):
    from pyiron_nodes.atomistic.calculator.ase import StaticEnergy
    from pyiron_nodes.atomistic.engine.ase import GRACE
    from core import Workflow
    
    inner_wf = Workflow("group_GRACE_StaticEnergy")
    inner_wf.GRACE = GRACE()
    inner_wf.StaticEnergy = StaticEnergy(structure=structure, engine=inner_wf.GRACE)
    return inner_wf.StaticEnergy

wf = Workflow("point_def_energies")

wf.CreateStructures = CreateStructures(name='Al', cubic=True, repeat_scalar=3, element='Ni')

wf.Linspace_1 = Linspace(x_min=3, x_max=5, num_points=5)

wf.group_GRACE_StaticEnergy = group_GRACE_StaticEnergy(structure='NotData')

wf.GetColumnFromDataFrame = GetColumnFromDataFrame(df=wf.CreateStructures, column_name='structure')

wf.IterToDataFrame = IterToDataFrame(node=wf.group_GRACE_StaticEnergy, input_label='structure', values=wf.GetColumnFromDataFrame)
wf.IterToDataFrame.inputs.add("debug", port_type=bool, default=False, value=True, has_explicit_default=True)
wf.IterToDataFrame.inputs.add("executor", port_type=object, default=None, value=None, has_explicit_default=True)
wf.IterToDataFrame.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)

wf.AddElementCountColumns = AddElementCountColumns(df=wf.IterToDataFrame)

wf.AddDefectConcentrationColumns = AddDefectConcentrationColumns(df=wf.AddElementCountColumns)

wf.ComputeChemicalPotentials = ComputeChemicalPotentials(df=wf.AddDefectConcentrationColumns, mu_reference=wf.Linspace_1, mu_reference_element='Ni')

wf.ComputeDefectFormationEnergy = ComputeDefectFormationEnergy(df=wf.AddDefectConcentrationColumns, chemical_potentials=wf.ComputeChemicalPotentials)

wf.PlotDefectFormationEnergy = PlotDefectFormationEnergy(formation_energies=wf.ComputeDefectFormationEnergy)
