from core import Workflow, group_node
from pyiron_nodes.atomistic.thermodynamics.defect_phases import (
    AddDefectConcentrationColumns,
    AddElementCountColumns,
    ComputeChemicalPotentials,
    ComputeDefectFormationEnergy,
    PlotDefectFormationEnergy,
)
from pyiron_nodes.controls import IterToDataFrame
from pyiron_nodes.dataframe import GetColumnFromDataFrame
from pyiron_nodes.math_utils import Linspace

# ── Group node factories ─────────────────────────────


@group_node("table")
def CreateDefectStructures(
    name="Al",
    cubic=True,
    repeat_scalar=3,
    substitution_element="Mg",
    interstitial_element="Mg",
):
    """
    Build a pristine host structure plus one defect of each type (vacancy,
    substitution, interstitial), and a chained defect (a second vacancy
    added on top of the interstitial). Mirrors the container_new example
    in point_defects_metals.py, packaged as a group node so it can feed
    into an energy/formation-energy pipeline like the one below.
    """
    from core import Workflow
    from pyiron_nodes.atomistic.structure.build import Bulk
    from pyiron_nodes.atomistic.structure.container_new import (
        AddPristine,
        CreateDefectFromIds,
        GetStructureTable,
        GetVoronoiInterstitialSites,
    )
    from pyiron_nodes.atomistic.structure.transform import Repeat

    inner_wf = Workflow("CreateDefectStructures")
    inner_wf.Bulk = Bulk(name=name, cubic=cubic)
    inner_wf.Repeat = Repeat(structure=inner_wf.Bulk, repeat_scalar=repeat_scalar)
    inner_wf.AddPristine = AddPristine(atoms=inner_wf.Repeat)

    inner_wf.GetVoronoiInterstitialSites = GetVoronoiInterstitialSites(
        atoms=inner_wf.Repeat
    )

    inner_wf.CreateVacancy = CreateDefectFromIds(
        structure_container=inner_wf.AddPristine,
        defect_type="vacancy",
        atom_ids=[0],
    )

    inner_wf.CreateSubstitution = CreateDefectFromIds(
        structure_container=inner_wf.CreateVacancy,
        defect_type="substitution",
        atom_ids=[5],
        to_element=substitution_element,
    )

    inner_wf.CreateInterstitial = CreateDefectFromIds(
        structure_container=inner_wf.CreateSubstitution,
        defect_type="interstitial",
        sublattice=inner_wf.GetVoronoiInterstitialSites.outputs.all_sites,
        site_ids=[0],
        element=interstitial_element,
    )

    # Chain a second defect onto the interstitial (generation 2).
    inner_wf.CreateVacancyNearInterstitial = CreateDefectFromIds(
        structure_container=inner_wf.CreateInterstitial,
        defect_type="vacancy",
        atom_ids=[20],
        parent_defect_index=-1,
    )

    inner_wf.GetStructureTable = GetStructureTable(
        structure_container=inner_wf.CreateVacancyNearInterstitial
    )
    return inner_wf.GetStructureTable


@group_node("energy")
def group_GRACE_StaticEnergy(structure):
    from core import Workflow
    from pyiron_nodes.atomistic.calculator.ase import StaticEnergy
    from pyiron_nodes.atomistic.engine.ase import GRACE

    inner_wf = Workflow("group_GRACE_StaticEnergy")
    inner_wf.GRACE = GRACE()
    inner_wf.StaticEnergy = StaticEnergy(structure=structure, engine=inner_wf.GRACE)
    return inner_wf.StaticEnergy


wf = Workflow("point_defect_metals_energies")

wf.CreateDefectStructures = CreateDefectStructures(
    name="Al",
    cubic=True,
    repeat_scalar=3,
    substitution_element="Mg",
    interstitial_element="Mg",
)

# Sweep of candidate Mg chemical potentials (eV) -- Mg never appears alone
# in the structure table above, so its potential must be supplied
# externally rather than read off a unary reference row.
wf.Linspace_1 = Linspace(x_min=-2.0, x_max=-1.0, num_points=5)

wf.group_GRACE_StaticEnergy = group_GRACE_StaticEnergy(structure="NotData")

wf.GetColumnFromDataFrame = GetColumnFromDataFrame(
    df=wf.CreateDefectStructures, column_name="structure"
)

wf.IterToDataFrame = IterToDataFrame(
    node=wf.group_GRACE_StaticEnergy,
    input_label="structure",
    values=wf.GetColumnFromDataFrame,
)
wf.IterToDataFrame.inputs.add(
    "debug", port_type=bool, default=False, value=True, has_explicit_default=True
)
wf.IterToDataFrame.inputs.add(
    "executor", port_type=object, default=None, value=None, has_explicit_default=True
)
wf.IterToDataFrame.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.AddElementCountColumns = AddElementCountColumns(df=wf.IterToDataFrame)

wf.AddDefectConcentrationColumns = AddDefectConcentrationColumns(
    df=wf.AddElementCountColumns
)

wf.ComputeChemicalPotentials = ComputeChemicalPotentials(
    df=wf.AddDefectConcentrationColumns,
    mu_reference=wf.Linspace_1,
    mu_reference_element="Mg",
)

wf.ComputeDefectFormationEnergy = ComputeDefectFormationEnergy(
    df=wf.AddDefectConcentrationColumns,
    chemical_potentials=wf.ComputeChemicalPotentials,
)

wf.PlotDefectFormationEnergy = PlotDefectFormationEnergy(
    formation_energies=wf.ComputeDefectFormationEnergy
)
