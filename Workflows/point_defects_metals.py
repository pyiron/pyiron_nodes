from core import Workflow
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.container_new import (
    AddPristine,
    CreateDefectFromIds,
    GetDefectTable,
    GetStructureTable,
    GetVoronoiInterstitialSites,
)
from pyiron_nodes.atomistic.structure.transform import Repeat

wf = Workflow("point_defects_metals")

# Build a pristine host structure.
wf.Bulk = Bulk(name="Al", cubic=True)
wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar=3)
wf.AddPristine = AddPristine(atoms=wf.Repeat)

# Candidate interstitial sites, found once from the pristine supercell.
wf.GetVoronoiInterstitialSites = GetVoronoiInterstitialSites(atoms=wf.Repeat)

# --- One independent defect of each type ---
# Each call below leaves parent_defect_index at its default (None), which
# resolves to the latest pristine structure -- so these three are
# independent, single-defect structures (generation 1), not a chain.
wf.CreateVacancy = CreateDefectFromIds(
    structure_container=wf.AddPristine,
    defect_type="vacancy",
    atom_ids=[0],
)

wf.CreateSubstitution = CreateDefectFromIds(
    structure_container=wf.CreateVacancy,
    defect_type="substitution",
    atom_ids=[5],
    to_element="Mg",
)

wf.CreateInterstitial = CreateDefectFromIds(
    structure_container=wf.CreateSubstitution,
    defect_type="interstitial",
    sublattice=wf.GetVoronoiInterstitialSites.outputs.all_sites,
    site_ids=[0],
    element="Mg",
)

# --- Chaining: build a second defect on top of an existing one ---
# parent_defect_index=-1 targets "the most recently created defect in
# structure_container" instead of falling back to pristine. At this point
# in the chain that is the interstitial created above, so this adds a
# vacancy directly on top of it -- a two-defect structure (generation 2),
# e.g. for studying interstitial-vacancy interaction as a function of
# separation distance.
wf.CreateVacancyNearInterstitial = CreateDefectFromIds(
    structure_container=wf.CreateInterstitial,
    defect_type="vacancy",
    atom_ids=[20],
    parent_defect_index=-1,
)

wf.GetStructureTable = GetStructureTable(
    structure_container=wf.CreateVacancyNearInterstitial
)
wf.GetDefectTable = GetDefectTable(structure_container=wf.CreateVacancyNearInterstitial)
