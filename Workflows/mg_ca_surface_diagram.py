from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.dpg2026.atomistic.calculator.optimize import GenericOptimizerSettings, Relax
from pyiron_nodes.dpg2026.atomistic.engine.grace import Grace
from core import Workflow
from core import group_node
from core import as_function_node
import pandas as pd

# ── Local node definitions ──────────────────────

@as_function_node("df")
def BuildDecoratedStructures(
    host_structure: Atoms,
    solute: str = "Ca",
    host: str = "Mg",
    max_solute_atoms: int = 4,
    n_seeds: int = 3,
) -> pd.DataFrame:
    """
    Return a DataFrame with the pristine surface plus n_seeds random
    realisations of 1..max_solute_atoms solute-for-host substitutions.

    Columns: structure (ASE Atoms), name (str, legend label).
    """
    from pyiron_nodes.atomistic.structure.build_point_defects import (
        make_pristine_reference,
        make_config_row,
        op_substitute,
        expand_configs,
    )

    atoms0, pristine_pos = make_pristine_reference._original_func(host_structure)
    base = make_config_row(
        atoms=atoms0,
        structure_id="surface_pristine",
        events=[],
        seed=0,
        pristine_n_sites=len(atoms0),
        pristine_positions=pristine_pos,
    )
    base_df = pd.DataFrame([base])

    rows = [base_df]
    for n_solute in range(1, max_solute_atoms + 1):
        kwargs_list = [
            {"from_element": host, "to_element": solute, "n": n_solute, "seed": s}
            for s in range(n_seeds)
        ]
        rows.append(
            expand_configs._original_func(
                base_df, op_substitute._original_func, kwargs_list, keep_input=False
            )
        )

    combined = pd.concat(rows, ignore_index=True)
    combined["structure"] = combined["atoms"]

    def _name(row):
        events = row.get("events") or []
        n = sum(
            1
            for e in events
            if e.get("type") == "substitution" and e.get("to") == solute
        )
        if n == 0:
            return "pristine"
        seed = row.get("seed", 0) or 0
        return f"{solute}_{n} (seed {seed})"

    combined["name"] = [_name(r) for _, r in combined.iterrows()]
    return combined[["structure", "name"]]


@as_function_node("df")
def RelaxStructuresDataFrame(
    df: pd.DataFrame,
    relax_node: Node,
) -> pd.DataFrame:
    """
    Relax every structure in df['structure'] using the provided relax_node
    template (engine and settings already connected).

    Returns df with 'structure' (relaxed ASE Atoms) and 'energy' (eV) columns.
    """
    from pyiron_nodes.atomistic.structure._atoms import to_ase

    relaxed, energies = [], []
    for s in df["structure"]:
        relax_node.inputs.structure = to_ase(s)
        out = relax_node.pull()
        relaxed.append(to_ase(out.structure))
        energies.append(out.energy)

    result = df.copy()
    result["structure"] = relaxed
    result["energy"] = energies
    return result


@as_function_node("mu_sweep")
def ChemicalPotentialSweep(
    mu_ref: float, delta_mu: float = -1.5, num_points: int = 200
):
    """Return num_points values linearly spaced from mu_ref to mu_ref + delta_mu."""
    import numpy as np

    return np.linspace(mu_ref, mu_ref + delta_mu, int(num_points))


# ── Group node factories ─────────────────────────────

@group_node("surface")
def mg_surface(size='1 1 1', vacuum=1.0):
    from pyiron_nodes.atomistic.structure.build import Surface
    from core import Workflow
    
    inner_wf = Workflow("mg_surface")
    inner_wf.slab = Surface(element='Mg', surface_type='hcp0001', size=size, vacuum=vacuum, orthogonal=True)
    return inner_wf.slab

@group_node("mu")
def mu_mg(structure, engine, optimizer_settings=None):
    from pyiron_nodes.dpg2026.atomistic.calculator.optimize import Relax
    from core import Workflow
    from core import as_function_node
    
    @as_function_node("mu")
    def _energy_per_atom(calc_result):
        from pyiron_nodes.atomistic.structure._atoms import to_ase
    
        s = to_ase(calc_result.structure)
        return calc_result.energy / len(s)
    
    inner_wf = Workflow("mu_mg")
    inner_wf.relaxed = Relax(structure=structure, engine=engine, opt_parameters=optimizer_settings, opt_mode='full')
    inner_wf.relaxed.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)
    inner_wf.mu = _energy_per_atom(calc_result=inner_wf.relaxed)
    return inner_wf.mu

@group_node("mu")
def mu_ca_ref(structure, engine, optimizer_settings=None):
    from pyiron_nodes.dpg2026.atomistic.calculator.optimize import Relax
    from core import Workflow
    from core import as_function_node
    
    @as_function_node("mu")
    def _energy_per_atom(calc_result):
        from pyiron_nodes.atomistic.structure._atoms import to_ase
    
        s = to_ase(calc_result.structure)
        return calc_result.energy / len(s)
    
    inner_wf = Workflow("mu_ca_ref")
    inner_wf.relaxed = Relax(structure=structure, engine=engine, opt_parameters=optimizer_settings, opt_mode='full')
    inner_wf.relaxed.inputs.add("store", port_type=bool, default=False, value=True, has_explicit_default=True)
    inner_wf.mu = _energy_per_atom(calc_result=inner_wf.relaxed)
    return inner_wf.mu

@group_node("result")
def FormationEnergies(df, mu_host, mu_solute):
    from pyiron_nodes.atomistic.thermodynamics.defect_phases import AddDefectConcentrationColumns, AddElementCountColumns, ComputeDefectFormationEnergy
    from core import Workflow
    from core import as_function_node
    
    @as_function_node("chemical_potentials")
    def _pack_chemical_potentials(mu_host, mu_solute, host="Mg", solute="Ca"):
        return {host: mu_host, solute: mu_solute}
    
    inner_wf = Workflow("FormationEnergies")
    inner_wf.chemical_potentials = _pack_chemical_potentials(mu_host=mu_host, mu_solute=mu_solute)
    inner_wf.with_counts = AddElementCountColumns(df=df)
    inner_wf.with_deltas = AddDefectConcentrationColumns(df=inner_wf.with_counts)
    inner_wf.formation_energies = ComputeDefectFormationEnergy(df=inner_wf.with_deltas, chemical_potentials=inner_wf.chemical_potentials)
    return inner_wf.formation_energies

wf = Workflow("mg_ca_surface_diagram")

wf.bulk_ca = Bulk(name='Ca', crystalstructure='fcc', a=5.59)

wf.bulk_mg = Bulk(name='Mg', crystalstructure='hcp', a=3.21, c=5.21, orthorhombic=True)

wf.grace_engine = Grace(model='GRACE-2L-OAM')

wf.mg_surface = mg_surface(size='2 2 6', vacuum=10.0)

wf.optimizer_settings = GenericOptimizerSettings(max_steps=500, force_tolerance=0.001)

wf.decorated_surface_structures = BuildDecoratedStructures(host_structure=wf.mg_surface)

wf.mu_mg = mu_mg(structure=wf.bulk_mg, engine=wf.grace_engine, optimizer_settings=wf.optimizer_settings)

wf.mu_ca_ref = mu_ca_ref(structure=wf.bulk_ca, engine=wf.grace_engine, optimizer_settings=wf.optimizer_settings)

wf.surface_relax_template = Relax(engine=wf.grace_engine, opt_parameters=wf.optimizer_settings, opt_mode='full')
wf.surface_relax_template.inputs.add("store", port_type=bool, default=False, value=False, has_explicit_default=True)

wf.mu_ca_sweep = ChemicalPotentialSweep(mu_ref=wf.mu_ca_ref)

wf.relaxed_surface_df = RelaxStructuresDataFrame(df=wf.decorated_surface_structures, relax_node=wf.surface_relax_template)

wf.FormationEnergies = FormationEnergies(df=wf.relaxed_surface_df, mu_host=wf.mu_mg, mu_solute=wf.mu_ca_ref)
