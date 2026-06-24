import os
import shutil
import subprocess
from dataclasses import asdict
from typing import Optional, Literal

from ase.atoms import Atoms

from core import as_function_node

import pandas as pd

from ase import Atoms
from core import as_function_node, as_out_dataclass_node
from dataclasses import dataclass
from sphinx_parser.toolkit import to_sphinx
from sphinx_parser.jobs import set_base_parameters
from sphinx_parser.output import collect_energy_dat

from core.data_fields import DataArray, EmptyArrayField


@dataclass
class SphinxIOBundle:
    structure: Atoms
    sphinx_input = None
    working_directory: str = "."
    sphinx_input_filename: str = "input.sx"


@as_function_node
def CreateSphinxInput(
    structure: Atoms,
    k_point_folding: str,
    eCut: float = 350.,
    xc: str = "PBE",
    smearing_width: float = 0.2,
    smearing_type: Literal["gaussian", "fermi-dirac", "fermi-dirac-1", "methfessel-paxton"] = "gaussian",
    k_point_coords: str = "0.5 0.5 0.5",
    working_directory: str = ".",
):
    io_bundle = SphinxIOBundle(structure=structure, working_directory=working_directory)

    k_point_coords = [float(x) for x in k_point_coords.split()]
    k_point_folding
    main_group = sphinx.main(
        scfDiag=sphinx.main.scfDiag(
            maxSteps=maxSteps, blockCCG=sphinx.main.scfDiag.blockCCG()
        )
    )
    pawPot_group = get_paw_from_structure(structure)
    basis_group = sphinx.basis(
        eCut=eCut, kPoint=sphinx.basis.kPoint(coords=k_point_coords), folding=k_point_folding
    )
    smearing_arg={}
    if smearing_type == "gaussian":
        smearing_arg = { "MethfesselPaxton" : 0}
    elif smearing_type == "fermi-dirac":
        smearing_arg = { "FermiDirac" : 0}
    elif smearing_type  == "fermi-dirac-1":
        smearing_arg = { "FermiDirac" : 1}
    elif smearing_type == "methfessel-paxton":
        smearing_arg = { "MethfesselPaxton" : 1}

    paw_group = sphinx.PAWHamiltonian(xc=xc, spinPolarized=spinPolarized, ekt=smearing_width,**smearing_arg)
    initial_guess_group = sphinx.initialGuess(
        waves=sphinx.initialGuess.waves(lcao=sphinx.initialGuess.waves.lcao()),
        rho=sphinx.initialGuess.rho(atomicOrbitals=True, atomicSpin=spin_lst),
    )
    input_sx = sphinx(
        pawPot=pawPot_group,
        structure=struct_group,
        main=main_group,
        basis=basis_group,
        PAWHamiltonian=paw_group,
        initialGuess=initial_guess_group,
    )
    
    io_bundle.sphinx_input = input_sx

    return io_bundle


@as_function_node
def WriteSphinxInput(
    io_bundle: SphinxIOBundle, sphinx_input_filename: str = "input.sx"
):

    io_bundle.sphinx_input_filename = sphinx_input_filename

    os.makedirs(io_bundle.working_directory, exist_ok=True)

    with open(
        os.path.join(io_bundle.working_directory, sphinx_input_filename), "w"
    ) as f:
        f.write(to_sphinx(io_bundle.sphinx_input))

    return io_bundle


def CreateSphinxMinimizeInput():
    return None


@as_function_node
def RunSphinxCalculation(io_bundle: SphinxIOBundle):
    command = ["sphinx"]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=io_bundle.working_directory,
    )

    stdout, stderr = process.communicate()

    return io_bundle, stdout, stderr


@as_out_dataclass_node
class SphinxEnergyOutput:
    scf_computation_time: DataArray = EmptyArrayField()
    scf_energy_int: DataArray = EmptyArrayField()
    scf_energy_free: DataArray = EmptyArrayField()
    scf_energy_zero: DataArray = EmptyArrayField()
    scf_energy_band: DataArray = EmptyArrayField()
    scf_electronic_entropy: DataArray = EmptyArrayField()


@as_function_node
def ParseSphinxOutput(io_bundle: SphinxIOBundle, output_filename: str = "energy.dat"):

    collected_out = collect_energy_dat(
        os.path.join(io_bundle.working_directory, output_filename)
    )

    output = SphinxEnergyOutput().pure_dataclass()

    output.scf_computation_time = collected_out["scf_computation_time"]
    output.scf_energy_int = collected_out["scf_energy_int"]
    output.scf_energy_free = collected_out["scf_energy_free"]
    output.scf_energy_zero = collected_out["scf_energy_zero"]
    output.scf_energy_band = collected_out["scf_energy_band"]
    output.scf_electronic_entropy = collected_out["scf_electronic_entropy"]

    return output
