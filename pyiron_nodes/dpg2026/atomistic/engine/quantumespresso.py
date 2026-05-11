import os
import subprocess
from ase.build import bulk
from ase.io import write
from core import as_function_node, Node


def _write_input(input_dict, working_directory="."):
    filename = os.path.join(working_directory, "input.pwi")
    os.makedirs(working_directory, exist_ok=True)
    write(
        filename=filename,
        images=input_dict["structure"],
        Crystal=True,
        kpts=input_dict["kpts"],
        input_data={
            "calculation": input_dict["calculation"],
            "occupations": "smearing",
            "degauss": input_dict["smearing"],
            "ecutwfc": input_dict["ecutwfc"],
        },
        pseudopotentials=input_dict["pseudopotentials"],
        tstress=True,
        tprnfor=True,
    )


def _collect_output(working_directory="."):
    from qe_xml_parser.parsers import parse_pw
    
    output = parse_pw(os.path.join(working_directory, "pwscf.xml"))
    return {
        "energy": output["energy"],
        "volume": output["ase_structure"].get_volume(),
    }


@as_function_node
def calculate_qe(working_directory, pseudopotentials, structure, encut, kpts=(3, 3, 3), store: bool=True):
    if structure is None:
        structure = bulk(name="Al", a=4.04, cubic=False)
    element = structure.get_chemical_symbols()[-1]
    input_dict = {
        "structure": structure,
        "pseudopotentials": {element: pseudopotentials},
        "kpts": kpts,
        "calculation": "scf",
        "smearing": 0.02,
        "ecutwfc": encut,
    }
    _write_input(
        input_dict=input_dict,
        working_directory=working_directory,
    )
    subprocess.check_output(
        "mpirun -np 1 pw.x -in input.pwi > output.pwo",
        cwd=working_directory,
        shell=True,
    )
    result_dict = _collect_output(working_directory=working_directory)
    energy, volume = result_dict["energy"], result_dict["volume"]
    return energy, volume


@as_function_node
def converge_energy_cutoff(dft_function: Node, limit: float = 0.0001, max_steps: int = 10):
    import numpy as np
    
    encut = dft_function.inputs.encut.value
    energy_lst = [dft_function()[0]]
    for i in range(max_steps):
        dft_function.inputs.encut.value += 1
        energy_lst.append(dft_function()[0])
        print("loop: ", dft_function.inputs.encut.value, energy_lst)

        if np.abs(energy_lst[-2] - energy_lst[-1]) < limit:
            break

    return energy_lst
