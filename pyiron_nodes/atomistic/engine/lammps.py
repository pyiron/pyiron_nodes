from __future__ import annotations

from typing import Optional

from pyiron_atomistics.lammps.control import LammpsControl
from pyiron_workflow import as_function_node

from pyiron_nodes.atomistic.calculator.data import (
    InputCalcMinimize,
    InputCalcMD,
    InputCalcStatic,
)
from pyiron_nodes.dev_tools import FileObject, parse_input_kwargs
from pyiron_nodes.dev_tools import wf_data_class
from dataclasses import asdict


@as_function_node("calculator")
def Calc(parameters):
    from pyiron_atomistics.lammps.control import LammpsControl

    calculator = LammpsControl()

    if isinstance(parameters, InputCalcMD.dataclass):
        calculator.calc_md(**asdict(parameters))
        calculator.mode = "md"
    elif isinstance(parameters, InputCalcMinimize):
        calculator.calc_minimize(**parameters)
        calculator.mode = "minimize"
    elif isinstance(parameters, InputCalcStatic):
        calculator.calc_static(**parameters)
        calculator.mode = "static"
    else:
        raise TypeError(f"Unexpected parameters type {parameters}")

    return calculator


@as_function_node("calculator")
def CalcStatic(calculator_input: Optional[InputCalcStatic | dict] = None):
    calculator_kwargs = parse_input_kwargs(calculator_input, InputCalcStatic)
    calculator = LammpsControl()
    calculator.calc_static(**calculator_kwargs)
    calculator.mode = "static"

    return calculator


@as_function_node("calculator")
def CalcMinimize(calculator_input: Optional[InputCalcMinimize | dict] = None):
    calculator_kwargs = parse_input_kwargs(calculator_input, InputCalcMinimize)
    calculator = LammpsControl()
    calculator.calc_minimize(**calculator_kwargs)
    calculator.mode = "static"

    return calculator


@as_function_node("calculator")
def CalcMD(calculator_input: Optional[InputCalcMD.dataclass] = None):
    from dataclasses import asdict

    if calculator_input is None:
        calculator_input = InputCalcMD.dataclass()

    calculator_kwargs = asdict(calculator_input)
    # calculator_kwargs = parse_input_kwargs(calculator_input, InputCalcMD)
    calculator = LammpsControl()
    calculator.calc_md(**calculator_kwargs)
    calculator.mode = "md"

    return calculator


@as_function_node("path")
def InitLammps(
    structure,
    potential: str,
    calculator,
    working_directory: str,
    create_dir: bool = True,
):
    import os
    from pyiron_atomistics.lammps.potential import LammpsPotential, LammpsPotentialFile

    if create_dir:
        os.makedirs(working_directory, exist_ok=True)
    else:
        assert os.path.isdir(
            working_directory
        ), f"working directory {working_directory} is missing, create it!"

    pot = LammpsPotential()
    pot.df = LammpsPotentialFile().find_by_name(potential)
    pot.write_file(file_name="potential.inp", cwd=working_directory)
    pot.copy_pot_files(working_directory)

    with open(os.path.join(working_directory, "structure.inp"), "w") as f:
        structure.write(f, format="lammps-data", specorder=pot.get_element_lst())

    calculator.write_file(file_name="control.inp", cwd=working_directory)

    return os.path.abspath(working_directory)


@as_function_node("log")
def ParseLogFile(log_file):
    from pymatgen.io.lammps.outputs import parse_lammps_log

    log = parse_lammps_log(log_file.path)
    if len(log) == 0:
        print(f"check {log_file.path}")
        raise ValueError("lammps_log_parser: failed")

    return log


@as_function_node("dump")
def ParseDumpFile(dump_file):
    from pymatgen.io.lammps.outputs import parse_lammps_dumps

    dump = list(parse_lammps_dumps(dump_file.path))
    return dump


@wf_data_class()
class ShellOutput:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    dump: FileObject = FileObject()  # TODO: should be done in a specific lammps object
    log: FileObject = FileObject()


@as_function_node("output", "dump", "log")
def Shell(
    working_directory: str,
    command: str = "lmp",
    environment: Optional[dict] = None,
    arguments: Optional[list] = None,
):
    arguments = ["-in", "control.inp"] if arguments is None else arguments
    # -> (ShellOutput, FileObject, FileObject):  TODO: fails -> why
    import os
    import subprocess

    if environment is None:
        environment = {}
    if arguments is None:
        arguments = []

    environ = dict(os.environ)
    environ.update({k: str(v) for k, v in environment.items()})
    # print ([str(command), *map(str, arguments)], working_directory, environment)
    # print("start shell")
    proc = subprocess.run(
        [command, *map(str, arguments)],
        capture_output=True,
        cwd=working_directory,
        encoding="utf8",
        env=environ,
    )
    # print("end shell")

    output = ShellOutput()
    output.stdout = proc.stdout
    output.stderr = proc.stderr
    output.return_code = proc.returncode
    dump = FileObject("dump.out", working_directory)
    log = FileObject("log.lammps", working_directory)

    return output, dump, log


@wf_data_class()
class GenericOutput:
    energy_pot = []
    energy_kin = []
    forces = []


@as_function_node
def Collect(
    out_dump,
    out_log,
    calc_mode: str | LammpsControl | InputCalcMinimize | InputCalcMD | InputCalcStatic,
):
    import numpy as np

    from pyiron_nodes.atomistic.calculator.data import (
        OutputCalcStatic,
        OutputCalcMinimize,
        OutputCalcMD,
    )

    log = out_log[0]

    if isinstance(calc_mode, str) and calc_mode in ["static", "minimize", "md"]:
        pass
    elif isinstance(calc_mode, (InputCalcMinimize, InputCalcMD, InputCalcStatic)):
        calc_mode = calc_mode.__class__.__name__.replace("InputCalc", "").lower()
    elif isinstance(calc_mode, LammpsControl):
        calc_mode = calc_mode.mode
    else:
        raise ValueError(f"Unexpected calc_mode {calc_mode}")

    if calc_mode == "static":
        generic = OutputCalcStatic()
        # print("output Collect: ", generic, isinstance(generic, OutputCalcStatic))
        # if isinstance(generic, OutputCalcStatic):
        generic.energy_pot = log["PotEng"].values[0]
        generic.force = np.array([o.data[["fx", "fy", "fz"]] for o in out_dump])[0]

    elif calc_mode == "minimize":
        generic = OutputCalcMinimize()

    elif calc_mode == "md":
        generic = OutputCalcMD()
        generic.energies_pot = log["PotEng"].values
        generic.energies_kin = log["TotEng"].values - generic.energies_pot
        generic.forces = np.array([o.data[["fx", "fy", "fz"]] for o in out_dump])

    return generic


@as_function_node("potential")
def Potential(structure, name=None, index=0):
    from pyiron_atomistics.lammps.potential import list_potentials as lp

    potentials = lp(structure)
    if name is None:
        pot = potentials[index]
    else:
        if name in potentials:
            pot = name
        else:
            raise ValueError("Unknown potential")
    return pot


@as_function_node("potentials")
def ListPotentials(structure):
    from pyiron_atomistics.lammps.potential import list_potentials as lp

    potentials = lp(structure)
    return potentials


def get_calculators():
    calc_dict = dict()
    calc_dict["md"] = CalcMD
    calc_dict["minimize"] = CalcMinimize
    calc_dict["static"] = CalcStatic

    return calc_dict


@as_function_node("generic")
def GetEnergyPot(generic, i_start: int = 0, i_end: int = -1):
    # print("energies_pot: ", generic.energies_pot)
    return generic.energies_pot[i_start:i_end]


from pyiron_workflow import as_macro_node

# from pyiron_workflow.pyiron_nodes.atomistic.engine.lammps import get_calculators
# from pyiron_workflow.pyiron_nodes.dev_tools import set_replacer

from ase import Atoms


@as_macro_node("generic")
def Code(
    wf,
    structure: Atoms,
    calculator=InputCalcStatic(),  # TODO: Don't use mutable defaults
    potential: Optional[str] = None,
    working_dir: str = "test2",
):
    # from pyiron_contrib.tinybase.shell import ExecutablePathResolver

    # print("Lammps: ", structure)
    wf.Potential = Potential(structure=structure, name=potential)

    wf.ListPotentials = ListPotentials(structure=structure)

    wf.calc = Calc(calculator)

    wf.InitLammps = InitLammps(
        structure=structure,
        potential=wf.Potential,
        calculator=wf.calc,
        working_directory=working_dir,
    )

    wf.Shell = Shell(
        # command=ExecutablePathResolver(module="lammps", code="lammps").path(),
        working_directory=wf.InitLammps,
    )

    wf.ParseLogFile = ParseLogFile(log_file=wf.Shell.outputs.log)
    wf.ParseDumpFile = ParseDumpFile(dump_file=wf.Shell.outputs.dump)
    wf.Collect = Collect(
        out_dump=wf.ParseDumpFile.outputs.dump,
        out_log=wf.ParseLogFile.outputs.log,
        calc_mode="md",  # wf.calc,
    )

    return wf.Collect
