import os
import shutil
import subprocess
from dataclasses import asdict
from typing import Optional, Literal

from ase.atoms import Atoms

from core import as_function_node

import pandas as pd

from ase import Atoms
from core import as_function_node
from dataclasses import dataclass
from sphinx_parser.toolkit import to_sphinx
from sphinx_parser.jobs import set_base_parameters
from sphinx_parser.output import collect_energy_dat

@dataclass
class SphinxIOBundle:
    structure: Atoms
    sphinx_input=None
    working_directory: str = "."
    sphinx_input_filename: str


def CreateSphinxStructure(
        structure: Atoms,
        working_directory: str = ".",
):
    io_bundle=SphinxIOBundle(
        structure=structure,
        working_directory=working_directory
    )

    input_sx = set_base_parameters(structure)
    io_bundle.sphinx_input=input_sx

    return io_bundle

def CreateSphinxMinimizeInput():
    return None

def RunSphinxCalculation():
    return None

def ParseSphinxOutput():
    return None 