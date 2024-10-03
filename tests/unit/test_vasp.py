import unittest
import os
import filecmp
from pathlib import Path
import shutil
from pyiron_nodes.atomistic.engine.vasp import (
    vasp_job,
    run_job,
    create_WorkingDirectory,
    VaspInput,
    parse_VaspOutput,
    get_default_POTCAR_paths,
    write_POTCAR,
    is_line_in_file,
    write_POSCAR,
    write_INCAR,
    write_KPOINTS,
    write_VaspInputSet,
    check_convergence,
    stack_element_string,
)
from pymatgen.io.vasp.inputs import Incar, Kpoints
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from typing import List, Tuple
import pandas as pd


class TestVaspJob(unittest.TestCase):
    def setUp(self):
        self.incar_dict = {
            "ENCUT": 520,
            "EDIFF": 1e-4,
            "ISMEAR": 1,
            "SIGMA": 0.1,
        }
        self.bulk_Fe = bulk("Fe", cubic=True) * [2, 2, 2]
        self.structure = AseAtomsAdaptor().get_structure(self.bulk_Fe)
        self.structure[1] = "Mn"
        self.workdir = os.path.join(os.getcwd(), "unittest_case")
        os.makedirs(self.workdir, exist_ok=True)
        self.incar = Incar.from_dict(self.incar_dict)
        resources = Path(__file__).parent.parent.joinpath("resources", "vasp")
        self.POTCAR_library_path = str(resources.joinpath("POTCAR_lib").resolve())
        self.vasp_input = VaspInput(
            self.structure, self.incar, pseudopot_lib_path=self.POTCAR_library_path
        )
        self.example_converged_path = str(resources.joinpath("example1").resolve())
        self.example_failed_path = str(resources.joinpath("example2").resolve())
        self.POTCAR_specification_data = str(
            resources.joinpath("POTCAR_lib", "pseudopotential_PBE_data.csv").resolve()
        )

    def tearDown(self):
        if os.path.exists("./POTCAR"):
            os.remove("./POTCAR")
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)

    def test_write_POTCAR(self):
        write_POTCAR(".", vasp_input=self.vasp_input)
        self.assertTrue(os.path.exists("./POTCAR"))

    def test_stack_element_string(self):
        element_list, element_count = stack_element_string(self.structure)
        self.assertEqual(element_list, ["Fe", "Mn", "Fe"])
        self.assertEqual(element_count, [1, 1, 14])

    def test_is_line_in_file(self):
        test_file = "test_file.txt"
        with open(test_file, "w") as f:
            f.write("This is a test line.\n")
        self.assertTrue(is_line_in_file(test_file, "This is a test line."))
        self.assertFalse(is_line_in_file(test_file, "This is not in the file."))
        os.remove(test_file)

    def test_create_WorkingDirectory(self):
        create_WorkingDirectory(self.workdir)()
        self.assertTrue(os.path.exists(self.workdir))

    def test_write_POSCAR(self):
        poscar_path = write_POSCAR(self.workdir, self.structure)
        self.assertTrue(os.path.exists(poscar_path))

    def test_write_INCAR(self):
        incar_path = write_INCAR(self.workdir, self.incar)
        self.assertTrue(os.path.exists(incar_path))

    def test_write_POTCAR(self):
        potcar_path = write_POTCAR(self.workdir, self.vasp_input)
        self.assertTrue(os.path.exists(potcar_path))

    def test_write_KPOINTS(self):
        kpoints = Kpoints.gamma_automatic([1, 1, 1])
        kpoints_path = write_KPOINTS(self.workdir, kpoints)
        self.assertTrue(os.path.exists(kpoints_path))

    def test_write_VaspInputSet(self):
        write_VaspInputSet(self.workdir, self.vasp_input)()
        self.assertTrue(os.path.exists(os.path.join(self.workdir, "POSCAR")))
        self.assertTrue(os.path.exists(os.path.join(self.workdir, "INCAR")))
        self.assertTrue(os.path.exists(os.path.join(self.workdir, "POTCAR")))

def test_run_job(self):
    output = run_job(f"cp -r {self.example_converged_path} .", self.workdir)()

    # Check the standard output, error, and return code
    self.assertEqual(output.stdout, "")
    self.assertEqual(output.stderr, "")
    self.assertEqual(output.return_code, 0)

    # Define the source and destination directories
    source_dir = self.example_converged_path
    destination_dir = os.path.join(self.workdir, os.path.basename(self.example_converged_path))

    # Check if destination directory exists
    self.assertTrue(os.path.exists(destination_dir), "Destination directory was not created")

    # Compare the contents of source and destination directories
    dir_comparison = filecmp.dircmp(source_dir, destination_dir)

    # Ensure the contents are identical
    self.assertEqual(dir_comparison.left_only, [], "There are files only in the source directory")
    self.assertEqual(dir_comparison.right_only, [], "There are files only in the destination directory")
    self.assertEqual(dir_comparison.diff_files, [], "There are files that differ between source and destination")

    # Optionally, compare recursively if the directories contain subdirectories
    self.assertTrue(filecmp.cmpfiles(source_dir, destination_dir, commonfiles=dir_comparison.common_files, shallow=False)[2] == [],
                    "There are differences in common files between source and destination")

    def test_parse_VaspOutput(self):
        output = parse_VaspOutput(self.example_converged_path)()
        # Assuming parse_VaspOutput returns a dictionary or similar structure
        self.assertIsInstance(output, dict)
        # Further checks can be added here depending on the actual structure of the output

    def test_check_convergence(self):
        converged = check_convergence(
            self.example_converged_path, filename_vasprun="vasprun.xml"
        )()
        self.assertTrue(converged)

        converged = check_convergence(
            self.example_failed_path, filename_vasprun="vasprun.xml"
        )()
        self.assertFalse(converged)

    def test_get_default_POTCAR_paths(self):
        paths = get_default_POTCAR_paths(
            self.structure,
            pseudopot_lib_path=self.POTCAR_library_path,
            potcar_df=pd.read_csv(self.POTCAR_specification_data),
        )
        expected_paths = [
            os.path.join(self.POTCAR_library_path, "Fe", "POTCAR"),
            os.path.join(self.POTCAR_library_path, "Mn_pv", "POTCAR"),
            os.path.join(self.POTCAR_library_path, "Fe", "POTCAR"),
        ]
        self.assertEqual(paths, expected_paths)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
