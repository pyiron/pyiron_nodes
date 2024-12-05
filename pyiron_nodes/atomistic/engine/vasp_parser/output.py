import glob
import os
import shutil
import tarfile
import tempfile
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints, Vasprun

from pyiron_nodes.atomistic.engine.vasp_parser.outcar import Outcar

def is_line_in_file(filepath, line, exact_match=True):
    """
    Check if a line is present in a file.

    Args:
        filepath (str): The path to the file.
        line (str): The line to search for in the file.
        exact_match (bool, optional): Determines whether the search should be an exact match (default: True).

    Returns:
        bool: True if the line is found in the file, False otherwise.

    Example:
        >>> filepath = 'path/to/your/file.txt'
        >>> line_to_search = 'Hello, world!'
        >>> exact_match = True  # Toggle this flag to change between exact and partial match

        >>> if is_line_in_file(filepath, line_to_search, exact_match):
        ...     print("Line found in the file.")
        ... else:
        ...     print("Line not found in the file.")
    """
    try:
        with open(filepath, "r") as file:
            for file_line in file:
                if exact_match and line == file_line.strip():
                    return True
                elif not exact_match and line in file_line:
                    return True
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
    return False

def check_convergence(
    directory,
    filename_vasprun="vasprun.xml",
    filename_vasplog="vasp.log",
    backup_vasplog="error.out",
):
    """
    Check the convergence status of a VASP calculation.

    Args:
        directory (str): The directory containing the VASP files.
        filename_vasprun (str, optional): The name of the vasprun.xml file (default: "vasprun.xml").
        filename_vasplog (str, optional): The name of the vasp.log file (default: "vasp.log").

    Returns:
        bool: True if the calculation has converged, False otherwise.
    """
    try:
        vr = Vasprun(filename=os.path.join(directory, filename_vasprun))
        return vr.converged
    except:
        line_converged = (
            "reached required accuracy - stopping structural energy minimisation"
        )
        try:
            return is_line_in_file(
                os.path.join(directory, filename_vasplog),
                line=line_converged,
                exact_match=False,
            )
        except:
            try:
                return is_line_in_file(
                    os.path.join(directory, backup_vasplog),
                    line=line_converged,
                    exact_match=False,
                )
            except:
                return False


def process_error_archives(directory):
    """
    Processes all tar or tar.gz files starting with 'error' in the specified directory and its subdirectories.

    Args:
        directory (str): The directory to search for tar files.

    Returns:
        pd.DataFrame: DataFrame containing the processed VASP outputs from error archives.
    """
    error_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(directory)
        for file in files
        if file.startswith("error")
        and (file.endswith(".tar") or file.endswith(".tar.gz"))
    ]

    df_list = []
    for error_file in error_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with tarfile.open(error_file, "r:*") as tar:
                    tar.extractall(path=temp_dir)
                df_list.append(pd.DataFrame(_get_vasp_outputs(temp_dir)))
            except tarfile.ReadError as e:
                print(f"Error extracting {error_file}: {e}")
            except Exception as e:
                print(f"Unexpected error processing {error_file}: {e}")

    print(f"Processing error dirs in {directory} complete.")
    return pd.concat(df_list) if df_list else pd.DataFrame()


def _get_vasp_outputs_from_files(
    structure, outcar_path="OUTCAR", incar_path="INCAR", kpoints_path="KPOINTS"
):
    file_data = {
        "POSCAR": [structure],
        "OUTCAR": [np.nan],
        "INCAR": [np.nan],
        "KPOINTS": [np.nan],
    }

    if os.path.isfile(outcar_path):
        try:
            outcar = Outcar()
            outcar.from_file(outcar_path)
            file_data["OUTCAR"] = [outcar]
        except Exception as e:
            print(f"Error reading OUTCAR file {outcar_path}: {e}")

    if os.path.isfile(incar_path):
        try:
            incar = Incar.from_file(incar_path).as_dict()
            file_data["INCAR"] = [incar]
        except Exception as e:
            print(f"Error reading INCAR file {incar_path}: {e}")

    if os.path.isfile(kpoints_path):
        try:
            kpoints = Kpoints.from_file(kpoints_path).as_dict()
            file_data["KPOINTS"] = [kpoints]
        except Exception as e:
            pass

    return pd.DataFrame(file_data)


def _get_vasp_outputs(directory, structure=None, parse_all_in_dir=True):
    outcar_files = (
        glob.glob(os.path.join(directory, "OUTCAR*"))
        if parse_all_in_dir
        else glob.glob(os.path.join(directory, "OUTCAR"))
    )

    if structure is None:
        structure = get_structure(directory)

    if outcar_files:
        data = []
        for outcar_file in outcar_files:
            suffix = os.path.basename(outcar_file).replace("OUTCAR", "")
            incar_file = os.path.join(directory, f"INCAR{suffix}")
            kpoints_file = os.path.join(directory, f"KPOINTS{suffix}")

            output_df = _get_vasp_outputs_from_files(
                structure,
                outcar_path=outcar_file,
                incar_path=incar_file,
                kpoints_path=kpoints_file,
            )
            data.append(output_df)
        data = pd.concat(data)
    else:
        data = pd.DataFrame(
            {
                "POSCAR": [structure],
                "OUTCAR": [np.nan],
                "INCAR": [np.nan],
                "KPOINTS": [np.nan],
            }
        )

    return data


def get_SCF_cycle_convergence(outcar_scf_arrays, threshold=1e-5):
    diff = outcar_scf_arrays[-1] - outcar_scf_arrays[-2]
    return abs(diff) < threshold


def _get_KPOINTS_info(KPOINTS, INCAR):
    try:
        if np.isnan(KPOINTS):
            kpoints_key = "KSPACING"
            return f"KSPACING: {INCAR.get(kpoints_key, 0.5)}"
        else:
            return KPOINTS
    except Exception as e:
        print(e)
        return np.nan


def process_outcar(outcar, structure):
    if pd.isna(outcar) or pd.isna(structure):
        warning_message = (
            "Both OUTCAR and structure data are missing. Returning DataFrame with np.nan values."
            if pd.isna(outcar) and pd.isna(structure)
            else (
                "OUTCAR data is missing. Returning DataFrame with np.nan values for OUTCAR-related fields."
                if pd.isna(outcar)
                else "Structure data is missing. Returning DataFrame with np.nan values for structure-related fields."
            )
        )
        warnings.warn(warning_message)

        return pd.DataFrame(
            [
                {
                    "calc_start_time": np.nan,
                    "consumed_time": np.nan,
                    "structures": np.nan,
                    "energy": np.nan,
                    "energy_zero": np.nan,
                    "forces": np.nan,
                    "stresses": np.nan,
                    "magmoms": np.nan,
                    "scf_steps": np.nan,
                    "scf_convergence": np.nan,
                }
            ]
        )

    try:
        energies = outcar.parse_dict["energies"]
    except:
        energies = np.nan

    try:
        ionic_step_structures = np.array(
            [
                Structure(
                    cell,
                    structure.species,
                    outcar.parse_dict["positions"][i],
                    coords_are_cartesian=True,
                ).to_json()
                for i, cell in enumerate(outcar.parse_dict["cells"])
            ]
        )
    except:
        ionic_step_structures = np.nan

    try:
        energies_zero = outcar.parse_dict["energies_zero"]
    except:
        energies_zero = np.nan

    try:
        forces = outcar.parse_dict["forces"]
    except:
        forces = np.nan

    try:
        stresses = outcar.parse_dict["stresses"]
    except:
        stresses = np.nan

    try:
        magmoms = np.array(outcar.parse_dict["final_magmoms"])
    except:
        magmoms = np.nan

    try:
        scf_steps = [len(i) for i in outcar.parse_dict["scf_energies"]]
        scf_conv_list = [
            get_SCF_cycle_convergence(
                d, threshold=outcar.parse_dict["electronic_stop_criteria"]
            )
            for d in outcar.parse_dict["scf_energies"]
        ]
    except Exception as e:
        print(e)
        scf_steps = np.nan
        scf_conv_list = np.nan

    try:
        calc_start_time = outcar.parse_dict["execution_datetime"]
    except:
        calc_start_time = np.nan

    try:
        consumed_time = outcar.parse_dict["resources"]
    except:
        consumed_time = np.nan

    return pd.DataFrame(
        [
            {
                "calc_start_time": calc_start_time,
                "consumed_time": consumed_time,
                "structures": ionic_step_structures,
                "energy": energies,
                "energy_zero": energies_zero,
                "forces": forces,
                "stresses": stresses,
                "magmoms": magmoms,
                "scf_steps": scf_steps,
                "scf_convergence": scf_conv_list,
            }
        ]
    )


def get_structure(directory):
    """
    Attempts to read the structure from various file names in the specified order.

    Args:
        directory (str): The directory where the files are located.

    Returns:
        pymatgen.core.Structure: The structure object if successful, None otherwise.
    """
    structure_filenames = ["CONTCAR", "POSCAR"] + glob.glob(
        os.path.join(directory, "starter*.vasp")
    )

    for filename in structure_filenames:
        try:
            return Structure.from_file(os.path.join(directory, filename))
        except Exception as e:
            # print(f"Failed to parse structure file {filename}: {e}")
            pass
    print("Failed to parse appropriate structure file completely")
    return np.nan


def get_vasp_outputs(directory, extract_error_dirs=True, parse_all_in_dir=True):
    df_direct_outputs = _get_vasp_outputs(directory, parse_all_in_dir=parse_all_in_dir)
    df_error_outputs = (
        process_error_archives(directory) if extract_error_dirs else pd.DataFrame()
    )
    return pd.concat([df_direct_outputs, df_error_outputs])


def grab_electron_info(
    directory_path, line_before_elec_str="PAW_PBE", potcar_filename="POTCAR"
):
    structure = get_structure(directory_path)
    if structure:
        element_list, element_count = element_count_ordered(structure)

    electron_of_potcar = []
    with open(os.path.join(directory_path, potcar_filename), "r") as file:
        lines = file.readlines()
        should_append = False
        for line in lines:
            stripped_line = line.strip()
            if should_append:
                electron_of_potcar.append(float(stripped_line))
                should_append = False
            if stripped_line.startswith(line_before_elec_str):
                should_append = True

    return element_list, element_count, electron_of_potcar


def get_total_electron_count(
    directory_path, line_before_elec_str="PAW_PBE", potcar_filename="POTCAR"
):
    ele_list, ele_count, electron_of_potcar = grab_electron_info(
        directory_path, line_before_elec_str, potcar_filename
    )
    return np.dot(ele_count, electron_of_potcar)


def element_count_ordered(structure):
    site_element_list = [site.species_string for site in structure]
    past_element = site_element_list[0]
    element_list = [past_element]
    element_count = []
    count = 0
    for element in site_element_list:
        if element == past_element:
            count += 1
        else:
            element_count.append(count)
            element_list.append(element)
            count = 1
            past_element = element
    element_count.append(count)
    return element_list, element_count


def parse_vasp_directory(directory, extract_error_dirs=True, parse_all_in_dir=True):
    df = get_vasp_outputs(
        directory,
        extract_error_dirs=extract_error_dirs,
        parse_all_in_dir=parse_all_in_dir,
    )
    results_df = []
    kpoints_list = []
    for _, row in df.iterrows():
        results_df.append(process_outcar(row.OUTCAR, row.POSCAR))
        kpoints_list.append(_get_KPOINTS_info(row.KPOINTS, row.INCAR))

    results_df = pd.concat(results_df).sort_values(by="calc_start_time")
    results_df["KPOINTS"] = kpoints_list
    results_df["INCAR"] = df["INCAR"].tolist()

    try:
        element_list, element_count, electron_of_potcar = grab_electron_info(
            directory_path=directory, potcar_filename="POTCAR"
        )
    except:
        element_list = np.nan
        element_count = np.nan
        electron_of_potcar = np.nan

    try:
        electron_count = get_total_electron_count(directory_path=directory)
    except Exception as e:
        print(e)
        electron_count = np.nan

    results_df["element_list"] = [element_list] * len(results_df)
    results_df["element_count"] = [element_count] * len(results_df)
    results_df["electron_count"] = [electron_count] * len(results_df)
    results_df["potcar_electron_count"] = [electron_of_potcar] * len(results_df)
    results_df["job_name"] = [os.path.basename(directory)] * len(results_df)
    results_df["filepath"] = [directory] * len(results_df)
    results_df["convergence"] = [check_convergence(directory)] * len(results_df)
    return results_df
