from __future__ import annotations

from core import as_function_node


@as_function_node("dataframe")
def DryadElasticity(
    url: str = "https://datadryad.org/downloads/file_stream/88988",
    max_index: int | None = None,
):
    """
    Download and parse the de Jong et al. elastic constants JSON database from
    the Dryad repository (file_stream/88988).  Appends an 'atoms' column with
    parsed ASE Atoms objects.
    """
    import io
    import urllib.request

    import pandas as pd
    from ase.io import read

    with urllib.request.urlopen(url) as response:
        raw = response.read()

    df = pd.read_json(io.BytesIO(raw))

    if max_index is None:
        max_index = len(df.structure)
    df = df.drop(index=range(max_index, len(df))).reset_index(drop=True)

    structures = []
    for structure in df.structure:
        atoms = read(io.StringIO(structure), format="cif")
        structures.append(atoms)

    df["atoms"] = structures
    return df


@as_function_node("dataframe")
def DeJong(max_index: int | None = None, file_name="ec.json"):
    """
    Expects the file to be the "ec.json" database referenced by:
    Ref. de Jong et al. https://www.nature.com/articles/sdata20159#MOESM77

    :return:
    """
    import io
    import os

    import pandas as pd
    from ase.io import read

    module_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(module_dir, file_name)
    print("de jong database, path: ", module_dir, file_path)

    df = pd.read_json(file_path)

    structures = []
    if max_index is None:
        max_index = len(df.structure)

    df = df.drop(index=range(max_index, len(df)))
    for structure in df.structure:
        f = io.StringIO(structure)
        atoms = read(f, format="cif")
        structures.append(atoms)

    df["atoms"] = structures

    return df
