from core import Node, as_function_node
import getpass

USERNAME = getpass.getuser()

from typing import Literal


@as_function_node
def CreateDB(
    user: str = USERNAME,
    password: str = "none",
    host: str = "130.183.217.189",
    port: int = 5432,
    database: str = "pyiron",
    table_name: Literal["test_nodes_cmmc", "nodes_cmmc"] = "test_nodes_cmmc",
):
    import pyiron_database

    if database == "none":
        database = user

    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    db = pyiron_database.PostgreSQLInstanceDatabase(
        connection_str, table_name=table_name, storage_path=table_name
    )
    db.init()

    return db


@as_function_node
def DeleteDB(db, reinitialize: bool = False, delete_hash_files: bool = False):
    """
    Delete the database and all its contents!

    Args:
        db: Database instance
        reinitialize: If True, reinitialize the database after deletion (default: False)
        delete_hash_files: If True, also delete hash files stored in
                          ~/pyiron_core_data/.storage (default: False)
    """
    db.drop()

    # Delete hash files if requested
    if delete_hash_files:
        import pathlib
        import shutil
        from core.config import paths as _cfg_paths

        # Get the storage path from config
        # storage_path = pathlib.Path(_cfg_paths.DATA_STORAGE)

        # if storage_path.exists():
        #     print(f"[DeleteDB] Removing hash files from: {storage_path}")
        #     # Remove the entire storage directory
        #     shutil.rmtree(storage_path)
        #     # Recreate the empty directory
        #     storage_path.mkdir(parents=True, exist_ok=True)
        #     print(f"[DeleteDB] Hash files deleted and directory recreated")
    if delete_hash_files:
        db.cleanup_storage()

    if reinitialize:
        db.init()
    return db


@as_function_node
def DeleteNode(db, index: int = 0):
    """Delete the node at row *index* from the database.

    Looks up the node's hash by its position in the table and removes that
    entry.  The associated HDF5 output file (if any) is also deleted.

    Args:
        db: InstanceDatabase connection.
        index: 0-based row index of the node to delete.

    Returns:
        db: The database connection (pass-through for chaining).
    """
    import pandas as pd
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db.engine)
    session = Session()
    df = pd.read_sql(session.query(db.table).statement, session.bind)
    session.close()

    if index >= len(df):
        print(f"DeleteNode: index {index} out of range ({len(df)} entries).")
        result = db
        return result

    node_hash = df["hash"].iloc[index]
    output_path = df["output_path"].iloc[index]

    db.delete(node_hash)

    if output_path:
        import pathlib

        p = pathlib.Path(output_path)
        if p.exists():
            p.unlink()

    print(f"Deleted node at index {index} (hash {node_hash[:16]}…)")
    result = db
    return result


@as_function_node
def ShowTable(db):
    import pandas as pd
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db.engine)
    session = Session()

    df = pd.read_sql(session.query(db.table).statement, session.bind)

    session.close()

    return df


@as_function_node
def FilterTable(
    db,
    qualname: str = "",
    module: str = "",
    user: str = "",
    has_output: bool = False,
):
    """Filter the database table by one or more criteria (AND-combined).

    Each non-empty string argument requires an exact match on the
    corresponding column.  ``has_output=True`` keeps only rows that have a
    stored HDF5 output file.  The returned DataFrame includes an ``id``
    column with the original row index so the result can be passed directly
    to ``GetUpstreamGraph`` or ``GetDownstreamNodes``.

    Args:
        db: InstanceDatabase connection.
        qualname: Exact match on qualname, e.g. ``"Bulk"``.
                  Empty string = no filter.
        module: Exact match on module, e.g. ``"pyiron_nodes.atomistics"``.
                Empty string = no filter.
        user: Exact match on user.  Empty string = no filter.
        has_output: If True, keep only rows whose ``output_path`` is set.

    Returns:
        pd.DataFrame: Filtered rows with an added ``id`` column (0-based
        row index in the full table).
    """
    import pandas as pd
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db.engine)
    session = Session()
    df = pd.read_sql(session.query(db.table).statement, session.bind)
    session.close()

    df.insert(0, "id", range(len(df)))

    if qualname:
        df = df[df["qualname"] == qualname]
    if module:
        df = df[df["module"] == module]
    if user:
        df = df[df["user"] == user]
    if has_output:
        df = df[df["output_path"].notna()]

    result = df.reset_index(drop=True)
    return result


@as_function_node
def GetDownstreamNodes(db, node_id: int, qualname: str = ""):
    """Find all nodes that directly consume the node at row *node_id* as input.

    Connected inputs are stored in the ``inputs`` JSONB column as
    ``"{upstream_hash}@{port_name}"`` strings.  This function searches every
    row for values that contain the target node's hash, then optionally
    filters the results by qualname.

    The returned DataFrame includes an ``id`` column with the original row
    index in the full table, which can be passed directly to
    ``GetUpstreamGraph`` or chained into further ``GetDownstreamNodes`` calls.

    Args:
        db: InstanceDatabase connection.
        node_id: Row index of the upstream node in the full database table.
        qualname: If given, keep only downstream nodes whose qualname matches
                  exactly.  E.g. ``"RunNEB"`` to find only NEB calculations.
                  Empty string = return all downstream nodes.

    Returns:
        pd.DataFrame: Rows for all matching downstream nodes with an added
        ``id`` column (0-based row index in the full table).
    """
    import pandas as pd
    from sqlalchemy import cast, Text
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db.engine)
    session = Session()
    df_full = pd.read_sql(session.query(db.table).statement, session.bind)
    session.close()

    if node_id >= len(df_full):
        result = pd.DataFrame()
        return result

    target_hash = df_full["hash"].iloc[node_id]

    # Build a full-table index so we can report id values in the result.
    df_full.insert(0, "id", range(len(df_full)))

    # Connected inputs are encoded as "{hash}@{port}", so any row whose
    # inputs JSON text contains the target hash is a direct downstream node.
    with db.engine.connect() as conn:
        stmt = db.table.select().where(
            cast(db.table.c.inputs, Text).like(f"%{target_hash}%")
        )
        query_result = conn.execute(stmt)
        rows = query_result.fetchall()
        col_names = list(query_result.keys())

    if not rows:
        result = pd.DataFrame(columns=["id"] + col_names)
        return result

    df_downstream = pd.DataFrame(rows, columns=col_names)

    # Attach the original row index from the full table.
    df_downstream = df_downstream.merge(df_full[["id", "hash"]], on="hash", how="left")
    # Move id to the front.
    cols = ["id"] + [c for c in df_downstream.columns if c != "id"]
    df_downstream = df_downstream[cols]

    if qualname:
        df_downstream = df_downstream[df_downstream["qualname"] == qualname]

    result = df_downstream.reset_index(drop=True)
    return result


@as_function_node
def GetNode(db, node_id: int):
    """
    Get the graph of a node with id *node_id from the database.
    """
    import pandas as pd
    from sqlalchemy.orm import sessionmaker

    import pyiron_database
    from pyironflow.gui_utilities import GuiGraph

    Session = sessionmaker(bind=db.engine)
    session = Session()

    df = pd.read_sql(session.query(db.table).statement, session.bind)

    session.close()

    _, graph = pyiron_database.restore_node_from_database(
        db=db, node_hash=df.hash.iloc[node_id]
    )

    gui_graph = GuiGraph(graph)

    return gui_graph


@as_function_node
def GetHash(node: Node):
    """
    Get the hash of a node
    """
    import pyiron_database

    print("inputs: ", node.inputs)
    hash = pyiron_database.get_hash(node)
    return hash


@as_function_node
def GetIdFromHash(db, node_hash: str = ""):
    """Return the 0-based row index of the node whose hash matches *node_hash*.

    Args:
        db: InstanceDatabase connection.
        node_hash: Full or leading-prefix SHA-256 hash string to look up.

    Returns:
        index (int) of the matching row, or -1 if not found.
    """
    import pandas as pd
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db.engine)
    session = Session()
    df = pd.read_sql(session.query(db.table).statement, session.bind)
    session.close()

    matches = df.index[df["hash"].str.startswith(node_hash)].tolist()
    index = matches[0] if matches else -1
    return index


@as_function_node
def GetStoredOutput(db, index: int = 0):
    """
    Return the stored outputs of the node at row *index* in the database table.

    On success returns a dict mapping output-port name → stored value.
    On failure (no HDF5 file found or index out of range) returns an error
    string that is displayed in the GUI output window.
    """
    import pandas as pd
    from sqlalchemy.orm import sessionmaker
    import pyiron_database
    from pyiron_database.instance_database.node import restore_node_outputs

    Session = sessionmaker(bind=db.engine)
    session = Session()
    df = pd.read_sql(session.query(db.table).statement, session.bind)
    session.close()

    if index >= len(df):
        result = (
            f"Error: index {index} is out of range — "
            f"the table contains {len(df)} entr{'y' if len(df) == 1 else 'ies'}."
        )
        return result

    node_hash = df.hash.iloc[index]

    node, _ = pyiron_database.restore_node_from_database(db=db, node_hash=node_hash)

    success = restore_node_outputs(node, storage_path=db.storage_path)

    if not success:
        result = (
            f"Error: no stored output found for node at index {index} "
            f"(hash: {node_hash[:16]}...). "
            f"The node was either not executed with store=True or its HDF5 file is missing."
        )
        return result

    outputs = {name: port.value for name, port in node.outputs.items()}
    result = next(iter(outputs.values())) if len(outputs) == 1 else outputs
    return result


@as_function_node
def GetUpstreamGraph(db, node_id: int, group: bool = False, workflow_name: str = None):
    """
    Get the upstream workflow containing the node with id *node_id* from the database.

    This function restores the complete upstream workflow including the specified node
    and all nodes it depends on (connected via input edges). This is useful for
    understanding the full computation graph that feeds into a particular node.

    Unlike GetNode which shows only the single node, GetUpstreamGraph recursively
    restores all upstream nodes connected through input dependencies.

    The returned Graph object will automatically be opened as a new workflow tab
    in the GUI.

    Args:
        db: InstanceDatabase connection
        node_id: Integer ID/index of the node in the database table
        group: If True, collapse all nodes in the restored graph into a single
               GroupNode before opening the tab.  The GroupNode can be expanded
               in the GUI to inspect the inner workflow.  Default: False.
        workflow_name: When given, the upstream graph is merged into the existing
               tab whose label matches this name.  If no such tab exists, a new
               tab with this label is created.  When None (default) a fresh tab
               is always created.

    Returns:
        Graph: The complete upstream workflow ready for display in a new tab.
               When group=True the graph contains a single collapsed GroupNode.
    """
    import pandas as pd
    from sqlalchemy.orm import sessionmaker

    import pyiron_database

    # Get the hash for the specified node
    Session = sessionmaker(bind=db.engine)
    session = Session()

    df = pd.read_sql(session.query(db.table).statement, session.bind)
    session.close()

    node_hash = df.hash.iloc[node_id]

    # Restore the node - this recursively restores upstream connected nodes
    _, graph = pyiron_database.restore_node_from_database(db=db, node_hash=node_hash)

    # Set a meaningful label for the graph
    if workflow_name is not None:
        graph.label = workflow_name
        graph._target_workflow = workflow_name
    elif graph.label is None or graph.label == "":
        graph.label = f"Upstream_{node_id}"

    if group and graph.nodes:
        group_name = graph.label or f"upstream_{node_id}"
        graph.group_nodes(list(graph.nodes.keys()), group_name=group_name)

    return graph
