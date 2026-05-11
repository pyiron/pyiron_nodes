from core import Node, as_function_node
import getpass

USERNAME = getpass.getuser()


@as_function_node
def CreateDB(
    user: str = USERNAME,
    password: str = "none",
    host: str = "localhost",
    port: int = 5432,
    database: str = "none",
):
    import pyiron_database

    if database == "none":
        database = user

    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    db = pyiron_database.PostgreSQLInstanceDatabase(connection_str)
    db.init()

    return db


@as_function_node
def DeleteDB(db):
    """
    Delete the database and all its contents!
    """

    db.drop()
    db.init()
    return db


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
def GetUpstreamGraph(db, node_id: int):
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
    
    Returns:
        Graph: The complete upstream workflow ready for display in a new tab
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
    _, graph = pyiron_database.restore_node_from_database(
        db=db, node_hash=node_hash
    )
    
    # Set a meaningful label for the graph based on the node_id
    # This ensures the tab shows a proper name when the graph is opened
    if graph.label is None or graph.label == "":
        graph.label = f"Upstream_{node_id}"

    return graph
