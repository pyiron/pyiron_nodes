# Define functions needed to construct and utilize a hash based database for node storage
# Could/should be later moved to pyiron_workflows

from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.engine import reflection

from datetime import datetime
from pathlib import Path

# from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from dataclasses import dataclass

import sqlalchemy
import hashlib
import json
import pandas as pd
import os


def compute_hash_value(input_dict, length=256):
    """
    Converts a dictionary into a hash.

    This function converts a dictionary into a JSON object to ensure a
    consistent hash and hashes it using SHA-256 by default but can be
    adjusted for SHA-512 or MD5.

    Parameters:
    input_dict (dict): The dictionary to convert into a hash.
    length (int, optional): The desired length of the hash, either
                             128, 256 (default), or 512.

    Returns:
    str: The hexadecimal representation of the hash of the dictionary,
         matching the specified length.
    """

    # Convert dictionary to JSON object to ensure consistent hash
    jsonified_dict = json.dumps(input_dict, sort_keys=True)

    # Create a new hash object based on specified length
    if length == 256:
        hasher = hashlib.sha256()
    elif length == 512:
        hasher = hashlib.sha512()
    elif length == 128:
        hasher = hashlib.md5()
    else:
        raise ValueError("Length must be either 128, 256, or 512.")

    # Update the hash object with the bytes of the JSON object
    hasher.update(jsonified_dict.encode('utf-8'))

    # Return the hexadecimal representation of the hash
    hash_value = hasher.hexdigest()
    # print('hash: ', hash_value, input_dict)
    return hash_value


Base = declarative_base()


class Node(Base):
    __tablename__ = 'node'

    node_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    hash_value = Column(String)
    lib_path = Column(String)
    creation_date = Column(DateTime, default=datetime.utcnow)
    data = Column(JSON)
    output_ready = Column(Boolean)
    file_path = Column(String)

    def __init__(self, name, input_dict, path, length=256):
        """
        Initializes a Node object with a name, input dictionary, hash, file path,
        and creation date.

        The function hashes the input dictionary and stores both the dictionary and its hash,
        and the file path where further data is stored. The `node_id` and `creation_date`
        are auto-generated.

        Parameters:
        name (str): The name of the node.
        input_dict (dict): The dictionary to convert into a hash.
        path (str): The path where further data is stored.
        length (int, optional): The desired length of the hash, either 128, 256 (default), or 512.
        """

        self.name = name
        self.hash_value = compute_hash_value(input_dict, length=length)
        # print('Node_init: ', input_dict)
        self.lib_path = path  # str(Path(path).resolve())
        self.data = input_dict['inputs']
        self.output_ready = False
        self.file_path = ''


@dataclass
class Database:
    """
    Data class that represents a Database.

    Attributes:
    Session (sqlalchemy.orm.sessionmaker): The sessionmaker object that creates session.
    Node (Base): The SQLAlchemy Base object that represents a node.
    """
    Session: any
    Node: any


def create_nodes_table(db_url='postgresql://localhost/joerg', echo=False):
    """
    Creates a nodess table in the database.

    This function creates a SQLAlchemy engine with the provided database URL,
    sets up a sessionmaker with this engine, ensures the nodes table is
    created, and returns a Database object with the sessionmaker and the Node class.

    Parameters:
    db_url (str, optional): The database URL. Defaults to 'postgresql://localhost/joerg'.
    echo (bool, optional): If true, the engine will log all statements
                           as well as a repr() of their parameter lists to the engines logger.
                           Defaults to False.

    Returns:
    Database: a Database object with two attributes: a configured "Session" class and the "Node" class.
    """

    # Create engine
    engine = create_engine(db_url, echo=echo)

    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)

    # Create table
    Node.__table__.create(bind=engine, checkfirst=True)

    # Return Database class with Session and Node classes
    return Database(Session, Node)


def add_node_dict_to_db(db, data, name='', path='.', length=256):
    """
    Adds a dictionary as a new Node to a database.

    Parameters:
    db (Database): The Database object, which includes the Session and Node classes.
    name (str): The name of the Node.
    data (dict): The dictionary to convert into a hash and store as data.
    path (str): The path where further data is stored.
    length (int): The desired length of the hash.


    Returns:
    Node:  The Node object that was added to the database.
    """

    # Start session
    session = db.Session()

    # Create new node, without providing node_id
    new_node = db.Node(name=name, input_dict=data, path=path, length=length)

    # Check if a node with this hash exists
    exists = session.query(db.Node).filter_by(hash_value=new_node.hash_value).scalar() is not None

    # If the node does not exist, add it to the session
    if not exists:
        # Add new node to session
        session.add(new_node)

        # Commit session
        session.commit()

    # Close session
    session.close()

    # If the node was added, return it; otherwise return None
    # return new_node if not exists else None
    return exists


def edit_node_dict_in_db(db, node_id, data=None, name=None, lib_path=None, output_ready=None, file_path=None):
    """
    Edits the attributes of a Node in a database.

    Parameters:
    db (Database): The Database object, which includes the Session and Node classes.
    node_id (int): The db id of the node to be updated.
    name (str): The new name of the Node.
    data (dict): The new dictionary to convert into a hash and store as data.
    lib_path (str): The new path where further data is stored.
    output_ready (bool): Flag determines if the node's output is ready.
    file_path (str): path to node storage file ('' if no file has been stored)

    Returns:
    Node: The Node object that was updated in the database.
    """

    # Start session
    session = db.Session()

    # Get the node to update
    node_to_update = session.query(db.Node).filter_by(node_id=node_id).first()

    # If the node exists, update its attributes and commit the session
    if node_to_update:
        if name is not None:
            node_to_update.name = name
        if data is not None:
            node_to_update.input_dict = data
        if lib_path is not None:
            node_to_update.lib_path = lib_path
        if output_ready is not None:
            node_to_update.output_ready = output_ready
        if file_path is not None:
            node_to_update.file_path = str(file_path)

        # Commit session
        session.commit()

    # Close session
    session.close()

    # If the node was updated, return it; otherwise return None
    return node_to_update if node_to_update else None


def remove_nodes_from_db(db, indices=None, verbose=False):
    """
    Deletes selected/named rows in the nodes table.

    Args:
        db (Database): A Database instance containing the Session and Node classes.
        indices (list): Optional. List of indices to delete. If None, deletes all rows.
        verbose (bool): If True, prints out messages about what the function does.

    Returns:
        Database: The original Database instance passed in.
    """
    # Extract Session and Node classes from Database instance
    Session = db.Session
    Node = db.Node

    # Start a new session
    session = Session()

    ids_to_delete = []

    if indices is None:
        # Delete all rows
        session.query(Node).delete()
        message = "All rows deleted. TODO: Delete all files!"
    else:
        # Check if provided indices exist in the database
        for index in indices:
            node = session.get(Node, int(index))  # using session.get instead of query.get
            if node is None:
                if verbose:
                    print(f"Row with id {index} does not exist.")
            else:
                print ('file path: ', node.file_path)
                remove_directory_if_contains_file(node.file_path)
                ids_to_delete.append(int(index))

        # Delete only the existing rows with the provided indices
        for index in ids_to_delete:
            session.query(Node).filter(Node.node_id == int(index)).delete(synchronize_session='fetch')
            session.expire_all()  # synchronize the session
        message = f"Rows with ids {ids_to_delete} deleted."

    session.commit()

    session.close()

    if verbose:
        print(message)

    return db


def get_node_by_hash(db, hash_value):
    """
    Queries a database for a node with a specific hash value.

    Parameters:
    db (Database): The Database object, which includes the Session and Node classes.
    hash_value (str): The hash value to query for.

    Returns:
    dict: The corresponding database row as a dict if the node with the given hash exists. None otherwise.
    """

    # Start a session
    session = db.Session()

    # Query for the node
    node = session.query(db.Node).filter_by(hash_value=hash_value).first()

    # Close the session
    session.close()

    # Declare result variable
    result = None

    # If the node is found, convert it into a dictionary
    if node:
        result = {c.key: getattr(node, c.key) for c in sqlalchemy.inspect(node).mapper.column_attrs}

    return result


def list_table(db):
    """
    Get all data in the 'node' database table as a pandas DataFrame.

    Parameters:
    db (Database): The Database object, which includes the Session and Node classes.

    Returns:
    DataFrame: The entire 'node' database table as a pandas DataFrame.
    """

    # Start session
    session = db.Session()

    # Run SELECT * FROM node
    df = pd.read_sql(session.query(db.Node).statement, session.bind)

    return df


def drop_table(db, table_name):
    """
    Deletes a table from a database.

    Parameters:
    db (Database): Database data class that contains Session as sessionmaker and Node as SQLAlchemy Base object.
    table_name (str): The name of the table to be deleted.
    """

    engine = db.Session().get_bind()

    base = declarative_base()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables[table_name]
    if table is not None:
        base.metadata.drop_all(engine, [table], checkfirst=True)


def db_query_dict(db, **kwargs):
    # Extract Session and Node classes from Database instance
    Session = db.Session
    # Node = db.Node

    # Start a new session
    session = Session()

    # Convert kwargs values to strings
    kwargs = {k: str(v) for k, v in kwargs.items()}

    # Start building the SQL query
    query_text = "SELECT * FROM node WHERE "

    # Add conditions for each key-value pair
    for key, value in kwargs.items():
        query_text += f"data ->> '{key}' = '{value}' AND "

    # Remove trailing "AND "
    query_text = query_text[:-4]

    # Execute the query
    query = text(query_text)
    result = session.execute(query)

    data = result.fetchall()
    if data:
        # Convert result to a pandas DataFrame
        df = pd.DataFrame(data)
        df.columns = result.keys()
    else:
        df = pd.DataFrame()  # empty DataFrame

    session.close()

    # return the DataFrame
    return df


def transform_data_column(df):
    """
    Transforms the 'data' column of the DataFrame into separate columns.

    Args:
        df (DataFrame): A pandas DataFrame containing the id and data columns.

    Returns:
        DataFrame: A pandas DataFrame where each key in the 'data' column is a separate column.
    """
    # Transform 'data' column into a DataFrame
    data_df = pd.json_normalize(df['data'])

    # Concatenate original DataFrame (minus 'data' column) with the new DataFrame
    df = pd.concat([df.drop('data', axis=1), data_df], axis=1)

    return df


def list_column_names(db, table_name):
    """
    Returns a list of column names from a specific table in a database.

    Parameters:
    db (Database): Database data class that must contain Session as sessionmaker and Node as SQLAlchemy Base object.
    table_name (str): The name of the table.

    Returns:
    list: A list of column names in the specified table.
    """

    # create a temporary session to get the engine
    session = db.Session()
    engine = session.bind

    # initialize inspector
    insp = reflection.Inspector.from_engine(engine)

    # get table columns
    columns = insp.get_columns(table_name)

    # close session
    session.close()

    # return list of column names
    return [column['name'] for column in columns]


def extract_node_input(node):
    """
    This function extracts input from a pyiron_workflow node object as a dictionary.

    Arguments:
    node -- the pyiron_workflow node object from which to extract the input

    Returns:
    A dictionary where each key-value pair is the name of a channel and its corresponding value.
    """
    ic = node.to_dict()['inputs']['channels']
    input_dict = {k: v['value'] for k, v in ic.items()}
    return input_dict


def extract_unique_identifier(node):
    """
    This function creates a unique identifier for a given node. In a first approach the unique identifier is created
    by combining the package identifier and label of the node. More elaborate schemas taking into account the function
    code may be considered later.

    Arguments:
    node -- the node object for which to create the unique identifier

    Returns:
    A string representing the unique identifier for the node.
    """
    return str(node.package_identifier) + "." + str(node.label)


def get_node_storage_path(node):
    node_storage_path = extract_unique_identifier(node).replace('.', '/')
    return node_storage_path


def extract_node_dictionary(node):
    """
    This function creates a dictionary from a given node. The dictionary combines the output of
    'extract_node_input' and 'create_unique_identifier'.

    Arguments:
    node -- the node object from which to create the dictionary

    Returns:
    A dictionary that contains node inputs and a unique identifier for the node.
    """
    node_input = extract_node_input(node)
    node_id = extract_unique_identifier(node)

    return {"inputs": node_input, "node_identifier": node_id}


def create_node(node_lib_path):
    """
    This function creates the node from the Workflow.create object
    specified by the node_lib_path.

    Arguments:
    node_lib_path -- a string in the form 'attr1.attr2....attrn', where each attr is an attribute
                   of the object or one of its nested attributes

    Returns:
    The node as given by the node_lib_path.
    """
    # Initial object is Workflow.create
    from pyiron_workflow import Workflow

    obj = Workflow.create

    # Split the string into individual attributes
    attrs = node_lib_path.split('.')

    # Access the nested attributes
    for attr in attrs:
        if callable(obj):
            obj = obj()
        obj = getattr(obj, attr)

    # if the final attribute is callable, call and return the result
    return obj() if callable(obj) else obj


def add_node_to_db(node, db):
    """
    Add node instance to database. Different node inputs (based on hashing it) will result in different database
    entries.

    Arguments:
    node -- the node object to add to the database
    db -- the database to which we want to add the node metadata

    Returns:
    None.
    """

    # extract input + node identifier dictionary from node
    node_dic = extract_node_dictionary(node)
    # print('add: ', node_dic)

    # Convert node_identifier to a path format by replacing '.' with '/'
    path_format = node_dic['node_identifier'].replace('.', '/')

    # Add the node's metadata to the database
    exists = add_node_dict_to_db(db, data=node_dic, path=path_format)

    # # if node instance is not yet in db save it as file (make sure that output exists)
    # if not exists:
    #     # check if node output is available/ready
    #     if not node.outputs.ready:
    #         node.run()

    return exists


def save_node(node, db):  # , node_id):
    """
    This function stores a node by setting its storage_directory attribute to a certain path.
    An additional directory is added to this path, named with the given database ID.

    Arguments:
    node -- the node object to be stored
    db -- the database dataclass object
    db_id -- the ID of the database where the node data is stored

    Returns:
       True if successful
    """

    path = get_node_storage_path(node)
    node_id = get_node_db_id(node, db)
    if node_id is None:
        add_node_to_db(node, db)
        node_id = get_node_db_id(node, db)

    # print('save: node_id: ', node_id)
    # Create a Path object and add the database id as a new directory
    file_path = Path(path) / str(node_id)

    # Set the node's storage_directory path
    node.storage_directory.path = file_path
    # node.file_path = f'/{path}/{db_id}'
    if not node.outputs.ready:
        node.run()

    edit_node_dict_in_db(db, node_id, output_ready=node.outputs.ready, file_path=file_path)
    node.save()
    return node_id


def get_node_hash(node, hash_length=256):
    hash_value = compute_hash_value(extract_node_dictionary(node), length=hash_length)
    return hash_value


def get_node_db_id(node, db):
    session = db.Session()
    hash_value = get_node_hash(node)

    q = session.query(db.Node).filter(db.Node.hash_value == hash_value).all()
    # alternative solution (fster by about a factor 2 but less pythonic)
    # query = f"SELECT * FROM node WHERE hash_value = '{hash}'"
    # result = session.execute(text(query)).fetchall()
    # result[0][0]

    if len(q) == 0:
        node_id = None
    elif len(q) == 1:
        node_id = q[0].node_id
    else:
        raise ValueError(f'Error: Multiple nodes with identical hash: {len(q)}')
    return node_id


def process_query_result(q):
    """
    Processes the result of a query.

    Parameters:
    q (list): The result of a SQLAlchemy query.

    Returns:
    None if list is empty, the list if it contains one element, raises an error otherwise.
    """

    query_len = len(q)
    assert query_len <= 1, "Query result should not contain more than one element."

    q_out = None if query_len == 0 else q[0]

    return q_out


def get_node_from_db_id(node_id, db):
    session = db.Session()

    # Check if a node with this node_id exists
    q = session.query(db.Node).filter_by(node_id=node_id).scalar()

    node = None
    if q is not None:
        # print('query: ', q.node_id, q.hash_value)
        lib_path = q.lib_path
        node_lib_path = '.'.join(lib_path.split('/')[1:])
        # print('path: ', node_lib_path)

        node = create_node(node_lib_path)

        if q.file_path != '':
            file_path = Path(q.file_path)

            # Set the node's storage_directory path
            node.storage_directory.path = file_path
            node.load()
        else:
            # load input from database
            node = set_node_input(node, q)

    return node


def remove_directory_if_contains_file(dir_path, filename='project.h5'):
    """
    Removes a directory if it contains a specific file and no other files or directories.

    Parameters:
    dir_path (str): The path to the directory.
    filename (str): The name of the file to check for. Default is 'parameter.h5'.

    Returns:
    None
    """

    file_path = os.path.join(dir_path, filename)

    # Check if the specified file is in the directory
    if os.path.exists(file_path):
        # Get list of all files and directories in dir_path
        directory_contents = os.listdir(dir_path)

        # Remove the file
        os.remove(file_path)

        # Check if there are no other files or directories in dir_path
        if len(directory_contents) == 1:
            # Delete the directory
            os.rmdir(dir_path)
        else:
            print(f"The directory {dir_path} contains other files or directories.")

    else:
        print(f"The directory {dir_path} does not contain the file {filename}.")


def set_node_input(node, q):
    data = q.data
    for key, value in data.items():
        node.inputs[key] = eval(value)

    return node


def run_node(node, db, verbose=False):
    node_id = get_node_db_id(node, db)
    if node_id is None:
        # node instance not yet in db
        new_node = node  # TODO implement .copy()
        out = new_node.run()
        save_node(node, db)
    else:
        new_node = get_node_from_db_id(node_id, db)
        if verbose:
            print (f'node with id={node_id} is loaded rather than recomputed')

    return new_node