from pyiron_nodes.databases.node_hash_db import CreateDB, DeleteDB, GetUpstreamGraph, ShowTable
from core import Workflow
from core import group_node

wf = Workflow("db_table")

wf.CreateDB = CreateDB(host='localhost', database='none')

wf.ShowTable = ShowTable(db=wf.CreateDB)

wf.DeleteDB = DeleteDB(db=wf.CreateDB, reinitialize=True, delete_hash_files=True)

wf.GetUpstreamGraph = GetUpstreamGraph(db=wf.CreateDB, node_id=2)
