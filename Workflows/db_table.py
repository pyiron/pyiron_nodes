from core import Workflow
from pyiron_nodes.databases.node_hash_db import (
    CreateDB,
    DeleteDB,
    DeleteNode,
    FilterTable,
    GetDownstreamNodes,
    GetIdFromHash,
    GetStoredOutput,
    GetUpstreamGraph,
    ShowTable,
)

wf = Workflow("db_table")

wf.CreateDB = CreateDB(host="localhost", database="none")

wf.ShowTable = ShowTable(db=wf.CreateDB)

wf.FilterTable = FilterTable(db=wf.CreateDB)

wf.GetDownstreamNodes = GetDownstreamNodes(db=wf.CreateDB, node_id=0)

wf.DeleteDB = DeleteDB(db=wf.CreateDB, reinitialize=True, delete_hash_files=True)

wf.DeleteNode = DeleteNode(db=wf.CreateDB, index=0)

wf.GetUpstreamGraph = GetUpstreamGraph(db=wf.CreateDB, node_id=2)

wf.GetIdFromHash = GetIdFromHash(db=wf.CreateDB)

wf.GetStoredOutput = GetStoredOutput(db=wf.CreateDB, index=0)
