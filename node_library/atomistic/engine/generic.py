from typing import Optional
from dataclasses import field

from node_library.dev_tools import wf_data_class, wfMetaData


@wf_data_class()
class OutputEngine:
    calculator: Optional[callable] = field(default=None, metadata=wfMetaData(log_level=0))
    engine_id: Optional[int] = field(default=None, metadata=wfMetaData(log_level=0))
    parameters: Optional[wf_data_class] = field(
        default=None, metadata=wfMetaData(log_level=10)
    )
