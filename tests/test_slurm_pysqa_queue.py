"""``SlurmExecutor`` with a pysqa config: ``partition`` must select the queue.

pysqa ignores ``partition`` and, without a ``queue``, falls back to the site's
``queue_primary`` — so the translation is what decides where the jobs run.
"""

import pytest

from pyiron_nodes.executors import _pysqa_queue

QUEUE_YAML = """\
queue_type: SLURM
queue_primary: cmti
queues:
  cmti: {cores_max: 1200, cores_min: 1, run_time_max: 345600, script: cmti.sh}
  s_cmmg: {cores_max: 256, cores_min: 1, run_time_max: 345600, script: cmmg.sh}
"""


@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "queue.yaml").write_text(QUEUE_YAML)
    return str(tmp_path)


@pytest.mark.parametrize("partition", ["s.cmmg", "s_cmmg"])
def test_partition_or_queue_name_selects_the_queue(config_dir, partition):
    assert _pysqa_queue(config_dir, partition) == "s_cmmg"


def test_unknown_partition_fails_instead_of_falling_back_to_the_primary(config_dir):
    with pytest.raises(ValueError, match="choose one of"):
        _pysqa_queue(config_dir, "normal")
