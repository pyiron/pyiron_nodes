from core import as_function_node
from functools import lru_cache


@as_function_node
@lru_cache
def Grace(model: str = "GRACE-FS-OAM", use_symmetry=True):
    """Grace potential node.

    Provides a GRACE (Universal Graph Atomic Cluster Expansion) calculator object.
    Typical use: define GRACE parameters and obtain a ready‑to‑use
    calculator.

        Parameters
        ----------
        model : str, optional
            Model identifier, default "GRACE-FS-OAM".
    use_symmetry : bool, optional
        Whether to use symmetry optimizations.

    Returns
    -------
    out
        OutputEngine object containing the initialized GRACE calculator.
        The @lru_cache decorator provides in-memory caching for model loading.
    """
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorpotential.calculator import grace_fm

    from pyiron_nodes.atomistic.engine.generic import OutputEngine

    out = OutputEngine(calculator=grace_fm(model))
    return out
