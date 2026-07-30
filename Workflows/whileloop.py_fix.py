from core import Workflow, as_function_node, group_node

# ── Local node definitions ──────────────────────


@as_function_node
class Node:
    """Base graph node with hierarchical parent reference.

    Parameters
    ----------
    label:
        Human-readable node identifier.
    parent:
        Parent :class:`GroupNode` when nested; ``None`` for top-level nodes.
    """

    def __init__(self, label: str, parent: Optional["GroupNode"] = None):
        self.label = label
        self.parent = parent
        self._graph = None
        self.expanded: bool = False
        self.inputs = PortCollection(self)
        self.outputs = PortCollection(self)

    # ------------------------------------------------------------------ #
    #  Port registration helpers                                           #
    # ------------------------------------------------------------------ #

    def add_input(
        self,
        name: str,
        port_type: type | str | None = None,
        default: Any = None,
        value: Any = None,
        has_explicit_default: bool = False,
    ) -> None:
        """Add an input port; raises if name already exists."""
        if name in self.inputs.ports:
            raise ValueError(
                f"Input port '{name}' already exists on node '{self.label}'"
            )
        self.inputs.add(name, port_type, default, value, has_explicit_default)

    def add_output(
        self,
        name: str,
        port_type: type | str | None = None,
        default: Any = None,
        value: Any = None,
        has_explicit_default: bool = False,
    ) -> None:
        """Add an output port; raises if name already exists."""
        if name in self.outputs.ports:
            raise ValueError(
                f"Output port '{name}' already exists on node '{self.label}'"
            )
        self.outputs.add(name, port_type, default, value, has_explicit_default)

    # ------------------------------------------------------------------ #
    #  Execution                                                           #
    # ------------------------------------------------------------------ #

    def execute(self) -> None:
        """Run node logic. Must be implemented by concrete subclasses."""
        raise NotImplementedError("Subclasses must implement execute()")

    def run(self, db=None) -> Any:
        import getpass
        from datetime import datetime

        try:
            import pyiron_database
        except ImportError:
            pyiron_database = None

        wants_storage = _node_wants_storage(self)

        # ── Attempt restore (only if store=True AND outputs were stored) ──
        if wants_storage and pyiron_database is not None and db is not None:
            try:
                restored = pyiron_database.restore_node_outputs(self, db)
                if restored:
                    return self._collect_outputs()
            except Exception:
                pass

        # ── Execute ───────────────────────────────────────────────────────
        self._start_time = datetime.now()
        self.execute()
        out = self._collect_outputs()
        self._execution_time = (datetime.now() - self._start_time).total_seconds()
        self._user = getpass.getuser()

        # Storage is now handled entirely by Graph.run() which has access
        # to the full graph needed for upstream hash resolution.
        # Node.run() no longer calls store directly.

        return out

    def _handle_storage_warning(
        self,
        exc: Exception,
        StorageSkippedWarning: type | None,
    ) -> None:
        """
        Route a storage exception to the appropriate output channel.

        * :class:`StorageSkippedWarning` — non-fatal; logged at WARNING
        level so the GUI log panel shows it.
         * Any other exception — re-raised (storage failures are errors).
        """
        import logging

        if StorageSkippedWarning is not None and isinstance(exc, StorageSkippedWarning):
            # Non-fatal — log so GUI log panel picks it up.
            logging.warning("Node '%s': storage skipped — %s", self.label, exc)
            # Also store the message on the node so the GUI can display it
            # directly (e.g. as a node badge or in the output panel).
            self._storage_warning = str(exc)
        else:
            raise

    def _collect_outputs(self) -> Any:
        """Return single value or tuple depending on output count."""
        outputs = list(self.outputs.values())
        if len(outputs) == 1:
            return outputs[0].value
        return tuple(p.value for p in outputs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Functional-style invocation: set inputs then :meth:`run`."""
        input_names = list(self.inputs.ports.keys())
        if len(args) > len(input_names):
            raise TypeError(
                f"{self.__class__.__name__} takes at most "
                f"{len(input_names)} positional arguments ({len(args)} given)"
            )
        for name, value in zip(input_names, args):
            self.inputs[name].value = value
        for key, value in kwargs.items():
            if key not in self.inputs:
                raise TypeError(f"Unexpected keyword argument '{key}'")
            self.inputs[key].value = value
        return self.run()

    def pull(self) -> Any:
        """Pull-based execution. Returns same format as run().

        When called as a standalone calculator (e.g. from MapCalculatorOnStructures),
        this node must execute unconditionally even if the top-level graph
        classifies it as reference-only. The force_execute flag overrides
        the _is_node_reference_only skip for this specific node label.
        """
        if self._graph is None:
            self.execute()
            outputs = list(self.outputs.ports.values())
            if len(outputs) == 1:
                return outputs[0].value
            return {p.label: p.value for p in outputs if p.label != "self"}

        result = self._graph.pull_node(self.label, force_execute=self.label)
        if len(result) == 1:
            return list(result.values())[0]
        return tuple(result.values())

    # ------------------------------------------------------------------ #
    #  Value resolution (used by FunctionNode and executor)               #
    # ------------------------------------------------------------------ #

    def _get_value(self, inp_port: Any, inp_type: Any) -> Any:
        """Resolve the actual argument value for a port during execution."""
        if isinstance(inp_port, Node):
            val = (
                inp_port.copy()
                if inp_type == "Node"
                else inp_port.outputs.data["value"][0]
            )
        elif isinstance(inp_port, Port):
            val = inp_port.node if inp_type in ("Node", Node) else inp_port.value
        elif hasattr(inp_port, "data"):
            val = inp_port.data["node"][0]
        else:
            val = inp_port

        if isinstance(val, Node):
            try:
                try:
                    import pyiron_database as _pdb
                except ImportError:
                    _pdb = None
                val._hash_parent = _pdb.get_hash(val) if _pdb else None
            except Exception as exc:
                logging.warning(
                    "Error hashing node %s: %s", getattr(val, "label", None), exc
                )
                val._hash_parent = None

        return val

    # ------------------------------------------------------------------ #
    #  Copy / serialisation                                                #
    # ------------------------------------------------------------------ #

    def copy(self) -> "Node":
        """Deep-copy without graph association (for closure semantics)."""
        cls = self.__class__
        new_node: Node = cls.__new__(cls)
        new_node.label = self.label
        new_node.parent = self.parent
        new_node._graph = self._graph
        new_node.expanded = getattr(self, "expanded", False)
        for attr in ("func", "state", "_module_path"):
            if hasattr(self, attr):
                setattr(new_node, attr, getattr(self, attr))

        new_node.inputs = PortCollection(new_node)
        for name, port in self.inputs.ports.items():
            new_node.inputs.add(
                name,
                port.type,
                port.default,
                port.value,
                has_explicit_default=port._has_explicit_default,
            )
            new_node.inputs[name].ready = port.ready

        new_node.outputs = PortCollection(new_node)
        for name, port in self.outputs.ports.items():
            new_node.outputs.add(
                name,
                port.type,
                port.default,
                port.default,
                has_explicit_default=port._has_explicit_default,
            )
            new_node.outputs[name].ready = False

        if self._graph is not None:
            new_node._graph = self._graph.copy()
            new_node._graph.nodes[self.label] = new_node
        else:
            new_node._graph = None
        return new_node

    def __getstate__(self) -> dict:
        return {
            "type": getattr(
                self,
                "_module_path",
                f"{self.__class__.__module__}.{self.__class__.__name__}",
            ),
            "label": self.label,
            "parent": (
                self.parent
                if isinstance(self.parent, str)
                else getattr(self.parent, "label", None) if self.parent else None
            ),
            "expanded": getattr(self, "expanded", None),
            "inputs": {
                name: {"value": port.value}
                for name, port in self.inputs.ports.items()
                if port.value != port.default
                and not isinstance(port.value, (Node, Port))
                and not (hasattr(port, "connections") and port.connections)
            },
        }

    def __setstate__(self, state: dict) -> None:
        self.label = state.get("label", "")
        self.parent = None
        self.expanded = state.get("expanded", False)
        if "type" in state:
            self._module_path = state["type"]
        # Guard: inputs may not exist if __init__ was bypassed via __new__
        if not hasattr(self, "inputs"):
            self.inputs = PortCollection(self)
            self.outputs = PortCollection(self)
            self._graph = None
        for inp_name, inp_data in state.get("inputs", {}).items():
            if inp_name in self.inputs:
                self.inputs[inp_name].value = inp_data["value"]
        if "state" in state:
            self.state = state["state"]


wf = Workflow("whileloop1")

wf.stop = Node(i=5)

wf.recursive_node = Node(x=0, stop_at=wf.stop)

wf.loop = Node(max_steps=20, recursive_function=wf.recursive_node)
