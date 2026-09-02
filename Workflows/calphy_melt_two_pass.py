"""Melting temperature the manual way, with the temperature window repaired once.

A calphy melting point is two reversible-scaling legs and the crossing of the
two G(T) curves they produce.  The whole calculation stands or falls on the
temperature window, and the window can be wrong in a way that *nothing reports*:
above T_m the liquid is the stable phase and sits at ``temperature_min`` quite
happily, while a defect-free periodic solid superheats past ``temperature_max``.
Both legs then finish clean, calphy raises nothing, and the two curves simply
never cross.  ``FindMeltingTemperature`` returns their closest approach for
this — a window endpoint — with no way to tell it from a result.

Those curves are not useless, though.  They are the best measurement of T_m
available; they just place it outside the range that was sampled.
``SuggestTemperatureWindow`` reads that off them and returns the window that
*would* have contained it, which is what makes a second pass possible:

    SystemWithPotential ──system──┬─> SuperheatingWindow
                                  │            │  T range
                                  │            v
        InputClass ──inp──────────┼───> MeltingPass (pass 1)
                                  │            │  T_melt, window, bracketed
                                  │            v
        InputClass ──inp──────────┴───> MeltingPass (pass 2)
                                               │
                                    T_melt, bracketed
                                               │
                                               └─> LegQuality

Four group nodes, each a block that would otherwise be five or six nodes on the
canvas, and one of them — ``MeltingPass`` — instantiated twice, because the two
passes *are* the same calculation on a different window.  Double-click a group
on the canvas to open it, or read ``wf.pass2.subgraph`` in Python; every node
inside is the same node it was when this workflow was flat, cached the same way
by ``store``.

``system`` is one port carrying a ``CalphySystem``: the crystal, its melt seed
and the potential name.  They travel together because they have to stay
consistent and every way they can fall out of step is silent — a window
measured with one potential against legs run with another is meaningless
without failing, and a liquid cell that is not the crystal's cell makes the two
free energies incomparable.  Likewise the sampling settings are typed once, in
one ``InputClass``, and each pass stamps its own window on a copy with
``ApplyTemperatureWindow``: free energies computed with different switching
lengths cannot be compared, and two hand-written copies of the same settings can
drift apart without anything noticing.

``bracketed`` on the *second* pass is the verdict.  ``True`` means the crossing
was measured inside the window both legs ran in, which is the only version of
this number worth quoting.  ``False`` means the second window still missed, and
its ``temperature_min`` / ``temperature_max`` are where a third pass belongs —
at which point use ``CalphyMeltingTemperatureSearch`` instead, which does exactly this
loop with a bracket that tightens on every attempt (see
``Workflows/calphy_melt_auto_window.py``).  At the demo settings below
(``n_switching_steps=5000``, one iteration) that does happen for the default
potential: runs have ended with pass 2 measuring the crossing inside its window
and with pass 2 placing it at the window's top edge, unbracketed.  The leg noise
there is comparable to the window width, which is what the switching length buys.

So why run it by hand at all?  Because each leg is a node with its own
diagnostics, cached independently by ``store``, and inspectable between passes.
That is what you want while deciding on cell size, switching length or a new
potential — the adaptive node is the right tool once those are settled.

Usage
-----
    import sys
    sys.path[:0] = ["pyiron_core/src", "/path/to/pyiron_nodes"]
    exec(open("Workflows/calphy_melt_two_pass.py").read())
    wf.run()
    print(wf.pass2.outputs.T_melt.value,
          wf.pass2.outputs.bracketed.value)

Then read, in this order:

  * ``wf.system.outputs.potentials`` — the list the potential was chosen from,
    and which entry ``index`` picked.  Check this first: everything
    below is a property of that potential, not of aluminium.
  * ``wf.pass1`` vs ``wf.pass2`` — how far the first window was off, and whether
    the correction landed.  ``wf.pass1.outputs.bracketed == False`` is the case
    this workflow exists for: the first window missed, silently, and the second
    was placed on the extrapolated crossing.  ``True`` means the starting window
    was already fine and pass 2 is a confirmation run centred on the crossing —
    the normal outcome for a well-behaved potential, and not a wasted pass, since
    a crossing sitting near a window edge was measured with one phase barely
    metastable.
  * ``wf.pass1.outputs.fig`` / ``wf.pass2.outputs.fig`` — the two G(T) pairs.
    When the first window missed, the first plot shows two curves that never
    meet and the second shows them crossing.
  * ``wf.checks.outputs.solid_melting`` — ``nan`` means the final solid leg
    stayed crystalline throughout.  A number means it partially melted, and the
    solid free energy above that temperature is not trustworthy however clean
    the fit looks.  Nothing else catches this: partial melting biases G without
    raising anything.
  * ``wf.checks.outputs.solid_table`` / ``.liquid_table`` —
    ``rs_max_dissipation`` should be below calphy's 1e-4 eV/atom.  Above it the
    switching was too fast; raise ``n_switching_steps``.  A bracketed crossing
    says T_m is in range, not that the free energies either side of it are
    accurate.
  * ``wf.checks.outputs.solid_hysteresis`` / ``.liquid_hysteresis`` — forward and
    backward G(T) should lie on top of each other.

When a pass misbehaves, open it: the legs, their windows and their diagnostics
are all inside, and ``wf.pass2.subgraph`` reaches them from Python.
"""

from core import Workflow, group_node

from pyiron_nodes.atomistic.property.calphy import InputClass

# ── Group node factories ───────────────────────────────────────────────────


@group_node("system", "potentials")
def SystemWithPotential(
    name="Al",
    cubic=True,
    repeat_scalar=4,
    stdev=0.5,
    type_filter="all",
    index=0,
):
    """A crystal, a melt seed and the potential to run both with.

    The three come out as one ``CalphySystem`` port, so the window estimate and
    all four legs are wired to the same crystal and the same potential by
    construction rather than by remembering to.

    Every parameter here is named after the inner port it feeds (``name`` for
    ``Bulk``, ``stdev`` for ``Rattle``, ``index`` for ``GetPotential``).  That is
    not cosmetic: a group binds a scalar parameter to an inner port by name
    first, and only falls back to searching for a port holding the same *value*
    when no name matches — which for an integer 0 can find any boolean port and
    silently rewire the group when the workflow is regenerated from the canvas.

    ``potentials`` is exposed alongside the system because ``index`` is a
    position in a catalogue, not a choice: index 0 is the *first match* for the
    structure, not the best potential for it.  For Al that is
    1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1, an alloy potential that accepts an
    Al-only cell and runs without complaint, but whose Al melts near 320 K
    against an experimental 933 K.  Read the list and set the index deliberately
    for anything quantitative — index 5 is 1999--Mishin-Y--Al--LAMMPS--ipr1, an
    Al potential fitted as one.

    The default is left at 0 on purpose, because that potential is the marginal
    case this workflow is built to show.  Its pristine 256-atom cell superheats
    to somewhere in 370-390 K, and where in that spread the ramp stops varies
    from run to run, so ``SuperheatingWindow`` has come back with both 313-368 K
    (contains T_m) and 332-391 K (entirely above it) for the same input.  Neither
    run fails.  ``pass1.bracketed`` is what distinguishes them, and pass 2 is
    centred on the crossing either way — which is the argument for running two
    passes rather than trusting one.

    ``type_filter`` is the other knob: calphy can only issue one ``pair_coeff``
    line, so prefer "eam" (which covers eam/alloy and eam/fs) or "meam" over a
    plain funcfl entry if the structure ever grows a second element.
    """
    from pyiron_nodes.atomistic.engine.lammps import GetPotential
    from pyiron_nodes.atomistic.property.calphy import BuildCalphySystem
    from pyiron_nodes.atomistic.structure.build import Bulk
    from pyiron_nodes.atomistic.structure.transform import Rattle, Repeat
    from core import Workflow

    inner = Workflow("SystemWithPotential")
    inner.crystal = Bulk(name=name, cubic=cubic)
    inner.supercell = Repeat(structure=inner.crystal, repeat_scalar=repeat_scalar)

    # The liquid leg needs a disordered starting point.  melting_cycle=True (the
    # InputClass default) then melts it properly at temperature_high and verifies
    # it, so the Rattle only has to be non-crystalline.
    inner.melt_seed = Rattle(structure=inner.supercell, stdev=stdev)

    inner.catalogue = GetPotential(
        structure=inner.supercell,
        type_filter=type_filter,
        index=index,
    )
    inner.bundle = BuildCalphySystem(
        structure=inner.supercell,
        liquid_structure=inner.melt_seed,
        potential=inner.catalogue.outputs.potential_name,
    )
    return inner.bundle, inner.catalogue.outputs.potentials


@group_node("temperature_min", "temperature_max")
def SuperheatingWindow(system, t_start=100.0, t_stop=3000.0):
    """A first temperature window, measured with the potential calphy will use.

    Heats the crystal until it loses order and puts a window below that point,
    which is the cheapest thing that is better than a guess.  ``t_start`` /
    ``t_stop`` are only search bounds and cost nothing when wide: the NPT ramp
    stops as soon as the crystal melts, so ``t_stop=3000`` is not 3000 K worth
    of MD.

    The ASE engine is built inside, from the bundle's own potential name, for
    the reason the bundle exists: a window measured with a different potential
    from the one the legs run is silently meaningless.
    """
    from pyiron_nodes.atomistic.engine.ase import LammpsEngine
    from pyiron_nodes.atomistic.property.calphy import (
        CalphySystem,
        EstimateCalphyTemperatureRange,
    )
    from core import Workflow

    inner = Workflow("SuperheatingWindow")
    inner.parts = CalphySystem(input=system)
    inner.ase_engine = LammpsEngine(potential=inner.parts.outputs.potential)
    inner.estimate = EstimateCalphyTemperatureRange(
        structure=inner.parts.outputs.structure,
        engine=inner.ase_engine,
        t_start=t_start,
        t_stop=t_stop,
    )
    inner.estimate.inputs.add(
        "store", port_type=bool, default=False, value=True, has_explicit_default=True
    )
    return (
        inner.estimate.outputs.temperature_min,
        inner.estimate.outputs.temperature_max,
    )


@group_node(
    "T_melt",
    "temperature_min",
    "temperature_max",
    "bracketed",
    "fig",
    "solid_diagnostics",
    "liquid_diagnostics",
)
def MeltingPass(system, inp, temperature_min, temperature_max, fit_order=1):
    """One melting-point attempt: two reversible-scaling legs and their crossing.

    The two G(T) legs are run in the given window and
    ``SuggestTemperatureWindow`` reads the crossing off them.  Its outputs are
    both the answer and the input to the next attempt: ``T_melt`` with
    ``bracketed`` saying whether that temperature was measured or extrapolated,
    and ``temperature_min`` / ``temperature_max`` re-centred on it, unchanged in
    width — a width both phases already survived is the one to keep, and
    widening is what pushes them back into metastability.

    ``inp`` carries the sampling settings and is left alone; the window is
    stamped onto a copy, so one ``InputClass`` can feed every pass without any
    of them being able to disagree about switching length.

    ``fit_order`` is a parameter rather than a constant because it depends on
    the pass: 1 for a first window whose curves may not cross at all (a higher
    order would wander), 2 once the crossing is bracketed and curvature is real.

    ``solid_diagnostics`` / ``liquid_diagnostics`` are the per-leg bundles, for
    ``LegQuality``.
    """
    from pyiron_nodes.atomistic.property.calphy import (
        ApplyTemperatureWindow,
        CalphySystem,
        LiquidFreeEnergyWithTemp,
        PlotSolidLiquidFreeEnergy,
        SolidFreeEnergyWithTemp,
        SuggestTemperatureWindow,
    )
    from core import Workflow

    inner = Workflow("MeltingPass")
    inner.run_input = ApplyTemperatureWindow(
        inp=inp,
        temperature_min=temperature_min,
        temperature_max=temperature_max,
    )
    inner.parts = CalphySystem(input=system)

    inner.solid_leg = SolidFreeEnergyWithTemp(
        inp=inner.run_input,
        structure=inner.parts.outputs.structure,
        potential=inner.parts.outputs.potential,
    )
    inner.liquid_leg = LiquidFreeEnergyWithTemp(
        inp=inner.run_input,
        structure=inner.parts.outputs.liquid_structure,
        potential=inner.parts.outputs.potential,
    )

    inner.crossing = SuggestTemperatureWindow(
        temp_solid=inner.solid_leg.outputs.temperature,
        fe_solid=inner.solid_leg.outputs.free_energy,
        temp_liquid=inner.liquid_leg.outputs.temperature,
        fe_liquid=inner.liquid_leg.outputs.free_energy,
        fit_order=fit_order,
    )
    # The plot fits the same order, so the drawn curve is the one the crossing
    # was read off.  (A group records a scalar parameter against one inner port
    # only, so code regenerated from the canvas hardcodes the literal here --
    # harmless, it changes nothing but the fitted line in the figure.)
    inner.plot = PlotSolidLiquidFreeEnergy(
        temp_solid=inner.solid_leg.outputs.temperature,
        fe_solid=inner.solid_leg.outputs.free_energy,
        temp_liquid=inner.liquid_leg.outputs.temperature,
        fe_liquid=inner.liquid_leg.outputs.free_energy,
        T_melt=inner.crossing.outputs.T_melt,
        fit_order=fit_order,
    )
    return (
        inner.crossing.outputs.T_melt,
        inner.crossing.outputs.temperature_min,
        inner.crossing.outputs.temperature_max,
        inner.crossing.outputs.bracketed,
        inner.plot,
        inner.solid_leg.outputs.diagnostics,
        inner.liquid_leg.outputs.diagnostics,
    )


@group_node(
    "solid_melting",
    "solid_table",
    "liquid_table",
    "solid_hysteresis",
    "liquid_hysteresis",
)
def LegQuality(solid_diagnostics, liquid_diagnostics):
    """Whether the two legs that produced the answer can be believed.

    A leg that returned without raising is one that did not fail outright, which
    is not the same as one that did not partially transform.  ``solid_melting``
    catches exactly that: splitting the diagnostics bundle exposes ``simfolder``,
    and ``CalphyMeltingFromTrajectory`` re-reads the frames calphy already dumped
    during the sweep and classifies each one, at no simulation cost.  ``nan``
    means the crystal held.  When it did not, the frame-by-frame classification
    behind that number is on the inner node's ``sweep`` port
    (``wf.checks.subgraph`` reaches it).

    The tables and hysteresis plots are the other half: forward and backward
    G(T) should lie on top of each other, and ``rs_max_dissipation`` should be
    below calphy's own 1e-4 eV/atom threshold.
    """
    from pyiron_nodes.atomistic.property.calphy import (
        CalphyDiagnostics,
        CalphyDiagnosticsTable,
        CalphyMeltingFromTrajectory,
        PlotCalphyHysteresis,
    )
    from core import Workflow

    inner = Workflow("LegQuality")
    inner.solid_parts = CalphyDiagnostics(input=solid_diagnostics)
    inner.melting = CalphyMeltingFromTrajectory(
        simfolder=inner.solid_parts.outputs.simfolder
    )

    inner.solid_numbers = CalphyDiagnosticsTable(diagnostics=solid_diagnostics)
    inner.liquid_numbers = CalphyDiagnosticsTable(diagnostics=liquid_diagnostics)
    inner.solid_sweep = PlotCalphyHysteresis(diagnostics=solid_diagnostics)
    inner.liquid_sweep = PlotCalphyHysteresis(diagnostics=liquid_diagnostics)
    return (
        inner.melting.outputs.t_melt,
        inner.solid_numbers,
        inner.liquid_numbers,
        inner.solid_sweep,
        inner.liquid_sweep,
    )


# ── The workflow ───────────────────────────────────────────────────────────

wf = Workflow("calphy_melt_two_pass")

# `index` picks the potential out of `wf.system.outputs.potentials` -- read that
# list before trusting any number below.
wf.system = SystemWithPotential(name="Al", repeat_scalar=4, index=0)

# Sampling settings, typed once for both passes.  Short switching so the demo
# finishes in about two minutes per pass; raise n_switching_steps towards
# calphy's 50000 for anything quantitative, and n_iterations above 1 to get an
# error bar rather than a single sample.  The temperature window is *not* set
# here — each pass stamps its own onto a copy.
wf.settings = InputClass(
    n_equilibration_steps=2500,
    n_switching_steps=5000,
    n_iterations=1,
)

wf.window = SuperheatingWindow(system=wf.system.outputs.system)

# Pass 1: run the window the estimator proposed.
wf.pass1 = MeltingPass(
    system=wf.system.outputs.system,
    inp=wf.settings,
    temperature_min=wf.window.outputs.temperature_min,
    temperature_max=wf.window.outputs.temperature_max,
    fit_order=1,
)

# Pass 2: the same two legs, in the window pass 1 asked for.  Its `bracketed` is
# the verdict — FindMeltingTemperature would return a number here either way and
# cannot tell you which.
wf.pass2 = MeltingPass(
    system=wf.system.outputs.system,
    inp=wf.settings,
    temperature_min=wf.pass1.outputs.temperature_min,
    temperature_max=wf.pass1.outputs.temperature_max,
    fit_order=2,
)

wf.checks = LegQuality(
    solid_diagnostics=wf.pass2.outputs.solid_diagnostics,
    liquid_diagnostics=wf.pass2.outputs.liquid_diagnostics,
)
