"""Melting temperature of fcc Al, with the temperature window found and repaired.

The fragile part of a calphy melting-point calculation is the temperature
window, and it is fragile in both directions at once.  Reversible scaling holds
the thermostat at ``temperature`` and reaches ``temperature_stop`` by scaling
the potential, so the *solid* leg is superheated at the top of the window and
the *liquid* leg undercooled at the bottom.  A window that contains the melting
point necessarily pushes both phases into metastability; a window that does not
contain it produces two free-energy curves that never cross.  And calphy does
not degrade gracefully when either phase gives up — it aborts, with
``MeltedError`` or ``SolidifiedError``, saying nothing about where the window
should have been.

Hard-coding the window means knowing the answer before computing it, and the
answer depends on the potential rather than on the element.  So this workflow
measures a starting window from the potential it is about to use, and then lets
the melting-point search move it:

    Bulk -> Repeat --+-> LammpsEngine -> EstimateCalphyTemperatureRange
                     |                            |
                     |                     InputClass(temperature=...,
                     |                               temperature_stop=...)
                     |                            |
                     +-> Rattle -----> CalphyMeltingTemperatureSearch
                                          |    (shifts the window until both
                                          |     phases survive and the crossing
                                          |     falls inside it)
                     +--> T_melt, converged, report, both G(T) legs, diagnostics

Two nodes carry the load and it is worth being clear about which does what:

``EstimateCalphyTemperatureRange`` finds the highest temperature a pristine
crystal survives a sustained NPT hold, using ``LammpsEngine`` so the window is
measured with the *same* potential calphy then runs.  That is an upper bound on
the melting point, not the melting point: a defect-free periodic cell has no
nucleation site.  Its job is to get the search started in the right hundred
kelvin, which costs about a minute against several minutes per calphy leg.  How
far above T_m that bound sits depends on the potential, and it is not always
inside the margin the estimator leaves — for the Ni-Al-H potential mentioned
below, the whole estimated window lands above the melting point.

``CalphyMeltingTemperatureSearch`` runs the two legs and turns every outcome into a
bound on T_m, then places the next window inside the resulting bracket: solid
melted, T_m is at or below where the trajectory shows it went; liquid froze, T_m
is above ``temperature_min``; no crossing in the window, the phase with the lower
G is the stable one there, which bounds T_m from that side too.  Reading the
melting temperature back out of the dumped trajectory is what makes a failed leg
worth its cost — it moves the window to where the crystal was actually seen to
go rather than by a fixed step.

Check ``converged`` before using ``T_melt``: ``False`` means the returned number
is an extrapolation, good as the next starting guess and nothing more.  In that
case ``temperature_min`` / ``temperature_max`` are the window to re-run in — feed
them into a fresh ``InputClass`` instead of guessing.  ``report`` shows every
window tried and why each was abandoned.

Usage
-----
    import sys
    sys.path[:0] = ["pyiron_core/src", "/path/to/pyiron_nodes"]
    exec(open("Workflows/calphy_melt_auto_window.py").read())
    wf.run()
    print(wf.melting_temperature.outputs.T_melt.value,
          wf.melting_temperature.outputs.converged.value)

Then read, in this order:

  * ``wf.melting_temperature.outputs.report`` — the windows tried and why each
    was abandoned.  A search that moved several times says the starting window
    was poor; one that ran out of attempts says ``T_melt`` is not a result yet.
  * ``wf.melting_temperature.outputs.temperature_min`` / ``temperature_max`` —
    where to run next when ``converged`` is ``False``.  Rebuild the input with
    ``InputClass(temperature=..., temperature_stop=...)`` from these two and run
    again; that is the whole of the manual second pass.  (``report`` will
    normally show the search having tried this window itself and run out of
    attempts, so raising ``max_attempts`` is the other way to spend the same
    compute.)
  * ``wf.solid_stability.outputs.t_melt`` — ``nan`` means the final solid leg
    stayed crystalline throughout.  A number means it partially melted, and the
    solid free energy above that temperature is not trustworthy however clean
    the fit looks.  Nothing else catches this: partial melting biases G without
    raising anything.
  * ``wf.solid_table`` / ``wf.liquid_table`` — ``rs_max_dissipation`` should be
    below calphy's 1e-4 eV/atom.  Above it, the switching was too fast; raise
    ``n_switching_steps``.  A converged window says the crossing is in range,
    not that the free energies either side of it are accurate.
  * ``wf.solid_hysteresis`` / ``wf.liquid_hysteresis`` — forward and backward
    G(T) should lie on top of each other.
  * ``wf.window_scan`` — solid fraction along the heating ramp, i.e. the
    evidence behind the starting window.
"""

from core import Workflow

from pyiron_nodes.atomistic.engine.ase import LammpsEngine
from pyiron_nodes.atomistic.property.calphy import (
    CalphyDiagnostics,
    CalphyDiagnosticsTable,
    CalphyMeltingFromTrajectory,
    CalphyMeltingTemperatureSearch,
    EstimateCalphyTemperatureRange,
    InputClass,
    PlotCalphyHysteresis,
    PlotSolidLiquidFreeEnergy,
)
from pyiron_nodes.atomistic.structure.build import Bulk
from pyiron_nodes.atomistic.structure.transform import Rattle, Repeat
from pyiron_nodes.dataframe import GetColumnFromDataFrame
from pyiron_nodes.plotting import Scatter

# Name the potential; do not let `GetPotential` choose.  It returns the first
# catalogue match, which for Al is the Ni-Al-H alloy potential
# 1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1 -- that accepts an Al-only structure
# and runs without complaint, but its Al melts near 400 K rather than 930 K.
# (For Cu the first match is a Ni-Cu potential, with the same problem.)
#
# That potential is also the worked example of a starting window that misses:
# its pristine cell superheats to roughly 800 K, so EstimateCalphyTemperatureRange
# proposes something like 680-800 K -- entirely above T_m, and entirely without
# failing.  Both calphy legs run, and the search has to notice that G_liquid is
# below G_solid throughout and move down onto the extrapolated crossing.  Swap
# POTENTIAL for it to watch that happen in `report`.
POTENTIAL = "1999--Mishin-Y--Al--LAMMPS--ipr1"

wf = Workflow("calphy_melt_auto_window")

# ── Structure ──────────────────────────────────────────────────────────────
wf.fcc_al = Bulk(name="Al", cubic=True)
wf.supercell = Repeat(structure=wf.fcc_al, repeat_scalar=5)  # 500 atoms

# ── Starting window, measured with the potential calphy will use ───────────
wf.lammps_engine = LammpsEngine(potential=POTENTIAL)

# t_start/t_stop are only search bounds and cost nothing when wide: the NPT
# heating ramp stops as soon as the crystal loses order, so t_stop=3000 is not
# 3000 K worth of MD.
wf.window = EstimateCalphyTemperatureRange(
    structure=wf.supercell,
    engine=wf.lammps_engine,
    t_start=100.0,
    t_stop=3000.0,
)
wf.window.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.calphy_input = InputClass(
    temperature=wf.window.outputs.temperature_min,
    temperature_stop=wf.window.outputs.temperature_max,
)

# ── The melting point, with the window free to move ────────────────────────
# The liquid leg needs a disordered starting point.  melting_cycle=True (the
# InputClass default) then melts it properly at temperature_high and verifies
# it, so the Rattle only has to be non-crystalline.
wf.liquid_seed = Rattle(structure=wf.supercell, stdev=0.5)

wf.melting_temperature = CalphyMeltingTemperatureSearch(
    inp=wf.calphy_input,
    structure=wf.supercell,
    liquid_structure=wf.liquid_seed,
    potential=POTENTIAL,
)
wf.melting_temperature.inputs.add(
    "store", port_type=bool, default=False, value=True, has_explicit_default=True
)

wf.free_energy_plot = PlotSolidLiquidFreeEnergy(
    temp_solid=wf.melting_temperature.outputs.temp_solid,
    fe_solid=wf.melting_temperature.outputs.fe_solid,
    temp_liquid=wf.melting_temperature.outputs.temp_liquid,
    fe_liquid=wf.melting_temperature.outputs.fe_liquid,
    T_melt=wf.melting_temperature.outputs.T_melt,
)

# ── Did the solid actually stay solid? ─────────────────────────────────────
# Splitting the diagnostics bundle exposes `simfolder`, which is all
# CalphyMeltingFromTrajectory needs: it re-reads the frames calphy dumped
# during the sweep and classifies each one, at no simulation cost.  A window
# the search accepted is one where neither phase failed outright -- which is
# not the same as neither phase having partially transformed.
wf.solid_diagnostics = CalphyDiagnostics(
    input=wf.melting_temperature.outputs.solid_diagnostics
)
wf.solid_stability = CalphyMeltingFromTrajectory(
    simfolder=wf.solid_diagnostics.outputs.simfolder
)

wf.solid_table = CalphyDiagnosticsTable(
    diagnostics=wf.melting_temperature.outputs.solid_diagnostics
)
wf.liquid_table = CalphyDiagnosticsTable(
    diagnostics=wf.melting_temperature.outputs.liquid_diagnostics
)
wf.solid_hysteresis = PlotCalphyHysteresis(
    diagnostics=wf.melting_temperature.outputs.solid_diagnostics
)
wf.liquid_hysteresis = PlotCalphyHysteresis(
    diagnostics=wf.melting_temperature.outputs.liquid_diagnostics
)

# The ramp that produced the starting window, so the choice can be inspected
# rather than taken on trust.  The `ramp` DataFrame also has a `leg` column
# separating the heating ramp from the bisection holds, and `volume_per_atom`,
# which shows the ~5 % expansion that constant-volume MD cannot allow and
# therefore cannot melt through.
wf.ramp_temperature = GetColumnFromDataFrame(
    df=wf.window.outputs.ramp, column_name="temperature"
)
wf.ramp_solid_fraction = GetColumnFromDataFrame(
    df=wf.window.outputs.ramp, column_name="solid_fraction"
)
wf.window_scan = Scatter(x=wf.ramp_temperature, y=wf.ramp_solid_fraction)
