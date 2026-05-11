from dataclasses import replace
import numpy as np
from core import as_function_node
import landau


@as_function_node
def TemperatureLinePhase(
        name: str,
        concentration: float,
        temperatures: np.ndarray | list[float],
        free_energies: np.ndarray | list[float],
        num_parameters: int | None = 3
):
    phase = landau.phases.TemperatureDependentLinePhase(
            name, concentration, temperatures, free_energies,
            landau.interpolate.SGTE(num_parameters)
    )
    return phase


@as_function_node
def TransitionTemperature(
        phase1, phase2,
        Tmin: int | float,
        Tmax: int | float,
) -> float:
    """Plot free energies of two phases and find their intersection, i.e. the transition temperature.

    Assumes that both phases are of the same concentration, otherwise the results will be off, as it takes the chemical
    potential difference to be zero.

    Args:
        phase1, phase2 (landau.phases.Phase): the two phases to plot
        Tmin (float): minimum temperature
        Tmax (float): maximum temperature

    Returns:
        float: transition temperature if found, else NaN
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    df = landau.calculate.calc_phase_diagram([phase1, phase2], np.linspace(Tmin, Tmax), mu=0.0, keep_unstable=True)
    try:
        fm, Tm = df.query('border and T!=@Tmin and T!=@Tmax')[['f','T']].iloc[0].tolist()
    except IndexError:
        print("Transition Point not found!")
        fm, Tm = np.nan, np.nan
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x='T', y='f',
        hue='phase',
        style='stable', style_order=[True, False],
        ax=ax,
    )
    ax.axvline(Tm, color='k', linestyle='dotted', alpha=.5)
    ax.scatter(Tm, fm, marker='o', c='k', zorder=10)

    dfa = np.ptp(df['f'].dropna())
    dft = np.ptp(df['T'].dropna())
    ax.text(Tm + .05 * dft, fm + dfa * .1, rf"$T_m = {Tm:.0f}\,\mathrm{{K}}$", rotation='vertical', ha='center')
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Free Energy [eV/atom]")
    return fig


@as_function_node
def LinePhase(name: str, concentration: float, energy: float, entropy: float) -> landau.phases.LinePhase:
    import landau
    phase = landau.phases.LinePhase(name, concentration, energy, entropy)
    return phase


@as_function_node
def TemperatureLinePhase(
        name: str,
        concentration: float,
        temperatures: np.ndarray | list[float],
        free_energies: np.ndarray | list[float],
        num_parameters: int | None = 3
):
    phase = landau.phases.TemperatureDependentLinePhase(
            name, concentration, temperatures, free_energies,
            landau.interpolate.SGTE(num_parameters)
    )
    return phase


@as_function_node
def IdealSolution(name: str, phase1: landau.phases.Phase, phase2: landau.phases.Phase) -> landau.phases.Phase:
    import landau
    phase = landau.phases.IdealSolution(name, phase1, phase2)
    return phase


def guess_mu_range(phases, Tmax, samples):
    """Guess chemical potential window from the ideal solution.

    Searches numerically for chemical potentials which stabilize
    concentrations close to 0 and 1 and then use the concentrations
    encountered along the way to numerically invert the c(mu) mapping.
    Using an even c grid with mu(c) then yields a decent sampling of mu
    space so that the final phase diagram is described everywhere equally.

    Args:
        phases: list of phases to consider
        Tmax: temperature at which to estimate 
        samples: how many mu samples to return

    Returns:
        array of chemical potentials that likely cover the whole concentration space
    """

    import scipy.optimize as so
    import scipy.interpolate as si
    import numpy as np
    # semigrand canonical "average" concentration
    # use this to avoid discontinuities and be phase agnostic
    def c(mu):
        phis = np.array([p.semigrand_potential(Tmax, mu) for p in phases])
        conc = np.array([p.concentration(Tmax, mu) for p in phases])
        phis -= phis.min()
        beta = 1/(Tmax*8.6e-5)
        prob = np.exp(-beta*phis)
        prob /= prob.sum()
        return (prob * conc).sum()
    cc, mm = [], []
    mu0, mu1 = 0, 0
    while (ci := c(mu0)) > 0.001:
        cc.append(ci)
        mm.append(mu0)
        mu0 -= 0.05
    while (ci := c(mu1)) < 0.999:
        cc.append(ci)
        mm.append(mu1)
        mu1 += 0.05
    cc = np.array(cc)
    mm = np.array(mm)
    I = cc.argsort()
    cc = cc[I]
    mm = mm[I]
    return si.interp1d(cc, mm)(np.linspace(min(cc), max(cc), samples))


@as_function_node
def CalcPhaseDiagram(
        phases: list,
        temperatures: list[float] | np.ndarray,
        chemical_potentials: list[float] | np.ndarray | int = 100,
        refine: bool = True
):
    """Calculate thermodynamic potentials and respective stable phases in a range of temperatures.

    The chemical potential range is chosen automatically to cover the full concentration space.

    Args:
        phases: list of phases to consider
        temperatures: temperature samples
        mu_samples: number of samples in chemical potential space
        refine (bool): add additional sampling points along exact phase transitions

    Returns:
        dataframe with phase data
    """
    import matplotlib.pyplot as plt
    import landau

    if isinstance(chemical_potentials, int):
        mus = guess_mu_range(phases, max(temperatures), chemical_potentials)
    else:
        mus = chemical_potentials
    phase_data = landau.calculate.calc_phase_diagram(
            phases, np.asarray(temperatures), mus,
            refine=refine, keep_unstable=False
    )
    return phase_data


@as_function_node
def PlotConcPhaseDiagram(
        phase_data,
        plot_samples: bool = False,
        plot_isolines: bool = False,
        plot_tielines: bool = True,
        linephase_width: float = 0.01,
        concavity: float | None = None,
):
    """Plot a concentration-temperature phase diagram.

    phase_data should originate from CalcPhaseDiagram.

    Args:
        phases: list of phases to consider
        plot_samples (bool): overlay points where phase data has been sampled
        plot_isolines (bool): overlay lines of constance chemical potential
        plot_tielines (bool): add grey lines connecting triple points
        linephase_width (float): phases that have a solubility less than this
            will be plotted as a rectangle
        concavity (float, optional, range in [0, 1]): how aggressive to be when
            fitting polyhedra to samples phase data; lower means more ragged
            shapes, higher means smoother; 1 corresponds to convex hull of points
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import landau
    
    fig, ax = plt.subplots()
    landau.plot.plot_phase_diagram(
            df=phase_data.drop('refined', errors='ignore', axis='columns'),
            min_c_width=linephase_width,
            alpha=concavity or 0.1,
    )
    if plot_samples:
        sns.scatterplot(
            data=phase_data,
            x='c', y='T',
            hue='phase',
            legend=False,
            s=1
        )
    if plot_isolines:
        sns.lineplot(
            data=phase_data.loc[np.isfinite(phase_data.mu)],
            x='c', y='T',
            hue='mu',
            units='phase', estimator=None,
            legend=False,
            sort=False,
        )
    if plot_tielines and 'refined' in phase_data.columns:
        # hasn't made it upstream yet
        for T, dd in phase_data.query('refined=="delaunay-triple"').groupby('T'):
            plt.plot(dd.c, [T]*3, c='k', alpha=.5, zorder=-10)
    plt.xlabel("Concentration")
    plt.ylabel("Temperature [K]")
    return fig


@as_function_node
def PlotMuPhaseDiagram(phase_data):
    """Plot a chemical potential-temperature phase diagram.

    phase_data should originate from CalcPhaseDiagram.
    Phase boundaries are plotted in black.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    border = None
    if 'border' not in phase_data.columns:
        body = phase_data.query('not border')
    else:
        border = phase_data.query('border')
        body = phase_data.query('not border')
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=body,
        x='mu', y='T',
        hue='phase',
        s=5,
        ax=ax
    )
    if border is not None:
        sns.scatterplot(
            data=border,
            x='mu', y='T',
            c='k',
            s=5,
            ax=ax
        )
    ax.set_xlabel("Chemical Potential Difference [eV]")
    ax.set_ylabel("Temperature [K]")
    return fig


@as_function_node
def PhasesFromDataFrame(
        dataframe,
        temperature_parameters: int | None = 4,
        concentration_parameters: int | None = 1,
):
    """Convert a dataframe of free energies to list of phase objects.

    Prints the names of all found phases.

    Args:
        dataframe: should contain columns
                    `phase`: the name of the phase; rows of the same phase name
                             will be grouped in a solution phase
                    `composition`: the mole fraction of one of the constitutents
                    `temperature`: array of temperature at which free energy
                                   was sampled
                    `free_energy`: corresponding free energies
        temperature_parameters (int): how many parameters to use when
                    interpolating free energies in temperature
        concentration_parameters (int, optional): how many parameters to use
                    when interpolating free energies in concentration; if not
                    given output individual phases and change the name to
                    include the concentration

    Returns:
        list of Phase objects
        dict of Phase objects, where the dict keys are the names of the phases
    """
    phases = dataframe.groupby('phase')[dataframe.columns].apply(
            make_phase, include_groups=False,
            temperature_parameters=temperature_parameters,
            concentration_parameters=concentration_parameters,
    )
    phase_dict = {p.name: p for p in phases.explode()}
    print("Found phases:", *phase_dict.keys(), sep='\n')
    phase_list = list(phase_dict.values())
    return phase_list, phase_dict


def make_phase(dd, temperature_parameters, concentration_parameters):
    name = dd.phase.iloc[0]
    # minus 2 for terminals
    # minus 1 to be not exactly interpolating
    sub = [landau.phases.TemperatureDependentLinePhase(
                f'{row.phase}_{c:.03}', c, 
                row.temperature, row.free_energy, 
                interpolator=landau.interpolate.SGTE(temperature_parameters)
            ) for c, row in dd.set_index('composition').iterrows()]
    # only a single concentration
    if len(sub) == 1:
        return replace(sub[0], name=name)
    if concentration_parameters is not None:
        interp_params = min(len(dd)-2-1, concentration_parameters)
        # terminals are present
        if len({0, 1}.intersection([s.line_concentration for s in sub]))==2:
            if len(sub) == 2: # only terminals are present
                return landau.phases.IdealSolution(name, *sub)
            else:
                return landau.phases.RegularSolution(name, sub, interp_params)
        else:
            return landau.phases.InterpolatingPhase(name, sub, interp_params, num_samples=1000)
    else:
        return sub
