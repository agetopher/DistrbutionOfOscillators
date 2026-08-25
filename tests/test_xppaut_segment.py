from pathlib import Path
import runpy

import numpy as np

import segment
import settings


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "xppaut" / "generate_single_segment.py"
ODE_FILE = ROOT / "xppaut" / "single_segment.ode"


def _generator_namespace():
    return runpy.run_path(str(GENERATOR), run_name="xppaut_generator")


def test_committed_xppaut_model_is_current():
    namespace = _generator_namespace()
    assert ODE_FILE.read_text() == namespace["build_ode"]()


def test_xppaut_drive_mapping_matches_segment_model():
    namespace = _generator_namespace()
    drive = namespace["_drive"]

    for index in range(segment.N_CELLS):
        iava = 2.0 if "iava" in drive(index) else 0.0
        iavb = 3.0 if "iavb" in drive(index) else 0.0
        expected = segment.drive_vector(IAVA=2.0, IAVB=3.0)[index]
        assert iava + iavb == expected


def test_xppaut_expanded_connectivity_matches_segment_matrices():
    namespace = _generator_namespace()
    weighted_sum = namespace["_weighted_sum"]
    gap_sum = namespace["_gap_sum"]

    for post in range(segment.N_CELLS):
        assert weighted_sum(segment.E_CONN[post], "s") in ODE_FILE.read_text()
        assert weighted_sum(segment.I_CONN[post], "s") in ODE_FILE.read_text()
        assert gap_sum(segment.GJ_CONN[post], post) in ODE_FILE.read_text()


def test_expanded_xppaut_equations_match_python_rhs_numerically():
    """Check the explicit XPPAUT current expansion against ``segment.rhs_vw``."""
    settings.reset_defaults()
    segment.configure()
    rng = np.random.default_rng(731)
    voltage = rng.uniform(-78.0, -28.0, segment.N_CELLS)
    recovery = rng.uniform(0.0, 12.0, segment.N_CELLS)
    iava, iavb = 1.7, 2.3

    gate = 1.0 / (
        1.0 + np.exp(-settings.k_syn * (voltage - settings.V_th))
    )
    excitatory_input = segment.E_CONN @ gate
    inhibitory_input = segment.I_CONN @ gate
    gap_input = np.array([
        np.sum(
            segment.GJ_CONN[post]
            * (voltage[post] - voltage)
        )
        for post in range(segment.N_CELLS)
    ])
    drive = segment.drive_vector(IAVA=iava, IAVB=iavb)

    membrane = np.select(
        [
            voltage > settings.H,
            voltage >= settings.T,
            voltage >= settings.L,
        ],
        [
            settings.m4 * (-voltage + settings.H),
            settings.m3 * (voltage - settings.H) * (voltage - settings.T),
            settings.m2 * (voltage - settings.L) * (voltage - settings.T),
        ],
        default=settings.m1 * (-voltage + settings.L),
    )
    xpp_voltage_dot = (
        settings.g * membrane
        - recovery
        - settings.G_syne
        * excitatory_input
        * (voltage - settings.E_syne)
        - settings.G_syni
        * inhibitory_input
        * (voltage - settings.E_syni)
        - settings.G_gap * gap_input
        + drive
    ) / settings.C
    xpp_recovery_dot = (
        settings.beta * np.maximum(voltage - settings.T, 0.0) - recovery
    ) / settings.tau_w

    python_rhs = segment.rhs_vw(
        np.concatenate([voltage, recovery]),
        drive,
    )
    np.testing.assert_allclose(
        np.concatenate([xpp_voltage_dot, xpp_recovery_dot]),
        python_rhs,
        rtol=1e-13,
        atol=1e-13,
    )


def test_xppaut_defaults_match_python_settings():
    settings.reset_defaults()
    source = ODE_FILE.read_text()

    assert f"par iava=0,iavb=0" in source
    assert f"gse={settings.G_syne:g}" in source
    assert f"gsi={settings.G_syni:g}" in source
    assert f"ggap={settings.G_gap:g}" in source
    assert f"ksyn={settings.k_syn:g}" in source
    assert f"beta={settings.beta:g}" in source
    assert f"tauw={settings.tau_w:g}" in source
    assert source.count("'=") == 2 * segment.N_CELLS
