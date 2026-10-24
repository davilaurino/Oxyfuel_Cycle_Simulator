"""Coupling, actual power accounting, initialization, and runner checks."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.constants import R

from ALLAM_v2.allam_setup import build_cycle
from ASU.asu_setup_lox import build_asu
from INTEGRATED import integrated_config as config
from INTEGRATED import integrated_main as runner
from INTEGRATED.integrated_reporting import integrated_report, results_table
from INTEGRATED.integrated_setup import (
    build_integrated_cycle,
    load_solution,
    standalone_initial_guess,
)
from common import utils
from common.residuals import ResidualScaling, evaluate_residuals
from common.solver import build_bounds, build_jacobian_sparsity


class IntegratedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states, cls.components, cls.x0 = build_integrated_cycle()
        cls.scaling = ResidualScaling.from_config(config)
        cls.residuals = evaluate_residuals(
            cls.x0, cls.states, cls.components, cls.scaling,
        )

    def test_coupling_is_square_and_shares_one_oxygen_stream(self):
        self.assertEqual(len(self.residuals), len(self.x0))
        self.assertTrue(np.all(np.isfinite(self.residuals)))
        self.assertIs(
            self.components['Oxidant mixer'].inlets[1], self.states['O2 product'],
        )
        self.assertNotIn('O2 Inlet', self.components)
        self.assertNotIn('O2 - Inlet Stream', self.states)
        pattern = build_jacobian_sparsity(
            self.states, self.components, self.scaling,
        )
        self.assertEqual(pattern.shape, (len(self.x0), len(self.x0)))
        self.assertTrue(np.all(pattern.any(axis=0)))

    def test_mac_flow_and_oxygen_flow_are_not_fixed_boundaries(self):
        boundary = self.components['MAC air intake - Input']
        self.assertIsNone(boundary.m_dot)
        self.assertIsNone(self.components['O2 Specification'].m_dot)
        before = boundary.residuals(self.scaling)
        original = boundary.outlet.m_dot
        try:
            boundary.outlet.m_dot *= 1.5
            np.testing.assert_array_equal(before, boundary.residuals(self.scaling))
        finally:
            boundary.outlet.m_dot = original
        self.assertEqual(len(self.components['O2 Specification'].residuals(self.scaling)), 1)

    def test_bounds_cover_both_subsystems(self):
        lower, upper = build_bounds(self.states, config)
        self.assertTrue(np.all(self.x0 >= lower))
        self.assertTrue(np.all(self.x0 <= upper))
        self.assertLess(self.states['LPC O2 product'].T, 180.0)
        self.assertGreater(self.states['Combustion Products'].T, 500.0)

    def test_report_accounts_for_modeled_asu_once_and_uses_pure_oxygen(self):
        with redirect_stdout(StringIO()) as output:
            report = integrated_report(self.states, self.components)
        asu_consumed = -sum(
            c.W for c in self.components.values()
            if c.subsystem == 'ASU' and getattr(c, 'W', 0.0) < 0.0
        )
        asu_generated = self.components['MP air turbine'].W
        net_asu = asu_consumed - asu_generated
        net_plant = sum(getattr(c, 'W', 0.0) for c in self.components.values())
        oxygen = self.states['O2 product']
        pure_flow = oxygen.m_dot*oxygen.w[oxygen.species.index(utils.O2)]
        self.assertAlmostEqual(report['asu_work_W'], net_asu)
        self.assertAlmostEqual(report['net_work_W'], net_plant)
        self.assertAlmostEqual(report['asu_specific_work_J_kg'], net_asu/pure_flow)
        self.assertAlmostEqual(
            report['asu_specific_work_kWh_Nm3_O2'],
            net_asu/(pure_flow/utils.O2.MW*R*273.15/101325.0)/3.6e6,
        )
        boundary_balance = (
            self.states['Fuel Compressor - Inlet'].m_dot
            + self.states['MAC air intake'].m_dot
            - self.states['N2 product'].m_dot
            - self.states['Condenser Liquid Outlet'].m_dot
            - self.states['Purged CO2'].m_dot
        )
        self.assertAlmostEqual(report['mass_balance_kg_s'], boundary_balance)
        for name in ('MAC - Compressor 1', 'LOX pump', 'MP air turbine', 'HP Turbine'):
            self.assertIn(name, output.getvalue())
        table = results_table(self.states)
        self.assertEqual(len(table), len(self.states))
        self.assertEqual(table.loc['LPC O2 product', 'P (MPa)'], 0.13)
        self.assertIn('z_CH4', table.columns)
        self.assertIn('z_AR', table.columns)

    def test_standalone_mapping_scales_asu_without_copying_allam_oxygen(self):
        asu_states, _, asu_x = build_asu()
        allam_states, _, allam_x = build_cycle()
        with tempfile.TemporaryDirectory() as directory:
            asu_path = Path(directory) / 'asu.npy'
            allam_path = Path(directory) / 'allam.npy'
            np.save(asu_path, asu_x)
            np.save(allam_path, allam_x)
            initial = standalone_initial_guess(self.states, asu_path, allam_path)
        offset = 0
        mapped = {}
        for name, state in self.states.items():
            width = len(state.flatten_vars())
            mapped[name] = initial[offset:offset + width]
            offset += width
        ratio = allam_states['O2 - Inlet Stream'].m_dot/asu_states['O2 product'].m_dot
        for name, state in asu_states.items():
            self.assertAlmostEqual(mapped[name][0], state.m_dot*ratio)
            np.testing.assert_array_equal(mapped[name][1:], state.flatten_vars()[1:])
        for name, state in allam_states.items():
            if name != 'O2 - Inlet Stream':
                np.testing.assert_array_equal(mapped[name], state.flatten_vars())
        self.assertNotEqual(
            mapped['O2 product'][4], allam_states['O2 - Inlet Stream'].z[1],
        )

    def test_invalid_saved_solutions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'solution.npy'
            for values in (np.zeros((2, 2)), np.array([0.0, np.nan]), np.array(['a', 'b'])):
                np.save(path, values)
                with self.subTest(values=values), self.assertRaises(ValueError):
                    load_solution(path, 2)

    def test_runner_rejects_false_convergence(self):
        with (
            patch.object(runner, 'build_integrated_cycle', return_value=(self.states, self.components, self.x0)),
            patch.object(runner, 'solve', return_value=(self.x0, True)),
            patch.object(runner, 'evaluate_residuals', return_value=np.array([0.01])),
            patch.object(runner, 'integrated_report') as report,
            patch('builtins.input', return_value='s'),
            redirect_stdout(StringIO()),
        ):
            runner.main()
        report.assert_not_called()

    def test_runner_saves_and_reloads_integrated_files(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, 'INTEGRATED_DIR', Path(directory)),
            patch.object(runner, 'build_integrated_cycle', return_value=(self.states, self.components, self.x0)),
            patch.object(runner, 'standalone_initial_guess', return_value=self.x0),
            patch.object(runner, 'solve', return_value=(self.x0, True)) as solve,
            patch.object(runner, 'evaluate_residuals', return_value=np.zeros_like(self.x0)),
            patch.object(runner, 'integrated_report'),
            redirect_stdout(StringIO()),
        ):
            with patch('builtins.input', side_effect=['b', 'y']):
                runner.main()
            np.testing.assert_array_equal(np.load(Path(directory) / '0_solution.npy'), self.x0)
            self.assertTrue((Path(directory) / '0_results.csv').is_file())
            self.assertEqual(len(list(Path(directory).glob('0_results_*.csv'))), 1)
            with patch('builtins.input', side_effect=['p', 'n']):
                runner.main()
            np.testing.assert_array_equal(solve.call_args.args[2], self.x0)


if __name__ == '__main__':
    unittest.main()
