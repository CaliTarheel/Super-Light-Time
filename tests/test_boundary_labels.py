"""Map and statistics labels follow the running mechanism, not the kinematic gate."""
from types import SimpleNamespace
import unittest

import numpy as np

import boundary_labels


def world():
    """Six contacts, one per case, with the kinematic gate already applied.

    0 slow head-on subduction, ocean going down  (gate: transform)
    1 fast continental collision                 (gate: collision)
    2 slow continental convergence               (gate: transform)
    3 spreading ocean ridge                      (gate: ridge)
    4 pure sliding                               (gate: transform)
    5 a zero-length graph edge                   (gate: none)
    """
    cells = 12
    crust = np.zeros(cells, np.uint8)
    crust[[2, 3, 4, 5]] = 1                      # continental on both sides of contacts 1 and 2
    ba = np.array([0, 2, 4, 6, 8, 10])
    bb = np.array([1, 3, 5, 7, 9, 11])
    return SimpleNamespace(
        ba=ba, bb=bb, bp=np.arange(6), bq=np.arange(6)+6,
        bl=np.array([100., 100., 100., 100., 100., 0.]),
        bcode=np.array([3, 4, 3, 1, 3, 0], np.uint8),
        down=np.array([0, -1, -1, -1, -1, -1]),
        normal_speed=np.array([-1.5, -20., -1.2, 30., 0., 0.]),
        crust=crust, normal_partition_version=1, boundary_label_version=boundary_labels.VERSION,
        t=120.)


class BoundaryLabelTests(unittest.TestCase):
    def test_mechanism_labels_replace_the_kinematic_gate(self):
        s = world()
        display = boundary_labels.codes(s)
        self.assertEqual(list(display), [boundary_labels.SUBDUCTION, boundary_labels.COLLISION,
                                         boundary_labels.COLLISION, boundary_labels.RIDGE,
                                         boundary_labels.TRANSFORM, boundary_labels.NONE])
        # The gate itself is untouched: no process changes.
        self.assertEqual(list(s.bcode), [3, 4, 3, 1, 3, 0])

    def test_legacy_worlds_keep_their_recorded_labels(self):
        s = world()
        s.boundary_label_version = 0
        np.testing.assert_array_equal(boundary_labels.codes(s), s.bcode)
        for bad in (True, 2, 'yes'):
            s.boundary_label_version = bad
            with self.assertRaises(ValueError):
                boundary_labels.codes(s)

    def test_gated_off_labels_read_only_the_kinematic_code(self):
        # native_spreading.boundary_segments calls codes() on states that carry
        # bcode but no boundary lengths. With labels off, nothing else is read.
        bcode = np.array([3, 4, 3, 1, 3, 0], np.uint8)
        s = SimpleNamespace(bcode=bcode)
        display = boundary_labels.codes(s)
        np.testing.assert_array_equal(display, bcode)
        self.assertIsNot(display, bcode)

    def test_upgrade_selects_labels_without_touching_physics(self):
        s = world()
        s.boundary_label_version = 0
        report = boundary_labels.upgrade(s)
        self.assertTrue(boundary_labels.enabled(s))
        self.assertEqual((report['physical_state_changed'], report['kinematic_code_retained']), (False, True))
        self.assertEqual(report['time_myr'], 120.)
        self.assertEqual(boundary_labels.upgrade(s), report)   # idempotent

    def test_subduction_needs_both_a_downgoing_plate_and_water(self):
        s = world()
        s.crust[[0, 1]] = 1                      # the incoming side is now continental
        self.assertEqual(boundary_labels.codes(s)[0], boundary_labels.COLLISION)
        s.crust[[0, 1]] = 0
        s.down = np.full(6, -1)                  # no polarity resolved anywhere
        self.assertEqual(boundary_labels.codes(s)[0], boundary_labels.TRANSFORM)

    def test_a_jammed_margin_is_collision_even_with_ocean_above_it(self):
        # physics_profile.stall_buoyant_incoming blocks a trench when a buoyant
        # plate arrives, and clears s.down. In run SEP21T at 156 Myr, 58 such
        # contacts (3,895 km) carried ocean on the upper plate, so neither
        # polarity nor "continental on both sides" identifies them. They are
        # jammed convergence, not sliding.
        s = world()
        s.crust[[0, 1]] = np.array([1, 0], np.uint8)   # continent arriving, ocean above
        s.down = np.full(6, -1)                        # the stall cleared the polarity
        s.bcode[0] = 4
        self.assertEqual(boundary_labels.codes(s)[0], boundary_labels.COLLISION)
        # A closing ocean-ocean contact with no resolved polarity is not a
        # collision: nothing buoyant is involved and no trench is running.
        s.crust[[0, 1]] = 0
        self.assertEqual(boundary_labels.codes(s)[0], boundary_labels.TRANSFORM)

    def test_opening_contacts_are_never_relabelled_convergent(self):
        s = world()
        s.normal_speed = np.array([30., 30., 30., 30., 30., 0.])
        s.bcode = np.array([1, 5, 1, 1, 3, 0], np.uint8)
        display = boundary_labels.codes(s)
        self.assertEqual(list(display[:4]), [boundary_labels.RIDGE, boundary_labels.RIFT,
                                             boundary_labels.RIDGE, boundary_labels.RIDGE])

    def test_lengths_and_disagreement_report_what_changed(self):
        s = world()
        lengths = boundary_labels.lengths(s)
        self.assertEqual(lengths['subduction'], 100.)
        self.assertEqual(lengths['collision'], 200.)
        self.assertEqual(lengths['transform'], 100.)
        self.assertEqual(lengths['ridge'], 100.)
        self.assertEqual(sum(lengths.values()), float(s.bl.sum()))
        report = boundary_labels.disagreement(s)
        self.assertEqual(report['relabelled_contacts'], 2)      # contacts 0 and 2
        self.assertEqual(report['relabelled_length_km'], 200.)
        self.assertEqual(report['by_label']['subduction']['contacts'], 1)
        self.assertEqual(report['by_label']['collision']['contacts'], 1)
        self.assertEqual(report['by_label']['collision']['from_kinematic'], {'3': 1})
        s.boundary_label_version = 0
        self.assertEqual(boundary_labels.disagreement(s)['relabelled_contacts'], 0)

    def test_zero_length_contacts_carry_no_label(self):
        s = world()
        s.bl = np.zeros(6)
        np.testing.assert_array_equal(boundary_labels.codes(s), 0)
        self.assertEqual(sum(boundary_labels.lengths(s).values()), 0.)


if __name__ == '__main__':
    unittest.main()
