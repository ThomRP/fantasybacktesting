"""Unit tests for pre-season and mid-season injury availability caps in draft_engine."""

from datetime import date
import unittest

import draft_engine as de


class TestEvaluateInjuryAvailabilityCap(unittest.TestCase):
    def setUp(self):
        self.target_year = 2026
        self.opening_night = date(2026, 10, 21)

    def test_preseason_anchors_to_opening_night(self):
        """Before Opening Night, missed games are counted from Opening Night on a full 82g / 65g scale."""
        inj = {
            "status": "Out",
            "injury_type": "Calf",
            "injury_detail": "Strain",
            "comment": "Expected to return late November.",
            "return_date": date(2026, 11, 24),
        }
        cap, missed, label = de.evaluate_injury_availability_cap(
            inj, self.target_year, opening_night=self.opening_night, as_of_date=date(2026, 9, 15),
        )
        # 34 days past Oct 21 -> round(34 / 175 * 82) = 16 games missed
        expected_cap = min(1.0, ((82.0 - 16) * 0.95) / 65.0)
        self.assertEqual(missed, 16)
        self.assertAlmostEqual(cap, expected_cap, places=6)
        self.assertEqual(label, "INJ_CAP(ret 2026-11-24, ~16g miss)")

    def test_preseason_before_opening_night_return_not_capped(self):
        """Pre-season minor injury returning before or right at Opening Night is not capped."""
        inj = {
            "status": "Day-To-Day",
            "injury_type": "Ankle",
            "injury_detail": "Sprain",
            "comment": "Expected ready for Opening Night.",
            "return_date": date(2026, 10, 22),
        }
        cap, missed, label = de.evaluate_injury_availability_cap(
            inj, self.target_year, opening_night=self.opening_night, as_of_date=date(2026, 10, 1),
        )
        self.assertIsNone(cap)
        self.assertEqual(missed, 0)
        self.assertEqual(label, "INJ_NOTE(Day-To-Day)")

    def test_midseason_returning_tomorrow_not_penalized_for_october(self):
        """Mid-season (Nov 23), a player returning tomorrow (Nov 24) must NOT be penalized for October games."""
        inj = {
            "status": "Out",
            "injury_type": "Rest",
            "injury_detail": "",
            "comment": "Ruled out for Sunday's game for rest.",
            "return_date": date(2026, 11, 24),
        }
        cap, missed, label = de.evaluate_injury_availability_cap(
            inj, self.target_year, opening_night=self.opening_night, as_of_date=date(2026, 11, 23),
        )
        self.assertIsNone(cap)
        self.assertEqual(missed, 0)
        self.assertEqual(label, "INJ_NOTE(Out)")

    def test_midseason_returning_today_or_past_not_capped(self):
        """Mid-season return date on or before today produces no availability cap."""
        for ret in (date(2026, 11, 20), date(2026, 11, 23)):
            inj = {
                "status": "Day-To-Day",
                "injury_type": "Knee",
                "injury_detail": "",
                "comment": "Probable for tonight.",
                "return_date": ret,
            }
            cap, missed, label = de.evaluate_injury_availability_cap(
                inj, self.target_year, opening_night=self.opening_night, as_of_date=date(2026, 11, 23),
            )
            self.assertIsNone(cap)
            self.assertEqual(missed, 0)
            self.assertEqual(label, "INJ_NOTE(Day-To-Day)")

    def test_midseason_extended_absence_scales_remaining_season(self):
        """Mid-season extended absence counts only forward missed games against scaled remaining games."""
        as_of = date(2026, 11, 23)
        ret = date(2027, 1, 2)  # 40 days from Nov 23 -> ~19 games missed
        inj = {
            "status": "Out",
            "injury_type": "Knee",
            "injury_detail": "Surgery",
            "comment": "Rehabbing from knee surgery, targeting January return.",
            "return_date": ret,
        }
        cap, missed, label = de.evaluate_injury_availability_cap(
            inj, self.target_year, opening_night=self.opening_night, as_of_date=as_of,
        )
        rem_frac = (de.SEASON_DAYS - (as_of - self.opening_night).days) / de.SEASON_DAYS
        rem_season_games = de.SEASON_GAMES * rem_frac
        rem_ref_games = de.REF_GAMES * rem_frac
        expected_missed = int(round((40 / de.SEASON_DAYS) * de.SEASON_GAMES))
        expected_cap = min(1.0, ((rem_season_games - expected_missed) * 0.85) / rem_ref_games)

        self.assertEqual(missed, expected_missed)
        self.assertAlmostEqual(cap, expected_cap, places=6)
        self.assertEqual(label, "INJ_CAP(ret 2027-01-02, ~19g miss)")

        # Same 19 missed games mid-season is a larger share of remaining season than pre-season
        pre_cap, pre_missed, _ = de.evaluate_injury_availability_cap(
            {**inj, "return_date": date(2026, 11, 30)},  # 40 days after Oct 21
            self.target_year,
            opening_night=self.opening_night,
            as_of_date=date(2026, 9, 15),
        )
        self.assertEqual(pre_missed, expected_missed)
        self.assertLess(cap, pre_cap)

    def test_season_ending_injury_preseason_and_midseason(self):
        """Return date on/after March 15 is always OUT_SEASON with avail_cap = 0.0."""
        inj = {
            "status": "Out",
            "injury_type": "Knee",
            "injury_detail": "ACL",
            "comment": "Torn ACL, out for season.",
            "return_date": date(2027, 3, 17),
        }
        pre_cap, pre_missed, pre_label = de.evaluate_injury_availability_cap(
            inj, self.target_year, opening_night=self.opening_night, as_of_date=date(2026, 9, 15),
        )
        mid_cap, mid_missed, mid_label = de.evaluate_injury_availability_cap(
            inj, self.target_year, opening_night=self.opening_night, as_of_date=date(2026, 11, 23),
        )
        self.assertEqual(pre_cap, 0.0)
        self.assertEqual(pre_missed, 82)
        self.assertEqual(pre_label, "OUT_SEASON(Out ret 2027-03-17)")
        self.assertEqual(mid_cap, 0.0)
        self.assertLess(mid_missed, 82)
        self.assertEqual(mid_label, "OUT_SEASON(Out ret 2027-03-17)")


if __name__ == "__main__":
    unittest.main()
