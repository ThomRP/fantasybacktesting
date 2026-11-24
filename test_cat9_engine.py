"""Unit tests for the Unified 9-Cat Fantasy Basketball Draft Engine (`cat9_engine.py`)."""

from datetime import date
import unittest

import numpy as np

import cat9_engine as c9
import draft_engine as de


class TestCat9MathAndScoring(unittest.TestCase):
    def test_z_slot_value_preserves_negative_z_scores(self):
        """Signed interpolation must not clamp negative Z-scores at 0.0."""
        lb = [8.0, 4.0, 1.0, -1.0, -3.0]
        self.assertAlmostEqual(c9.z_slot_value(lb, 1.0), 8.0)
        self.assertAlmostEqual(c9.z_slot_value(lb, 3.5), 0.0)
        self.assertAlmostEqual(c9.z_slot_value(lb, 4.5), -2.0)
        self.assertAlmostEqual(c9.z_slot_value(lb, 5.0), -3.0)
        self.assertAlmostEqual(c9.z_slot_value(lb, 99.0), -3.0)

    def test_volume_weighted_percentage_impact(self):
        """High-volume efficient shooter outscores low-volume shooter with identical raw FG%."""
        # League FG% = 0.50, FT% = 0.80
        high_vol = np.array([20, 5, 5, 1, 1, 2, 12, 20, 9, 10, 2], dtype=float)  # 60% FG on 20 FGA (+2.0), 90% FT on 10 FTA (+1.0)
        low_vol = np.array([6, 2, 1, 0, 0, 0, 3, 5, 0, 0, 1], dtype=float)      # 60% FG on 5 FGA (+0.5), 0 FTA (0.0)
        imp_hi = c9.stats11_to_cat9_impact(high_vol, 0.50, 0.80)
        imp_lo = c9.stats11_to_cat9_impact(low_vol, 0.50, 0.80)
        self.assertAlmostEqual(imp_hi[6], 2.0)
        self.assertAlmostEqual(imp_lo[6], 0.5)
        self.assertAlmostEqual(imp_hi[7], 1.0)
        self.assertAlmostEqual(imp_lo[7], 0.0)
        self.assertEqual(imp_hi[8], -2.0)  # inverted turnovers

    def test_three_format_scoring_invariants_and_turnover_direction(self):
        """Verify Roto total sum = 702 (12 teams), H2H zero-sum 50% mean, and fewer TOs wins TO cat."""
        rng = np.random.default_rng(7)
        n_teams, n_wks = 12, 10
        wk11 = rng.uniform(5.0, 50.0, size=(n_teams, n_wks, 11))
        # Ensure FGM <= FGA and FTM <= FTA
        wk11[:, :, 6] = np.minimum(wk11[:, :, 6], wk11[:, :, 7])
        wk11[:, :, 8] = np.minimum(wk11[:, :, 8], wk11[:, :, 9])
        tot11 = wk11.sum(axis=1)
        valid_wks = np.arange(n_wks)

        roto, h2he, h2hm = c9.score_cat9_league(tot11, wk11, valid_wks)
        # 12 teams -> sum(1..12) = 78 per cat * 9 cats = 702
        self.assertAlmostEqual(float(roto.sum()), 702.0, places=6)
        self.assertAlmostEqual(float(h2he.mean()), 0.5, places=6)
        self.assertAlmostEqual(float(h2hm.mean()), 0.5, places=6)

        # Lower turnovers must win the TO category: give Team 0 zero turnovers while all others are identical
        ident_wk = np.full((4, 3, 11), 10.0)
        ident_wk[:, :, 6] = 5.0
        ident_wk[:, :, 8] = 8.0
        ident_wk[0, :, 10] = 0.0  # Team 0 has fewer TOs -> strictly better!
        ident_tot = ident_wk.sum(axis=1)
        r2, e2, m2 = c9.score_cat9_league(ident_tot, ident_wk, np.arange(3))
        self.assertGreater(r2[0], r2[1])
        self.assertGreater(e2[0], e2[1])
        self.assertGreater(m2[0], m2[1])

    def test_daily_active_lineup_benches_lowest_priority_overflow(self):
        """With 11 active players on one day, the 11th lowest-priority player's 11-stat line is benched."""
        n_p, n_days, n_wks = 11, 1, 1
        played_mat = np.ones((n_p, n_days), dtype=np.uint8)
        daily_11 = []
        player_tot11 = np.zeros((n_p, 11), dtype=float)
        player_wk11 = np.zeros((n_p, n_wks, 11), dtype=float)
        positions = [("PG",), ("SG",), ("SF",), ("PF",), ("C",), ("PG",), ("SF",), ("SG",), ("PF",), ("PG",), ("SG",)]
        masks7, masks10 = [], []
        for i in range(n_p):
            m7, m10 = de._pos_to_masks(positions[i])
            masks7.append(m7)
            masks10.append(m10)
            v = np.full(11, float(i + 1))  # player 0 has 1s, player 10 has 11s
            daily_11.append({0: v})
            player_tot11[i] = v
            player_wk11[i, 0] = v
        # Player 0 has lowest priority (-10.0), so Player 0 should be benched!
        prio = [-10.0] + [float(i) for i in range(1, n_p)]
        bundle = {
            "prio": prio,
            "played_mat": played_mat,
            "played_lists": played_mat.tolist(),
            "masks10": masks10,
            "daily_11": daily_11,
            "player_tot11": player_tot11,
            "player_wk11": player_wk11,
            "day_to_week": np.array([0], dtype=np.int32),
            "roster_cache": {},
        }
        tot11, wk11, bg = c9.score_roster_cat9_active(list(range(n_p)), bundle)
        self.assertEqual(bg, 1)
        expected_sum = sum(range(2, 12))  # players 1..10 kept, player 0 (val=1) benched
        self.assertAlmostEqual(tot11[0], float(expected_sum))
        self.assertAlmostEqual(wk11[0, 0], float(expected_sum))

    def test_punt_parser_and_injury_cap_layer(self):
        """Verify --punt parsing and OUT_SEASON injury cap zeroes counting projection."""
        self.assertEqual(c9.parse_punt_cats("ft, TO"), (7, 8))
        w_punt = c9.make_cat_weights(to_weight=0.5, punt_indices=(7,))
        self.assertEqual(w_punt[7], 0.0)
        self.assertEqual(w_punt[8], 0.5)

        rows = [{"athlete_id": "1", "player": "Star Guard", "z_zero_floor": -6.0}]
        proj_tot = np.array([5.0])
        proj_cat = np.array([[1.5, 0.5, 1.5, 0.8, 0.2, 1.2, 0.3, 0.5, -1.0]])
        proj_avail = np.array([0.85])
        inj = [{
            "athlete_id": "1", "player": "Star Guard", "name_keys": {"star guard"},
            "status": "Out", "injury_type": "Knee", "injury_detail": "ACL",
            "return_date": date(2027, 4, 15), "comment": "Out for season",
        }]
        adj_tot, adj_cat, labels, capped = c9.apply_cat9_injury_layer(
            rows, proj_tot, proj_cat, proj_avail, inj, 2026, c9.make_cat_weights(0.5),
            opening_night=date(2026, 10, 21), as_of_date=date(2026, 11, 24),
        )
        self.assertEqual(capped, 1)
        self.assertAlmostEqual(adj_tot[0], -6.0)
        self.assertIn("OUT_SEASON", labels[0])
        self.assertLess(adj_cat[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
