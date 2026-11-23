"""Unit tests for the ADP-aware Reach-vs-Wait survival-probability draft optimizer."""

import unittest

import draft_engine as de


class TestReachOrWaitOptimizer(unittest.TestCase):
    def _make_pool(self, specs):
        """Build synthetic player arrays from list of (value, adp, pos_tuple)."""
        n = len(specs)
        vals = [s[0] for s in specs]
        adps = [s[1] for s in specs]
        masks7 = [de._pos_to_masks(s[2])[0] for s in specs]
        is_c = [s[2][0] == "C" for s in specs]
        hero_board = sorted(range(n), key=lambda i: (-vals[i], adps[i]))
        return hero_board, vals, adps, masks7, is_c

    def test_waits_when_top_target_will_survive_and_second_will_be_taken(self):
        """If #1 target (ADP 119.5) easily survives K=4 picks while #2 (ADP 106) won't, pick #2 now."""
        specs = [
            (27.28, 119.5, ("PG", "SG")),  # Target A: will easily survive to pick 110
            (26.92, 106.0, ("SF", "PF")),  # Target B: will be drafted in next 4 picks
            (26.10, 107.0, ("PG", "SG")),
            (26.00, 108.0, ("SF", "PF")),
            (25.90, 109.0, ("SG", "SF")),
            (25.80, 110.5, ("PF", "C")),
            (25.70, 111.5, ("SG", "SF")),
            (25.50, 120.0, ("PG", "SG")),
        ]
        board, vals, adps, masks7, is_c = self._make_pool(specs)
        choice, _ = de.choose_reach_or_wait_pick(
            board, vals, adps, taken=[False] * len(specs), roster=[],
            cur_c=0, cur_m=6, rem_after=5, masks7=masks7, is_c=is_c,
            max_centers=4, picks_until_next=4, surv_sigma=2.5, round_num=9, surv_margin=1.0,
        )
        self.assertEqual(choice, 1)  # Takes B now, waits to scoop A at next pick

    def test_reaches_when_top_target_will_not_survive_long_turn(self):
        """When K=6 opponent picks push cutoff past Target A's ADP, reach for Target A immediately."""
        specs = [
            (27.80, 110.0, ("PG", "SG")),  # Target A: gone before next pick!
            (26.90, 105.0, ("SF", "PF")),  # Target B: also gone
            (26.10, 106.0, ("PG", "SG")),
            (26.00, 107.0, ("SF", "PF")),
            (25.90, 108.0, ("SG", "SF")),
            (25.80, 109.0, ("PF", "C")),
            (25.70, 112.0, ("PG", "SG")),
            (25.50, 115.0, ("PG", "SG")),
        ]
        board, vals, adps, masks7, is_c = self._make_pool(specs)
        choice, _ = de.choose_reach_or_wait_pick(
            board, vals, adps, taken=[False] * len(specs), roster=[],
            cur_c=0, cur_m=6, rem_after=5, masks7=masks7, is_c=is_c,
            max_centers=4, picks_until_next=6, surv_sigma=2.5, round_num=9, surv_margin=1.0,
        )
        self.assertEqual(choice, 0)  # Reaches for Target A now

    def test_final_round_and_back_to_back_turn_are_greedy(self):
        """With K=None (final round) or K=0 (turn pick), always take the highest-value legal player."""
        specs = [
            (27.28, 130.0, ("PG", "SG")),
            (26.92, 105.0, ("SF", "PF")),
            (25.50, 106.0, ("PF", "C")),
        ]
        board, vals, adps, masks7, is_c = self._make_pool(specs)
        for k_opp in (None, 0):
            choice, _ = de.choose_reach_or_wait_pick(
                board, vals, adps, taken=[False] * len(specs), roster=[],
                cur_c=0, cur_m=0, rem_after=0 if k_opp is None else 5,
                masks7=masks7, is_c=is_c, max_centers=4,
                picks_until_next=k_opp, surv_sigma=2.5, round_num=13 if k_opp is None else 1,
            )
            self.assertEqual(choice, 0)

    def test_respects_four_center_cap(self):
        """Never drafts a 5th primary Center, and accounts for filling 4th C in next-pick EV."""
        specs = [
            (30.0, 120.0, ("C",)),         # Primary C (illegal when cur_c == 4)
            (28.0, 110.0, ("PG", "SG")),   # Best legal non-C
            (26.0, 111.0, ("SF", "PF")),
        ]
        board, vals, adps, masks7, is_c = self._make_pool(specs)
        choice, _ = de.choose_reach_or_wait_pick(
            board, vals, adps, taken=[False] * len(specs), roster=[],
            cur_c=4, cur_m=4, rem_after=3, masks7=masks7, is_c=is_c,
            max_centers=4, picks_until_next=1, surv_sigma=2.5, round_num=10,
        )
        self.assertEqual(choice, 1)


if __name__ == "__main__":
    unittest.main()
