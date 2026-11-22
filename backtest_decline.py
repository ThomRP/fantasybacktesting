"""Test whether players who decline after the All-Star break beat next season's ADP."""

from backtest_common import Backtest, run_backtest as run, run_cli

DECLINE_THRESHOLD = 0.10
DECLINE = Backtest(
    column="declined",
    label=f"Declined ({DECLINE_THRESHOLD:.0%} or more)",
    output_stem="backtest_decline_results",
    matches=lambda change: change <= -DECLINE_THRESHOLD,
)


def run_backtest(data, ages=None):
    run(data, DECLINE, ages)


def main():
    run_cli(__doc__, run_backtest)


if __name__ == "__main__":
    main()
