"""Test whether players who surge after the All-Star break beat next season's ADP."""

from backtest_common import Backtest, run_backtest as run, run_cli

SURGE_THRESHOLD = 0.10
SURGE = Backtest(
    column="surged",
    label=f"Surged ({SURGE_THRESHOLD:.0%} or more)",
    output_stem="backtest_results",
    matches=lambda change: change >= SURGE_THRESHOLD,
)


def run_backtest(data, ages=None):
    run(data, SURGE, ages)


def main():
    run_cli(__doc__, run_backtest)


if __name__ == "__main__":
    main()
