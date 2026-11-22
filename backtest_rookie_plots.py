"""Render FPPG-ranked ADP outcomes for second-year opening-game starters."""

from statistics import median

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator


TEXT = "#172E3C"
MUTED = "#486273"
GRID = "#CFDCE3"
STRIPE = "#F2F6F8"
WIN = "#087D88"
MISS = "#AD4C25"

DETAIL_HEADERS = (
    ("Slots", 0.09), ("ADP to FPPG rank", 0.385), ("Starts / GP", 0.61),
    ("FPPG", 0.76), ("FPPG vs ADP", 0.99),
)


def signed_number(value):
    if value is None:
        return "—"
    if value == 0:
        return "0"
    return f"{value:+,.3f}".rstrip("0").rstrip(".")


def details(row):
    rank = row["next_rank"]
    starts = row["next_starts"]
    fppg = row["next_fppg"]
    return (
        signed_number(row["adp_slots_gained"]),
        f"{row['adp']:g} to {rank if rank is not None else '—'}",
        f"{starts if starts is not None else '—'} / {row['next_games']}",
        f"{fppg:.3f}" if fppg is not None else "—",
        signed_number(row["fppg_above_adp"]),
    )


def starter_page(title, rows, limit):
    count = len(rows)
    row_height = 0.25
    height = max(4.0, row_height * count + 2.25)
    figure = Figure(figsize=(13.4, height), facecolor="white")
    FigureCanvasAgg(figure)
    figure.text(0.04, 1 - 0.30 / height, "Second-year players who started their team opener",
                fontsize=17, fontweight="bold", color=TEXT, va="top")
    subtitle = f"{title} · FPPG ranking · {count} {'player' if count == 1 else 'players'}"
    ranked = [row["adp_slots_gained"] for row in rows if row["adp_slots_gained"] is not None]
    if ranked:
        subtitle += f" · {signed_number(median(ranked))} median slots"
    figure.text(0.04, 1 - 0.67 / height,
                f"{subtitle} · Started more than half of their regular-season games",
                fontsize=10, color=MUTED, va="top")
    if not rows:
        figure.text(0.04, 1 - 1.44 / height, "No players in this group.", fontsize=11, color=TEXT, va="top")
        return figure

    top = 1 - 1.40 / height
    panel_height = row_height * count / height
    bottom = top - panel_height
    names = figure.add_axes([0.04, bottom, 0.239, panel_height])
    chart = figure.add_axes([0.285, bottom, 0.31, panel_height])
    table = figure.add_axes([0.615, bottom, 0.345, panel_height])
    for axis in (names, chart, table):
        axis.set_ylim(count - 0.5, -0.5)
        axis.set_yticks([])
        for row_number in range(0, count, 2):
            axis.axhspan(row_number - 0.5, row_number + 0.5, color=STRIPE, linewidth=0, zorder=0)
    for axis in (names, table):
        axis.set_xlim(0, 1)
        axis.set_axis_off()

    def heading(axis, x, label, align):
        axis.text(x, 1 + 0.09 / (count * row_height), label, transform=axis.transAxes,
                  fontsize=8.5, fontweight="bold", color=TEXT, ha=align, va="bottom")

    heading(names, 0, "Player", "left")
    heading(names, 0.98, "Draft", "right")
    heading(chart, 0.5, "ADP slots gained by FPPG rank", "center")
    for label, x in DETAIL_HEADERS:
        heading(table, x, label, "right")

    chart.set_xlim(-limit, limit)
    chart.xaxis.set_major_locator(MaxNLocator(nbins=6, symmetric=True))
    chart.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+g}" if value else "0"))
    chart.grid(axis="x", color=GRID, linewidth=0.55)
    chart.set_axisbelow(True)
    chart.axvline(0, color=MUTED, linewidth=0.85)
    chart.tick_params(axis="x", length=3, labelsize=8, colors=MUTED)
    for side in ("left", "right", "top"):
        chart.spines[side].set_visible(False)
    chart.spines["bottom"].set_color(GRID)

    for index, row in enumerate(rows):
        names.text(0, index, row["player"], va="center", fontsize=9, color=TEXT)
        names.text(0.98, index, row["draft_season"], ha="right", va="center", fontsize=8.5, color=MUTED)
        for (_, x), value in zip(DETAIL_HEADERS, details(row)):
            table.text(x, index, value, ha="right", va="center", fontsize=8.5, color=TEXT)
        slots = row["adp_slots_gained"]
        if slots:
            chart.barh(index, slots, height=0.60, color=WIN if row["beat_adp"] else MISS, zorder=3)
        elif slots == 0:
            chart.scatter(0, index, color=MUTED, s=13, zorder=3)

    figure.text(0.04, bottom - 0.47 / height,
                "Slots = average draft pick minus FPPG rank. FPPG vs ADP = fantasy points per game minus FPPG at the ADP slot.",
                fontsize=8.5, color=MUTED, va="top")
    return figure


def write_starter_plots(output, rows):
    """Write the full cohort, ADP winners, and ties or misses to a three-page PDF."""
    ordered = sorted(rows, key=lambda row: (
        row["adp_slots_gained"] is None, -(row["adp_slots_gained"] or 0), row["player"].casefold(), row["draft_season"],
    ))
    limit = max(1.0, max((abs(row["adp_slots_gained"] or 0) for row in ordered), default=0.0)) * 1.07
    pages = (
        ("All qualified players", ordered),
        ("Beat ADP", [row for row in ordered if row["beat_adp"]]),
        ("Tied or missed ADP", [row for row in ordered if not row["beat_adp"]]),
    )
    with PdfPages(output, metadata={"Title": "Second-year opening-game starters vs ADP by FPPG rank"}) as pdf:
        for title, group in pages:
            pdf.savefig(starter_page(title, group, limit))
