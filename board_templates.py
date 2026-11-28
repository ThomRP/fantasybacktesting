"""Shared dark-mode HTML/CSS/JS templates for 'The Draft Model' (Points & 9-Cat boards)."""

SHARED_DARK_CSS = r"""
  :root {
    --bg: #080c14;
    --surface: #0f1624;
    --surface-2: #151e30;
    --surface-3: #1b273d;
    --ink: #f1f5f9;
    --ink-2: #cbd5e1;
    --muted: #8494ad;
    --border: rgba(148, 163, 184, 0.14);
    --border-strong: rgba(148, 163, 184, 0.26);
    --row-alt: rgba(255, 255, 255, 0.018);
    --row-hover: rgba(56, 189, 248, 0.08);
    --pos: #34d399;
    --pos-bg: rgba(16, 185, 129, 0.13);
    --pos-bd: rgba(52, 211, 153, 0.30);
    --neg: #fb7185;
    --neg-bg: rgba(244, 63, 94, 0.13);
    --neg-bd: rgba(251, 113, 133, 0.30);
    --amber: #fbbf24;
    --amber-bg: rgba(245, 158, 11, 0.13);
    --amber-bd: rgba(251, 191, 36, 0.30);
    --accent: #38bdf8;
    --accent-solid: #0284c7;
    --accent-bg: rgba(56, 189, 248, 0.13);
    --accent-bd: rgba(56, 189, 248, 0.32);
    --orange: #fb923c;
    --shadow: 0 4px 20px rgba(0, 0, 0, 0.32);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 20px 18px 52px;
    background:
      radial-gradient(ellipse 85% 38% at 50% -10%, rgba(56, 189, 248, 0.09), transparent 70%),
      radial-gradient(ellipse 60% 30% at 85% 0%, rgba(52, 211, 153, 0.06), transparent 65%),
      var(--bg);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.45;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: __MAX_WIDTH__px; margin: 0 auto; }

  /* Minimal Top Bar */
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
  }
  .brand-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .brand-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--pos);
    box-shadow: 0 0 12px rgba(52, 211, 153, 0.75);
    flex-shrink: 0;
  }
  h1 {
    margin: 0;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--ink);
  }
  .fmt-pill {
    font-size: 10.5px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
    padding: 3px 9px; border-radius: 999px;
    background: var(--surface-3); color: var(--ink-2); border: 1px solid var(--border-strong);
  }
  .season-pill {
    font-size: 12px; font-weight: 700; color: var(--ink);
    padding: 3px 9px; border-radius: 999px;
    background: var(--surface-2); border: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
  }
  .split-badge {
    font-size: 10.5px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border);
    background: var(--surface-2); color: var(--muted);
  }
  .split-badge.live { background: var(--pos-bg); border-color: var(--pos-bd); color: var(--pos); }
  .split-badge.past { background: var(--surface-2); border-color: var(--border-strong); color: var(--ink-2); }
  .header-meta {
    font-size: 12.5px; color: var(--muted);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 999px; padding: 5px 13px;
    font-variant-numeric: tabular-nums;
  }
  .header-meta strong { color: var(--ink); }

  /* Ultra-Compact Collapsible Info Strip */
  .info-strip { margin-bottom: 12px; }
  .info-pills { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; }
  .info-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; font-weight: 600; color: var(--muted);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 999px; padding: 5px 12px; cursor: pointer;
    transition: all 0.14s ease;
  }
  .info-pill:hover { color: var(--ink); border-color: var(--border-strong); background: var(--surface-2); }
  .info-pill.active { color: var(--ink); background: var(--surface-2); border-color: var(--accent-bd); }
  .info-pill .dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
  .info-pill .dot.accent { background: var(--accent); }
  .info-pill .dot.pos { background: var(--pos); }
  .info-pill .dot.neg { background: var(--neg); }
  .info-pill .dot.amber { background: var(--amber); }
  .info-pill .chev { font-size: 10px; opacity: 0.7; transition: transform 0.15s ease; }
  .info-pill.active .chev { transform: rotate(180deg); opacity: 1; }
  .info-drawer {
    display: none;
    margin-top: 8px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 12px 15px;
    font-size: 13px;
    color: var(--ink-2);
    line-height: 1.52;
    box-shadow: var(--shadow);
  }
  .info-drawer.open { display: block; }
  .info-drawer.tone-pos { border-left-color: var(--pos); }
  .info-drawer.tone-neg { border-left-color: var(--neg); }
  .info-drawer.tone-amber { border-left-color: var(--amber); }
  .info-drawer strong { color: var(--ink); }

  /* Sleek 2-Column Pick-Band Edge Hero */
  .edge-hero {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, var(--surface) 58%);
    border: 1px solid var(--pos-bd);
    border-radius: 14px;
    padding: 15px 18px;
    margin-bottom: 12px;
    box-shadow: var(--shadow);
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: start;
  }
  @media (max-width: 920px) { .edge-hero { grid-template-columns: 1fr; gap: 14px; } }
  .edge-hero-lbl {
    font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--muted); margin-bottom: 4px;
  }
  .edge-hero-big { display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap; }
  .edge-hero-num {
    font-size: 28px; font-weight: 800; letter-spacing: -0.03em;
    line-height: 1.05; font-variant-numeric: tabular-nums;
  }
  .edge-hero-unit { font-size: 13.5px; font-weight: 700; color: var(--ink-2); }
  .edge-hero-pill {
    font-size: 11px; font-weight: 700;
    padding: 2px 9px; border-radius: 999px; border: 1px solid var(--border-strong);
    background: var(--surface-2); color: var(--ink-2);
  }
  .edge-hero-pill.pos { color: var(--pos); background: var(--pos-bg); border-color: var(--pos-bd); }
  .edge-hero-pill.neg { color: var(--neg); background: var(--neg-bg); border-color: var(--neg-bd); }
  .edge-hero-sub {
    font-size: 11.5px; color: var(--muted); margin-top: 4px; font-variant-numeric: tabular-nums;
  }
  .edge-hero-sub strong { color: var(--ink-2); }
  .edge-hero-div { border-left: 1px solid var(--border); padding-left: 20px; }
  @media (max-width: 920px) { .edge-hero-div { border-left: none; padding-left: 0; border-top: 1px solid var(--border); padding-top: 14px; } }

  .band-table { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }
  .band-head {
    display: grid;
    grid-template-columns: 88px 1fr 48px 82px;
    gap: 8px;
    align-items: center;
    font-size: 9.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--muted);
    padding-bottom: 2px;
  }
  .band-head .c-mid { text-align: center; }
  .band-head .c-r { text-align: right; }
  .band-row {
    display: grid;
    grid-template-columns: 88px 1fr 48px 82px;
    gap: 8px;
    align-items: center;
    font-size: 11.5px;
    font-variant-numeric: tabular-nums;
  }
  .band-lbl { font-weight: 600; color: var(--ink-2); white-space: nowrap; }
  .band-track {
    position: relative; height: 8px;
    background: rgba(148, 163, 184, 0.12);
    border-radius: 999px; overflow: hidden;
  }
  .band-midline {
    position: absolute; top: 0; bottom: 0; left: 50%; width: 1px;
    background: rgba(148, 163, 184, 0.48); z-index: 2;
  }
  .band-fill { position: absolute; top: 0; bottom: 0; border-radius: 2px; }
  .band-fill.pos { background: var(--pos); }
  .band-fill.neg { background: var(--neg); }
  .band-pct { text-align: right; font-weight: 700; }
  .band-gap { text-align: right; font-size: 11px; font-weight: 700; white-space: nowrap; }
  @media (max-width: 520px) {
    .band-head, .band-row { grid-template-columns: 78px 1fr 44px 72px; gap: 6px; font-size: 11px; }
  }

  /* Unified Season Switcher + Replay Card */
  .replay-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 13px 16px 12px;
    margin-bottom: 12px;
    box-shadow: var(--shadow);
  }
  .season-strip {
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px;
    padding-bottom: 11px; margin-bottom: 11px;
    border-bottom: 1px solid var(--border);
  }
  .season-group { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; }
  .season-group-lbl {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--muted); margin-right: 3px;
  }
  .season-btn {
    font-size: 11.5px; font-weight: 600;
    background: var(--surface-2); border: 1px solid var(--border); color: var(--ink-2);
    padding: 4px 9px; border-radius: 7px; cursor: pointer;
    transition: all 0.12s ease; font-variant-numeric: tabular-nums;
  }
  .season-btn:hover { background: var(--accent-bg); border-color: var(--accent-bd); color: var(--accent); }
  .season-btn.live { border-color: var(--pos-bd); color: var(--pos); background: var(--pos-bg); }
  .season-btn.active {
    background: var(--ink); border-color: var(--ink); color: var(--bg); font-weight: 800;
  }

  .replay-top {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 10px; margin-bottom: 9px;
  }
  .replay-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; }
  .play-btn {
    font-size: 12px; font-weight: 700;
    background: var(--pos); border: 1px solid var(--pos); color: #042f2e;
    padding: 5px 13px; border-radius: 8px; cursor: pointer; min-width: 96px;
    transition: opacity 0.12s ease;
  }
  .play-btn:hover { opacity: 0.9; }
  .play-btn.playing { background: var(--orange); border-color: var(--orange); color: #1c1917; }
  .mini-btn {
    font-size: 11.5px; font-weight: 600;
    background: var(--surface-2); border: 1px solid var(--border); color: var(--ink-2);
    padding: 5px 9px; border-radius: 7px; cursor: pointer;
    transition: all 0.12s ease;
  }
  .mini-btn:hover { border-color: var(--accent-bd); color: var(--accent); }
  .mini-btn.active { background: var(--accent-bg); border-color: var(--accent-bd); color: var(--accent); }
  .replay-clock { font-size: 12.5px; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
  .replay-clock span { color: var(--muted); font-weight: 500; }

  .slider-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .slider-end { font-size: 11px; font-weight: 600; color: var(--muted); min-width: 30px; font-variant-numeric: tabular-nums; }
  .slider-end.right { text-align: right; }
  input[type="range"]#game-slider { flex: 1; accent-color: var(--accent); cursor: pointer; height: 5px; }

  .chart-header {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 8px; margin-bottom: 7px;
  }
  .chart-title { font-size: 11.5px; font-weight: 600; color: var(--muted); }
  .chart-modes { display: flex; gap: 5px; flex-wrap: wrap; }
  #edge-chart {
    width: 100%; height: 110px; display: block;
    background: #0a101d; border: 1px solid var(--border);
    border-radius: 9px; cursor: crosshair;
  }

  /* Collapsible Season Breakdown Bar inside Replay Card */
  .breakdown-bar {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px;
    margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border);
    cursor: pointer; user-select: none;
  }
  .breakdown-title-open {
    display: none; font-size: 10.5px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
  }
  .breakdown-bar.is-open .breakdown-title-open { display: inline; }
  .breakdown-bar.is-open .breakdown-teaser { display: none; }
  .breakdown-teaser {
    display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px;
    font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums;
  }
  .breakdown-teaser strong { color: var(--ink); }
  .breakdown-btn {
    font-size: 11.5px; font-weight: 700; color: var(--accent);
    background: var(--accent-bg); border: 1px solid var(--accent-bd);
    border-radius: 999px; padding: 3px 11px; cursor: pointer; flex-shrink: 0;
  }
  .breakdown-body { display: none; margin-top: 12px; }
  .breakdown-body.open { display: block; }
  .stats-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px; margin-bottom: 10px;
  }
  .stat-box {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px;
    cursor: pointer; user-select: none;
    transition: border-color 0.14s ease, background 0.14s ease, box-shadow 0.14s ease;
  }
  .stat-box:hover {
    border-color: var(--accent-bd);
    background: var(--surface-3);
  }
  .stat-box.active {
    border-color: var(--accent);
    background: rgba(56, 189, 248, 0.08);
    box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.28);
  }
  .stat-head {
    display: flex; justify-content: space-between; align-items: flex-start; gap: 6px;
    margin-bottom: 5px;
  }
  .stat-label {
    font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--muted);
  }
  .stat-hint {
    font-size: 10px; font-weight: 700; color: var(--accent);
    white-space: nowrap; opacity: 0.85;
  }
  .stat-box:hover .stat-hint, .stat-box.active .stat-hint { opacity: 1; }
  .stat-val {
    font-size: 21px; font-weight: 800; letter-spacing: -0.02em;
    line-height: 1.15; color: var(--ink); font-variant-numeric: tabular-nums;
  }
  .stat-sub { font-size: 11.5px; color: var(--muted); margin-top: 5px; line-height: 1.4; }
  .stat-sub strong { color: var(--ink-2); font-weight: 600; }

  /* Click-to-Inspect Side-by-Side Comparison Drawer */
  .compare-drawer {
    display: none;
    background: #0b111f;
    border: 1px solid var(--accent-bd);
    border-radius: 11px;
    padding: 13px 14px;
    margin-bottom: 10px;
    box-shadow: var(--shadow);
  }
  .compare-drawer.open { display: block; }
  .compare-head {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start;
    gap: 8px 14px; margin-bottom: 11px;
  }
  .compare-title { font-size: 13.5px; font-weight: 800; color: var(--ink); letter-spacing: -0.01em; }
  .compare-sub { font-size: 12px; color: var(--muted); margin-top: 2px; font-variant-numeric: tabular-nums; }
  .compare-sub strong { color: var(--ink-2); }
  .compare-close {
    font-size: 11.5px; font-weight: 700; color: var(--muted);
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 7px; padding: 4px 10px; cursor: pointer; flex-shrink: 0;
  }
  .compare-close:hover { color: var(--ink); border-color: var(--border-strong); }
  .compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 900px) { .compare-grid { grid-template-columns: 1fr; } }
  .compare-col {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden;
    display: flex; flex-direction: column;
  }
  .compare-col-head {
    display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap;
    padding: 8px 11px; background: var(--surface-2); border-bottom: 1px solid var(--border);
    font-size: 11.5px; font-weight: 700; color: var(--ink);
  }
  .compare-col-head .c-dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; vertical-align: 1px;
  }
  .compare-col-head .c-dot.pos { background: var(--pos); }
  .compare-col-head .c-dot.neg { background: var(--neg); }
  .compare-col-meta { font-size: 11px; font-weight: 600; color: var(--muted); font-variant-numeric: tabular-nums; }
  .compare-col-meta strong { color: var(--ink-2); }
  .compare-tbl-wrap { max-height: 360px; overflow-y: auto; }
  .compare-tbl { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
  .compare-tbl thead th {
    position: sticky; top: 0; z-index: 2;
    background: #0d1526; border-bottom: 1px solid var(--border-strong);
    padding: 6px 8px; text-align: right; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); cursor: default;
  }
  .compare-tbl thead th:hover { background: #0d1526; color: var(--muted); }
  .compare-tbl thead th.left, .compare-tbl tbody td.left, .compare-tbl tfoot td.left { text-align: left; padding-left: 10px; }
  .compare-tbl tbody td { padding: 5.5px 8px; text-align: right; white-space: nowrap; color: var(--ink-2); }
  .compare-tbl tbody td.p-name { color: var(--ink); font-weight: 600; }
  .compare-tbl tfoot td {
    position: sticky; bottom: 0; z-index: 2;
    background: #0d1526; border-top: 1px solid var(--border-strong);
    padding: 7px 8px; text-align: right; font-weight: 700; color: var(--ink);
    white-space: nowrap;
  }
  .t150-tag {
    display: inline-block; font-size: 9px; font-weight: 700; padding: 1px 4px;
    border-radius: 4px; margin-left: 5px; vertical-align: 1px;
    background: var(--pos-bg); color: var(--pos); border: 1px solid var(--pos-bd);
  }
  .mini-call {
    display: inline-block; font-size: 9px; font-weight: 700; padding: 1px 5px;
    border-radius: 999px; margin-left: 5px; vertical-align: 1px;
    border: 1px solid transparent;
  }
  .mini-call.hit { color: var(--pos); background: var(--pos-bg); border-color: var(--pos-bd); }
  .mini-call.miss { color: var(--neg); background: var(--neg-bg); border-color: var(--neg-bd); }
  .bands {
    font-size: 12px; color: var(--muted);
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 9px; padding: 8px 13px;
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
    font-variant-numeric: tabular-nums;
  }
  .bands .sep { color: var(--border-strong); }
  .bands strong { color: var(--ink); font-weight: 700; }

  /* Controls & Collapsible Drawers */
  .controls {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 10px; margin-bottom: 9px;
  }
  .tabs { display: flex; flex-wrap: wrap; gap: 5px; }
  .tab-btn {
    font-size: 12.5px; font-weight: 600;
    background: var(--surface); border: 1px solid var(--border); color: var(--ink-2);
    padding: 6px 12px; border-radius: 9px; cursor: pointer; transition: all 0.12s ease;
  }
  .tab-btn:hover { border-color: var(--accent-bd); color: var(--accent); }
  .tab-btn.active { background: var(--accent-solid); border-color: var(--accent-solid); color: #ffffff; }
  .filters { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; }
  .filters input[type="text"] {
    font-size: 12.5px; padding: 6px 11px;
    border: 1px solid var(--border-strong); border-radius: 8px;
    background: var(--surface); color: var(--ink); width: 165px; outline: none;
  }
  .filters input[type="text"]:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-bg); }
  .filters input[type="text"]::placeholder { color: var(--muted); }
  .pill {
    font-size: 12px; font-weight: 600;
    background: var(--surface); border: 1px solid var(--border); color: var(--ink-2);
    padding: 6px 10px; border-radius: 8px; cursor: pointer; transition: all 0.12s ease;
  }
  .pill:hover { background: var(--surface-2); color: var(--accent); border-color: var(--accent-bd); }
  .pill.active { background: var(--accent-bg); border-color: var(--accent-bd); color: var(--accent); }
  .pill.punt-on { background: var(--neg-bg); border-color: var(--neg-bd); color: var(--neg); }

  .drawer-panel {
    display: none;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 11px 14px; margin-bottom: 10px;
    font-size: 12px; color: var(--ink-2); box-shadow: var(--shadow);
  }
  .drawer-panel.open { display: block; }
  .guide-items { display: flex; flex-wrap: wrap; gap: 8px 18px; }
  .guide-items span { white-space: nowrap; }
  .guide-items code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px; font-weight: 700;
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 4px; padding: 1px 5px; color: var(--ink); margin-right: 4px;
  }

  /* Collapsible Punt Bar (9-Cat) */
  .punt-row { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; }
  .punt-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-right: 3px; }
  .punt-chip {
    font-size: 11.5px; font-weight: 700;
    background: var(--pos-bg); border: 1px solid var(--pos-bd); color: var(--pos);
    padding: 4px 10px; border-radius: 8px; cursor: pointer; transition: all 0.12s ease;
  }
  .punt-chip:hover { opacity: 0.85; }
  .punt-chip.punted { background: var(--neg-bg); border-color: var(--neg-bd); color: var(--neg); text-decoration: line-through; }
  .punt-reset {
    font-size: 11.5px; font-weight: 600;
    background: var(--surface-2); border: 1px solid var(--border); color: var(--muted);
    padding: 4px 9px; border-radius: 7px; cursor: pointer;
  }
  .punt-reset:hover { color: var(--ink); border-color: var(--border-strong); }
  .punt-hint { margin-left: auto; font-size: 12px; color: var(--muted); }
  .punt-hint strong { color: var(--ink); }

  /* Compact Table Meta + Inline Call Key */
  .table-meta {
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
    gap: 8px 14px; font-size: 12px; color: var(--muted);
    margin-bottom: 8px; padding: 0 2px; font-variant-numeric: tabular-nums;
  }
  .table-meta strong { color: var(--ink-2); }
  .inline-key { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; font-size: 11.5px; color: var(--muted); }
  .inline-key .call-badge { min-width: auto; padding: 1px 6px; font-size: 10px; }

  /* Dark Data Table */
  .table-wrap {
    overflow-x: auto;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow);
  }
  table { width: 100%; border-collapse: collapse; font-size: __TABLE_FS__px; font-variant-numeric: tabular-nums; }
  thead th {
    position: sticky; top: 0; z-index: 2;
    background: #0c1322;
    border-bottom: 1px solid var(--border-strong);
    padding: 9px __TH_PAD__px; text-align: right;
    font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--muted);
    cursor: pointer; user-select: none; white-space: nowrap;
  }
  thead th.left { text-align: left; padding-left: 11px; }
  thead th.group-sep, tbody td.group-sep { border-left: 1px solid var(--border); }
  thead th.punted-col, tbody td.punted-col { opacity: 0.30; text-decoration: line-through; }
  thead th:hover { background: #131d32; color: var(--ink); }
  thead th .sort-arrow { color: var(--accent); font-size: 10px; margin-left: 3px; }
  tbody tr { border-bottom: 1px solid rgba(148, 163, 184, 0.08); transition: background 0.08s ease; }
  tbody tr:nth-child(even) { background: var(--row-alt); }
  tbody tr:hover { background: var(--row-hover); }
  tbody td { padding: 7.5px __TH_PAD__px; text-align: right; white-space: nowrap; color: var(--ink-2); }
  tbody td.left { text-align: left; padding-left: 11px; }
  td.player-cell { font-size: 13px; font-weight: 600; color: var(--ink); }
  .sub-num { color: var(--muted); font-size: 11px; margin-left: 3px; }
  .tag {
    display: inline-block; font-size: 9.5px; font-weight: 700;
    padding: 1px 5px; border-radius: 999px; border: 1px solid var(--border);
    color: var(--muted); margin-left: 5px; vertical-align: 1px;
  }
  .tag-rook { border-color: var(--accent-bd); color: var(--accent); background: var(--accent-bg); }
  .tag-ret { border-color: rgba(251, 146, 60, 0.35); color: var(--orange); background: rgba(251, 146, 60, 0.13); }
  .tag-inj { border-color: var(--amber-bd); color: var(--amber); background: var(--amber-bg); }
  .pos { color: var(--pos); font-weight: 700; }
  .neg { color: var(--neg); font-weight: 700; }
  .muted { color: var(--muted); }
  .edge-pill { display: inline-block; padding: 2px 6px; border-radius: 6px; min-width: 48px; text-align: right; font-weight: 700; }
  .edge-pos { background: var(--pos-bg); color: var(--pos); }
  .edge-neg { background: var(--neg-bg); color: var(--neg); }
  .edge-flat { color: var(--muted); font-weight: 500; }
  .cat-cell { font-size: 11.5px; font-weight: 600; border-radius: 4px; padding: 2px 4px; display: inline-block; min-width: 40px; text-align: right; }
  .cat-hi { background: rgba(16, 185, 129, 0.24); color: #6ee7b7; }
  .cat-midpos { background: rgba(16, 185, 129, 0.11); color: #34d399; }
  .cat-lo { background: rgba(244, 63, 94, 0.24); color: #fda4af; }
  .cat-midneg { background: rgba(244, 63, 94, 0.11); color: #fb7185; }
  .call-badge {
    display: inline-block; font-size: 10.5px; padding: 2px 7px; border-radius: 999px;
    border: 1px solid transparent; font-weight: 700; min-width: 58px; text-align: center;
  }
  .call-hit { color: var(--pos); background: var(--pos-bg); border-color: var(--pos-bd); }
  .call-miss { color: var(--neg); background: var(--neg-bg); border-color: var(--neg-bd); }
  .call-agree { color: var(--muted); background: var(--surface-2); border-color: var(--border); font-weight: 600; }

  footer {
    margin-top: 16px; font-size: 11.5px; color: var(--muted);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
  }
"""


POINTS_HTML_TEMPLATE = (
    r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Draft Model &mdash; Points</title>
<style>"""
    + SHARED_DARK_CSS.replace("__MAX_WIDTH__", "1220").replace("__TABLE_FS__", "13").replace("__TH_PAD__", "10")
    + r"""</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand-row">
      <span class="brand-dot"></span>
      <h1>The Draft Model</h1>
      <span class="fmt-pill">POINTS</span>
      <span class="season-pill" id="season-pill">2026&ndash;27</span>
      <span class="split-badge live" id="split-badge">LIVE</span>
    </div>
    <div class="header-meta" id="header-meta"></div>
  </header>

  <section class="info-strip">
    <div class="info-pills">
      <button class="info-pill" data-info="why"><span class="dot accent"></span>Why This Exists <span class="chev">&#9662;</span></button>
      <button class="info-pill" data-info="what"><span class="dot pos"></span>What This Is <span class="chev">&#9662;</span></button>
      <button class="info-pill" data-info="not"><span class="dot neg"></span>Not Covered <span class="chev">&#9662;</span></button>
      <button class="info-pill" data-info="edge"><span class="dot amber"></span>How Edge Is Measured <span class="chev">&#9662;</span></button>
    </div>
    <div class="info-drawer" id="info-drawer"></div>
  </section>

  <section class="edge-hero" id="edge-hero">
    <div id="edge-hero-multi"></div>
    <div class="edge-hero-div" id="edge-hero-season"></div>
  </section>

  <section class="replay-panel">
    <div class="season-strip" id="season-bar"></div>
    <div class="replay-top">
      <div class="replay-controls">
        <button class="play-btn" id="play-btn">&#9654; Play</button>
        <button class="mini-btn" id="reset-btn">G 1</button>
        <button class="mini-btn" id="end-btn">End</button>
        <button class="mini-btn active" data-speed="250">4&times;</button>
        <button class="mini-btn" data-speed="125">8&times;</button>
      </div>
      <div class="replay-clock" id="replay-clock"></div>
    </div>
    <div class="slider-row">
      <span class="slider-end">G 1</span>
      <input type="range" id="game-slider" min="1" max="82" value="82">
      <span class="slider-end right" id="slider-max-lbl">G 82</span>
    </div>
    <div class="chart-header">
      <div class="chart-title">Season Trajectory (click chart to jump)</div>
      <div class="chart-modes">
        <button class="mini-btn active" data-chart="t15_vs">15 Liked vs. Faded (vs ADP)</button>
        <button class="mini-btn" data-chart="top150_fpts">Top 150 Swaps</button>
        <button class="mini-btn" data-chart="flips_pct">% Calls Right</button>
      </div>
    </div>
    <svg id="edge-chart" viewBox="0 0 1000 110" preserveAspectRatio="none"></svg>

    <div class="breakdown-bar is-open" id="breakdown-toggle">
      <span class="breakdown-title-open">Season Breakdown</span>
      <div class="breakdown-teaser" id="breakdown-teaser"></div>
      <span class="breakdown-btn" id="breakdown-btn">Hide &#9652;</span>
    </div>
    <div class="breakdown-body open" id="breakdown-body">
      <section class="stats-grid" id="hero-cards"></section>
      <div class="compare-drawer" id="compare-drawer"></div>
      <div class="bands" id="band-strip"></div>
    </div>
  </section>

  <div class="controls">
    <div class="tabs" id="tabs">
      <button class="tab-btn active" data-tab="stage2_top150">Top 150</button>
      <button class="tab-btn" data-tab="stage1_top150">Pure Stats (0% ADP)</button>
      <button class="tab-btn" data-tab="deep_sleepers">Sleepers (151&ndash;225)</button>
    </div>
    <div class="filters">
      <input type="text" id="search" placeholder="Search player or pos...">
      <button class="pill active" data-filter="all">All</button>
      <button class="pill" data-filter="top15">Top 15</button>
      <button class="pill" data-filter="pos">Targets</button>
      <button class="pill" data-filter="neg">Fades</button>
      <button class="pill" data-filter="rook">Rookies</button>
      <button class="pill" id="guide-toggle">? Columns</button>
    </div>
  </div>

  <div class="drawer-panel" id="guide-drawer">
    <div class="guide-items">
      <span><code>&Delta;Rk</code> ADP rank minus model rank</span>
      <span><code>Edge</code> Projected gain vs. ADP peers</span>
      <span><code>Proj</code> Pre-draft 65-game reliability FPPG</span>
      <span><code>Act Rk</code> Actual rank at current game</span>
      <span><code>vs ADP</code> Actual pace minus ADP expectation</span>
      <span><code>Call</code> Grades |Edge| &ge; 0.25 or |&Delta;Rk| &ge; 3</span>
    </div>
  </div>

  <div class="table-meta">
    <div><span id="row-count"></span> &middot; <span id="filtered-summary"></span></div>
    <div class="inline-key">
      <span class="call-badge call-hit">TGT &#10003;</span><span class="call-badge call-miss">TGT &#10007;</span> Target
      <span class="call-badge call-hit">FADE &#10003;</span><span class="call-badge call-miss">FADE &#10007;</span> Fade
      <span class="call-badge call-agree">AGREE</span> Matches ADP
    </div>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th data-col="rk">Rk<span class="sort-arrow"></span></th>
          <th class="left" data-col="p">Player<span class="sort-arrow"></span></th>
          <th class="left" data-col="pos">Pos<span class="sort-arrow"></span></th>
          <th data-col="mrk">ADP<span class="sort-arrow"></span></th>
          <th data-col="drk">&Delta;Rk<span class="sort-arrow"></span></th>
          <th data-col="e">Edge<span class="sort-arrow"></span></th>
          <th data-col="pr">Proj<span class="sort-arrow"></span></th>
          <th class="group-sep" data-col="gp">GP<span class="sort-arrow"></span></th>
          <th data-col="fpts">FPTS<span class="sort-arrow"></span></th>
          <th data-col="act_rank">Act Rk<span class="sort-arrow"></span></th>
          <th data-col="act_vs_adp">vs ADP<span class="sort-arrow"></span></th>
          <th data-col="call_score">Call<span class="sort-arrow"></span></th>
        </tr>
      </thead>
      <tbody id="board-body"></tbody>
    </table>
  </div>

  <footer>
    <span>The Draft Model &middot; Strict out-of-sample pre-draft projections (ESPN Points)</span>
    <span id="footer-edge-summary"></span>
  </footer>
</div>

<script>
const APP = __PAYLOAD_JSON__;
const ANCHORS_RK = APP.anchor_ranks;
const PICK_BANDS = [
  { lbl: "Picks 1\u201325", lo: 1, hi: 25 },
  { lbl: "Picks 26\u201360", lo: 26, hi: 60 },
  { lbl: "Picks 61\u2013100", lo: 61, hi: 100 },
  { lbl: "Picks 101\u2013150", lo: 101, hi: 150 },
  { lbl: "Picks 151\u2013225", lo: 151, hi: 225 },
];

const state = {
  season: APP.default_season,
  step: 0,
  tab: "stage2_top150",
  filter: "all",
  search: "",
  sortCol: "rk",
  sortDir: 1,
  chartMode: "t15_vs",
  speedMs: 250,
  timer: null,
  activeInfo: null,
  compareBox: null,
};

const seasonCache = {};
let multiYearCache = null;

function getSeasonCum(seasonKey) {
  if (seasonCache[seasonKey]) return seasonCache[seasonKey];
  const S = APP.seasons[seasonKey];
  const nSteps = S.max_step;
  const cumGp = {};
  const cumFpts = {};
  for (const [aid, events] of Object.entries(S.logs)) {
    const gArr = new Int16Array(nSteps);
    const fArr = new Float32Array(nSteps);
    let ei = 0, runG = 0, runF = 0;
    for (let s = 0; s < nSteps; s++) {
      while (ei < events.length && events[ei][0] === s) {
        runG += events[ei][1];
        runF += events[ei][2];
        ei++;
      }
      gArr[s] = runG;
      fArr[s] = runF;
    }
    cumGp[aid] = gArr;
    cumFpts[aid] = fArr;
  }
  const out = { cumGp, cumFpts };
  seasonCache[seasonKey] = out;
  return out;
}

function interpSlotAdj(anchors, adp) {
  if (!anchors || !anchors.length) return 0;
  if (adp <= ANCHORS_RK[0]) return anchors[0];
  const last = ANCHORS_RK.length - 1;
  if (adp >= ANCHORS_RK[last]) return anchors[last];
  for (let i = 0; i < last; i++) {
    const r0 = ANCHORS_RK[i], r1 = ANCHORS_RK[i + 1];
    if (adp >= r0 && adp <= r1) {
      const t = (adp - r0) / (r1 - r0);
      return anchors[i] + t * (anchors[i + 1] - anchors[i]);
    }
  }
  return anchors[last];
}

function estimateRankFromAnchors(anchors, adjVal) {
  if (!anchors || !anchors.length) return 250;
  if (adjVal >= anchors[0]) return 1;
  const last = anchors.length - 1;
  if (adjVal <= anchors[last]) return ANCHORS_RK[last] + 1;
  for (let i = 0; i < last; i++) {
    const v0 = anchors[i], v1 = anchors[i + 1];
    if (adjVal <= v0 && adjVal >= v1) {
      const span = Math.max(v0 - v1, 1e-6);
      const t = (v0 - adjVal) / span;
      return Math.max(1, Math.round(ANCHORS_RK[i] + t * (ANCHORS_RK[i + 1] - ANCHORS_RK[i])));
    }
  }
  return ANCHORS_RK[last] + 1;
}

function fmtSigned(v, dec = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "&mdash;";
  return (v >= 0 ? "+" : "") + Number(v).toFixed(dec);
}
function fmtNum(v, dec = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "&mdash;";
  return Number(v).toFixed(dec);
}
function fmtInt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "&mdash;";
  return Math.round(Number(v)).toLocaleString("en-US");
}
function fmtSignedInt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "&mdash;";
  const n = Math.round(Number(v));
  return (n >= 0 ? "+" : "") + n.toLocaleString("en-US");
}

function enrichRowAt(r, cum, cp, stepIdx) {
  const gpArr = cum.cumGp[r.id];
  const fptsArr = cum.cumFpts[r.id];
  const gp = gpArr ? gpArr[stepIdx] : 0;
  const fpts = fptsArr ? fptsArr[stepIdx] : 0;
  const raw = gp > 0 ? fpts / gp : 0;
  const effG = Math.min(gp, cp.ref_g);
  const actAdj = gp > 0 ? ((effG * raw + (cp.ref_g - effG) * cp.waiver) / cp.ref_g) : cp.waiver;
  const actRank = gp > 0 ? estimateRankFromAnchors(cp.anchors, actAdj) : (ANCHORS_RK[ANCHORS_RK.length - 1] + 1);
  const slotAdj = r.adp !== null ? interpSlotAdj(cp.anchors, r.adp) : cp.waiver;
  const actVsAdp = actAdj - slotAdj;
  return {
    ...r,
    gp, fpts,
    act_fppg: raw,
    act_adj: actAdj,
    act_rank: actRank,
    act_vs_adp: actVsAdp,
    in_t150: actAdj >= cp.t150_cut - 1e-6 && gp > 0,
  };
}

function enrichRow(r, cum, cp) {
  return enrichRowAt(r, cum, cp, state.step);
}

function evalCall(r) {
  if (r.e === null || r.e === undefined) return { label: "&mdash;", cls: "call-agree", score: 0 };
  const bigEdge = Math.abs(r.e) >= 0.25;
  const bigShift = r.drk !== null && Math.abs(r.drk) >= 3;
  if (!bigEdge && !bigShift) return { label: "AGREE", cls: "call-agree", score: 0 };
  const liked = r.e > 0;
  const hit = liked ? (r.act_vs_adp > 0) : (r.act_vs_adp < 0);
  if (liked) return hit ? { label: "TGT \u2713", cls: "call-hit", score: 1 } : { label: "TGT \u2717", cls: "call-miss", score: -1 };
  return hit ? { label: "FADE \u2713", cls: "call-hit", score: 1 } : { label: "FADE \u2717", cls: "call-miss", score: -1 };
}

function getInfoContent(key) {
  if (key === "why") {
    return {
      tone: "",
      html: "Fantasy leagues are already <strong>well projected in general</strong>&mdash;consensus ADP gets most players roughly right, and throwing it away loses badly. The edge comes from repeatable <strong>human biases on a small group of players</strong>: overpaying for rookie hype, aging/injury-prone names, and hot streaks while overlooking durable, rising starters. Avoiding a few landmines and hitting a few undervalued targets builds league-winning teams.",
    };
  }
  if (key === "what") {
    return {
      tone: "tone-pos",
      html: "<strong>Pre-draft player projections</strong> built on top of consensus ADP (scored on a full-season 65-game reliability scale) that we believe outperform ADP on average&mdash;backed by strict out-of-sample historical replay and live tracking below. Green <strong style='color:var(--pos)'>Edge</strong> marks targets projected above ADP; red marks fades.",
    };
  }
  if (key === "not") {
    return {
      tone: "tone-neg",
      html: "Anything after draft day&mdash;<strong>no</strong> waiver pickups, trades, mid-season injuries, or rotation changes. This is strictly a pre-draft projection to help you leave the draft with a stronger roster; managing the season is up to you.",
    };
  }
  return {
    tone: "tone-amber",
    html: "Instead of tying results to one fixed league size, we grade every pre-draft disagreement with ADP across <strong>five pick ranges (1&ndash;25 through 151&ndash;225)</strong>. <strong>Calls Right %</strong> asks: when we rank Player A above Player B against ADP, how often does A actually outproduce B? (50% = coin flip.) <strong>Tgt vs. Fade</strong> compares how much more total fantasy points our liked targets scored than our fades drafted in the exact same pick range.",
  };
}

function computeBandsFromRows(rows) {
  return PICK_BANDS.map(b => {
    const sub = rows.filter(r => r.mrk !== null && r.mrk >= b.lo && r.mrk <= b.hi).sort((a, c) => a.mrk - c.mrk);
    let wins = 0, pairs = 0, tgtSum = 0, tgtCnt = 0, fadeSum = 0, fadeCnt = 0;
    for (let i = 0; i < sub.length; i++) {
      for (let j = i + 1; j < sub.length; j++) {
        if (sub[j].pr > sub[i].pr) {
          pairs++;
          if (sub[j].act_adj > sub[i].act_adj) wins++;
        }
      }
      const r = sub[i];
      const big = Math.abs(r.e) >= 0.25 || (r.drk !== null && Math.abs(r.drk) >= 3);
      if (big) {
        if (r.e > 0) { tgtSum += r.fpts; tgtCnt++; }
        else { fadeSum += r.fpts; fadeCnt++; }
      }
    }
    const pct = pairs > 0 ? (100 * wins / pairs) : null;
    const tAvg = tgtCnt > 0 ? tgtSum / tgtCnt : 0;
    const fAvg = fadeCnt > 0 ? fadeSum / fadeCnt : 0;
    const gapPct = (tgtCnt > 0 && fadeCnt > 0 && fAvg > 1) ? (100 * (tAvg - fAvg) / fAvg) : null;
    return { lbl: b.lbl, wins, pairs, pct, tgtSum, tgtCnt, fadeSum, fadeCnt, gapPct };
  });
}

function getMultiYearStats() {
  if (multiYearCache) return multiYearCache;
  const past = APP.season_order.filter(o => o.split !== "live");
  const agg = PICK_BANDS.map(b => ({ lbl: b.lbl, wins: 0, pairs: 0, tgtSum: 0, tgtCnt: 0, fadeSum: 0, fadeCnt: 0 }));
  let totalWins = 0, totalPairs = 0, winYrs = 0;
  for (const item of past) {
    const S = APP.seasons[item.season];
    const cum = getSeasonCum(item.season);
    const lastStep = S.max_step - 1;
    const cp = S.checkpoints[lastStep];
    const rows = S.s2_all.map(r => enrichRowAt(r, cum, cp, lastStep));
    const bands = computeBandsFromRows(rows);
    bands.forEach((b, idx) => {
      agg[idx].wins += b.wins;
      agg[idx].pairs += b.pairs;
      agg[idx].tgtSum += b.tgtSum;
      agg[idx].tgtCnt += b.tgtCnt;
      agg[idx].fadeSum += b.fadeSum;
      agg[idx].fadeCnt += b.fadeCnt;
    });
    const mkt150 = rows.filter(r => r.mrk !== null && r.mrk <= 150).sort((a, b) => a.mrk - b.mrk);
    let sw = 0, st = 0;
    for (let i = 0; i < mkt150.length; i++) {
      for (let j = i + 1; j < mkt150.length; j++) {
        if (mkt150[j].pr > mkt150[i].pr) {
          st++;
          if (mkt150[j].act_adj > mkt150[i].act_adj) sw++;
        }
      }
    }
    totalWins += sw;
    totalPairs += st;
    if (st > 0 && sw * 2 >= st) winYrs++;
  }
  const bands = agg.map(b => {
    const pct = b.pairs > 0 ? (100 * b.wins / b.pairs) : null;
    const tAvg = b.tgtCnt > 0 ? b.tgtSum / b.tgtCnt : 0;
    const fAvg = b.fadeCnt > 0 ? b.fadeSum / b.fadeCnt : 0;
    const gapPct = (b.tgtCnt > 0 && b.fadeCnt > 0 && fAvg > 1) ? (100 * (tAvg - fAvg) / fAvg) : null;
    return { ...b, pct, gapPct };
  });
  multiYearCache = {
    nYrs: past.length,
    winYrs,
    totalWins,
    totalPairs,
    overallPct: totalPairs > 0 ? (100 * totalWins / totalPairs) : 50,
    bands,
  };
  return multiYearCache;
}

function renderBandTableHtml(bands) {
  const head = `<div class="band-head"><span>Draft Range</span><span class="c-mid">vs. 50% Coin Flip</span><span class="c-r">Right</span><span class="c-r">Tgt vs Fade</span></div>`;
  const rows = bands.map(b => {
    let fillHtml = "";
    let pctCls = "muted", pctStr = "&mdash;";
    if (b.pct !== null) {
      pctCls = b.pct >= 50 ? "pos" : "neg";
      pctStr = `${b.pct.toFixed(1)}%`;
      const w = Math.min(50, (Math.abs(b.pct - 50) / 20) * 50);
      const style = b.pct >= 50 ? `left:50%;width:${w.toFixed(1)}%` : `left:${(50 - w).toFixed(1)}%;width:${w.toFixed(1)}%`;
      fillHtml = `<div class="band-fill ${pctCls}" style="${style}"></div>`;
    }
    const gapCls = b.gapPct === null ? "muted" : (b.gapPct >= 0 ? "pos" : "neg");
    const gapStr = b.gapPct === null ? "&mdash;" : `${b.gapPct >= 0 ? "+" : ""}${b.gapPct.toFixed(1)}% FPTS`;
    const tip = `${b.wins.toLocaleString()}/${b.pairs.toLocaleString()} disagreements right (${b.tgtCnt} targets vs ${b.fadeCnt} fades)`;
    return `<div class="band-row" title="${tip}">
      <span class="band-lbl">${b.lbl}</span>
      <div class="band-track"><div class="band-midline"></div>${fillHtml}</div>
      <span class="band-pct ${pctCls}">${pctStr}</span>
      <span class="band-gap ${gapCls}">${gapStr}</span>
    </div>`;
  }).join("");
  return `<div class="band-table">${head}${rows}</div>`;
}

function renderMultiYearHero() {
  const my = getMultiYearStats();
  const el = document.getElementById("edge-hero-multi");
  if (!el || !my.nYrs) return;
  const cls = my.overallPct >= 50 ? "pos" : "neg";
  const yrCls = my.winYrs * 2 >= my.nYrs ? "pos" : "neg";
  el.innerHTML = `
    <div class="edge-hero-lbl">${my.nYrs}-Year Track Record &middot; All League Sizes</div>
    <div class="edge-hero-big">
      <span class="edge-hero-num ${cls}">${my.overallPct.toFixed(1)}%</span>
      <span class="edge-hero-unit">Calls Right</span>
      <span class="edge-hero-pill ${yrCls}" title="Completed seasons where overall Picks 1&ndash;150 Calls Right finished above 50%">Beat ADP ${my.winYrs}/${my.nYrs} Yrs</span>
    </div>
    <div class="edge-hero-sub">Picks 1&ndash;150 overall (<strong>${my.totalWins.toLocaleString()}/${my.totalPairs.toLocaleString()}</strong> right &middot; 50% = coin flip)</div>
    ${renderBandTableHtml(my.bands)}
  `;
  const ft = document.getElementById("footer-edge-summary");
  const deep = my.bands[4];
  const deepStr = deep && deep.gapPct !== null ? ` &middot; ${deep.gapPct >= 0 ? "+" : ""}${deep.gapPct.toFixed(1)}% FPTS on Picks 151&ndash;225` : "";
  if (ft) {
    ft.innerHTML = `${my.nYrs}-Yr Out-of-Sample: ${my.overallPct.toFixed(1)}% Calls Right (1&ndash;150)${deepStr} &middot; Beat ADP ${my.winYrs}/${my.nYrs} yrs`;
  }
}

function renderSeasonHero(S, cp, curBands, flipsWon, flipsTotal, flipsPct) {
  const el = document.getElementById("edge-hero-season");
  if (!el) return;
  const isLive = S.split === "live";
  const cls = flipsPct >= 50 ? "pos" : "neg";
  const pill = isLive ? `${cp.g} Equalized G` : (cp.g === S.max_step ? "Full Season" : `Through G ${cp.g}`);
  el.innerHTML = `
    <div class="edge-hero-lbl">Selected Season &middot; ${S.season} (${isLive ? "LIVE" : "PAST"} &middot; G ${cp.g}/${S.max_step})</div>
    <div class="edge-hero-big">
      <span class="edge-hero-num ${cls}">${flipsPct.toFixed(1)}%</span>
      <span class="edge-hero-unit">Calls Right</span>
      <span class="edge-hero-pill">${pill}</span>
    </div>
    <div class="edge-hero-sub">Picks 1&ndash;150: <strong>${flipsWon.toLocaleString()}/${flipsTotal.toLocaleString()}</strong> disagreements right &middot; Scrub slider to replay</div>
    ${renderBandTableHtml(curBands)}
  `;
}

function buildPointsCompareCol(title, tone, rows, opts = {}) {
  const showCall = !!opts.showCall;
  const showT150 = !!opts.showT150;
  const showVsAdp = !!opts.showVsAdp;
  const totFpts = rows.reduce((a, r) => a + r.fpts, 0);
  const totGp = rows.reduce((a, r) => a + r.gp, 0);
  const avgFppg = totGp > 0 ? totFpts / totGp : 0;
  const avgGp = rows.length ? totGp / rows.length : 0;
  const avgVsAdp = rows.length ? rows.reduce((a, r) => a + r.act_vs_adp, 0) / rows.length : 0;
  const withAdp = rows.filter(r => r.mrk !== null);
  const avgAdp = withAdp.length ? withAdp.reduce((a, r) => a + r.mrk, 0) / withAdp.length : null;
  const avgFin = rows.length ? rows.reduce((a, r) => a + r.act_rank, 0) / rows.length : null;
  const hits150 = rows.filter(r => r.in_t150).length;
  const colMeta = showT150
    ? `<strong>${hits150}/${rows.length}</strong> Top-150 &middot; Avg <strong>${fmtNum(avgFppg, 1)}</strong> FPPG`
    : (showVsAdp
      ? `vs ADP <strong class="${avgVsAdp >= 0 ? "pos" : "neg"}">${fmtSigned(avgVsAdp, 2)}</strong> FPPG &middot; Avg <strong>${fmtNum(avgFppg, 1)}</strong> FPPG`
      : `Total <strong>${fmtInt(totFpts)}</strong> FPTS &middot; Avg <strong>${fmtNum(avgFppg, 1)}</strong> FPPG`);
  const midHdr = showCall ? "ADP&rarr;Rk" : "Edge";
  const nCols = showVsAdp ? 8 : 7;
  const bodyHtml = rows.length ? rows.map(r => {
    const call = showCall ? evalCall(r) : null;
    const callTag = call ? `<span class="mini-call ${call.score === 1 ? "hit" : "miss"}">${call.label}</span>` : "";
    const t150Tag = (showT150 && r.in_t150) ? `<span class="t150-tag">T150</span>` : "";
    const adpTxt = r.mrk !== null ? `#${r.mrk}` : "UNR";
    const midCell = showCall
      ? `<td class="${r.drk >= 0 ? "pos" : "neg"}">${adpTxt}&rarr;#${r.rk}</td>`
      : `<td class="${r.e >= 0.25 ? "pos" : (r.e <= -0.25 ? "neg" : "muted")}">${fmtSigned(r.e, 2)}</td>`;
    const fppgTxt = r.gp > 0 ? fmtNum(r.act_fppg, 1) : "&mdash;";
    const vsCell = showVsAdp
      ? `<td class="${r.act_vs_adp > 0.25 ? "pos" : (r.act_vs_adp < -0.25 ? "neg" : "muted")}">${fmtSigned(r.act_vs_adp, 1)}</td>`
      : "";
    const finTxt = r.act_rank > 250 ? "250+" : `#${r.act_rank}`;
    const finCls = (r.mrk !== null && r.act_rank <= r.mrk - 5) ? "pos" : ((r.mrk !== null && r.act_rank >= r.mrk + 5) ? "neg" : "");
    return `<tr>
      <td class="left p-name">${r.p}${callTag}${t150Tag}</td>
      <td class="muted">${adpTxt}</td>
      ${midCell}
      <td class="${r.gp === 0 ? "neg" : "muted"}">${r.gp}</td>
      <td>${fppgTxt}</td>
      <td><strong>${fmtInt(r.fpts)}</strong></td>
      ${vsCell}
      <td class="${finCls}"><strong>${finTxt}</strong></td>
    </tr>`;
  }).join("") : `<tr><td colspan="${nCols}" class="left muted">No players in this group</td></tr>`;

  const avgAdpTxt = avgAdp !== null ? `#${Math.round(avgAdp)}` : "&mdash;";
  const avgFinTxt = avgFin !== null ? `#${Math.round(avgFin)}` : "&mdash;";
  const avgFinCls = (avgAdp !== null && avgFin !== null) ? (avgFin <= avgAdp ? "pos" : "neg") : "";

  return `<div class="compare-col">
    <div class="compare-col-head">
      <span><span class="c-dot ${tone}"></span>${title} (${rows.length})</span>
      <span class="compare-col-meta">${colMeta}</span>
    </div>
    <div class="compare-tbl-wrap">
      <table class="compare-tbl">
        <thead><tr>
          <th class="left">Player</th>
          <th>ADP</th>
          <th>${midHdr}</th>
          <th>GP</th>
          <th>FPPG</th>
          <th>FPTS</th>
          ${showVsAdp ? "<th>vs ADP</th>" : ""}
          <th>Finish</th>
        </tr></thead>
        <tbody>${bodyHtml}</tbody>
        <tfoot><tr>
          <td class="left">Total / Avg</td>
          <td class="muted">${avgAdpTxt}</td>
          <td class="muted">&mdash;</td>
          <td>${fmtNum(avgGp, 1)}</td>
          <td>${fmtNum(avgFppg, 1)}</td>
          <td>${fmtInt(totFpts)}</td>
          ${showVsAdp ? `<td class="${avgVsAdp >= 0 ? "pos" : "neg"}">${fmtSigned(avgVsAdp, 1)}</td>` : ""}
          <td class="${avgFinCls}">${avgFinTxt}</td>
        </tr></tfoot>
      </table>
    </div>
  </div>`;
}

function renderCompareDrawer(ctx) {
  const drawer = document.getElementById("compare-drawer");
  if (!drawer) return;
  const key = state.compareBox;
  if (!key) {
    drawer.className = "compare-drawer";
    drawer.innerHTML = "";
    return;
  }

  let title = "", sub = "", leftHtml = "", rightHtml = "";
  if (key === "top150") {
    const swapsIn = ctx.s2Top150.filter(r => r.mrk === null || r.mrk > 150).sort((a, b) => b.fpts - a.fpts);
    const swapsOut = ctx.mktTop150.filter(r => r.rk > 150).sort((a, b) => b.fpts - a.fpts);
    const inGp = swapsIn.reduce((a, r) => a + r.gp, 0);
    const outGp = swapsOut.reduce((a, r) => a + r.gp, 0);
    const inFppg = inGp > 0 ? swapsIn.reduce((a, r) => a + r.fpts, 0) / inGp : 0;
    const outFppg = outGp > 0 ? swapsOut.reduce((a, r) => a + r.fpts, 0) / outGp : 0;
    title = `Our Top 150 Swaps In vs. Dropped from ADP Top 150`;
    sub = `Net difference: <strong class="${ctx.s2_150_diff >= 0 ? "pos" : "neg"}">${fmtSignedInt(ctx.s2_150_diff)} FPTS</strong> (Avg <strong>${fmtNum(inFppg, 1)}</strong> vs. <strong>${fmtNum(outFppg, 1)}</strong> FPPG) &middot; The other ${150 - swapsIn.length} players are on both lists`;
    leftHtml = buildPointsCompareCol("Added to Our Top 150", "pos", swapsIn, { showT150: true });
    rightHtml = buildPointsCompareCol("Dropped from ADP Top 150", "neg", swapsOut, { showT150: true });
  } else if (key === "t15") {
    const tTot = ctx.t15.reduce((a, r) => a + r.fpts, 0);
    const fTot = ctx.f15.reduce((a, r) => a + r.fpts, 0);
    title = `15 Most Liked Targets vs. 15 Most Faded Below ADP`;
    sub = `ADP-adjusted (each player vs. their own draft slot): <strong class="${ctx.t15VsSpread >= 0 ? "pos" : "neg"}">${fmtSigned(ctx.t15VsSpread, 2)} FPPG</strong> (Liked <strong>${fmtSigned(ctx.t15VsAvg, 2)}</strong> vs. Faded <strong>${fmtSigned(ctx.f15VsAvg, 2)}</strong>) &middot; Raw totals: ${fmtInt(tTot)} vs. ${fmtInt(fTot)} FPTS`;
    leftHtml = buildPointsCompareCol("15 Most Liked (Highest Edge)", "pos", ctx.t15, { showVsAdp: true });
    rightHtml = buildPointsCompareCol("15 Most Faded (Steepest Below ADP)", "neg", [...ctx.f15].sort((a, b) => a.e - b.e), { showVsAdp: true });
  } else if (key === "flips") {
    const graded = ctx.mktTop150
      .map(r => ({ r, sc: evalCall(r).score }))
      .filter(x => x.sc !== 0)
      .sort((a, b) => (Math.abs(b.r.drk || 0) - Math.abs(a.r.drk || 0)) || (Math.abs(b.r.e) - Math.abs(a.r.e)));
    const bigHits = graded.filter(x => x.sc === 1).slice(0, 15).map(x => x.r);
    const bigMiss = graded.filter(x => x.sc === -1).slice(0, 15).map(x => x.r);
    title = `Biggest Disagreements With ADP: Top Hits vs. Top Misses`;
    sub = `Overall Picks 1&ndash;150: <strong class="${ctx.flipsPct >= 50 ? "pos" : "neg"}">${ctx.flipsWon.toLocaleString()}/${ctx.flipsTotal.toLocaleString()} (${ctx.flipsPct.toFixed(1)}%)</strong> pairwise disagreements right &middot; Sorted by biggest rank shift vs. ADP`;
    leftHtml = buildPointsCompareCol("15 Biggest Calls Right", "pos", bigHits, { showCall: true });
    rightHtml = buildPointsCompareCol("15 Biggest Calls Missed", "neg", bigMiss, { showCall: true });
  } else if (key === "sleepers") {
    const posSorted = [...ctx.posDeep].sort((a, b) => b.act_adj - a.act_adj);
    const negSorted = [...ctx.negDeep].sort((a, b) => b.act_adj - a.act_adj);
    title = `Late-Round Sleepers (Picks 151&ndash;225): Targeted vs. Passed On`;
    sub = `Targeted sleepers reaching Top 150: <strong class="pos">${ctx.posDeepHit}/${ctx.posDeep.length} (${ctx.posDeepPct.toFixed(0)}%)</strong> vs. Passed-on picks: <strong>${ctx.negDeepHit}/${ctx.negDeep.length} (${ctx.negDeepPct.toFixed(0)}%)</strong>`;
    leftHtml = buildPointsCompareCol("Our Sleeper Targets (Edge > 0)", "pos", posSorted, { showT150: true });
    rightHtml = buildPointsCompareCol("Passed-On Late Picks (Edge \u2264 0)", "neg", negSorted, { showT150: true });
  }

  drawer.className = "compare-drawer open";
  drawer.innerHTML = `
    <div class="compare-head">
      <div>
        <div class="compare-title">${title}</div>
        <div class="compare-sub">${sub}</div>
      </div>
      <button class="compare-close" type="button">&times; Close</button>
    </div>
    <div class="compare-grid">${leftHtml}${rightHtml}</div>
  `;
}

function renderSeasonBar() {
  const past = APP.season_order.filter(x => x.split !== "live");
  const lv = APP.season_order.filter(x => x.split === "live");
  const mkBtns = (arr, cls) => arr.map(x => `
    <button class="season-btn ${cls} ${x.season === state.season ? "active" : ""}" data-season="${x.season}">${x.season}</button>
  `).join("");

  document.getElementById("season-bar").innerHTML = `
    <div class="season-group"><span class="season-group-lbl">Past Seasons</span>${mkBtns(past, "past")}</div>
    <div class="season-group"><span class="season-group-lbl">Live</span>${mkBtns(lv, "live")}</div>
  `;
  document.querySelectorAll(".season-btn").forEach(btn => {
    btn.addEventListener("click", () => selectSeason(btn.dataset.season));
  });
}

function renderChart() {
  const S = APP.seasons[state.season];
  const rawArr = S.ts[state.chartMode] || [];
  const n = rawArr.length;
  if (!n) return;
  const isPct = state.chartMode === "flips_pct";
  const isFppg = state.chartMode === "t15_vs";
  const arr = isFppg ? rawArr.map(v => v / 15) : rawArr;
  const baseRef = isPct ? 50 : 0;
  const svg = document.getElementById("edge-chart");
  const W = 1000, H = 110, padL = 56, padR = 16, padT = 13, padB = 20;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  let minV = Math.min(baseRef, ...arr);
  let maxV = Math.max(baseRef, ...arr);
  if (maxV - minV < (isPct ? 4 : (isFppg ? 0.5 : 1e-4))) {
    const pad = isPct ? 3 : (isFppg ? 0.5 : 1);
    maxV += pad; minV -= pad;
  }
  const margin = (maxV - minV) * 0.14;
  minV -= margin; maxV += margin;

  const xOf = (i) => padL + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
  const yOf = (v) => padT + ((maxV - v) / (maxV - minV)) * plotH;
  const y0 = yOf(baseRef);

  const fullPts = arr.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const curSlice = arr.slice(0, state.step + 1);
  const curPts = curSlice.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const curVal = arr[state.step];
  const curX = xOf(state.step);
  const curY = yOf(curVal);
  const strokeCol = curVal >= baseRef ? "#34d399" : "#fb7185";
  const fillCol = curVal >= baseRef ? "rgba(52,211,153,0.16)" : "rgba(251,113,133,0.14)";
  const areaPts = `${xOf(0).toFixed(1)},${y0.toFixed(1)} ${curPts} ${curX.toFixed(1)},${y0.toFixed(1)}`;
  const fmtTick = (v) => isPct ? `${v.toFixed(0)}%` : (isFppg ? fmtSigned(v, 1) : fmtSignedInt(v));
  const fmtCur = (v) => isPct ? `${v.toFixed(1)}%` : (isFppg ? `${fmtSigned(v, 1)} FPPG` : fmtSignedInt(v));
  const refLbl = isPct ? "50%" : "0";

  svg.innerHTML = `
    <line x1="${padL}" y1="${padT}" x2="${W - padR}" y2="${padT}" stroke="rgba(148,163,184,0.10)" stroke-width="1"/>
    <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="rgba(148,163,184,0.10)" stroke-width="1"/>
    <line x1="${padL}" y1="${y0.toFixed(1)}" x2="${W - padR}" y2="${y0.toFixed(1)}" stroke="rgba(148,163,184,0.32)" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="${padL - 8}" y="${padT + 9}" text-anchor="end" fill="#8494ad" font-size="10.5">${fmtTick(maxV)}</text>
    <text x="${padL - 8}" y="${(y0 + 3).toFixed(1)}" text-anchor="end" fill="#8494ad" font-size="10.5">${refLbl}</text>
    <text x="${padL - 8}" y="${H - padB}" text-anchor="end" fill="#8494ad" font-size="10.5">${fmtTick(minV)}</text>
    <polyline fill="none" stroke="rgba(148,163,184,0.28)" stroke-width="1.4" points="${fullPts}"/>
    <polygon fill="${fillCol}" points="${areaPts}"/>
    <polyline fill="none" stroke="${strokeCol}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" points="${curPts}"/>
    <line x1="${curX.toFixed(1)}" y1="${padT}" x2="${curX.toFixed(1)}" y2="${H - padB}" stroke="#38bdf8" stroke-width="1.2" stroke-dasharray="2,3"/>
    <circle cx="${curX.toFixed(1)}" cy="${curY.toFixed(1)}" r="4.5" fill="${strokeCol}" stroke="#0a101d" stroke-width="2"/>
    <text x="${Math.min(W - padR - 4, Math.max(padL + 6, curX + 9)).toFixed(1)}" y="${Math.max(padT + 12, curY - 7).toFixed(1)}" fill="${strokeCol}" font-size="11.5" font-weight="700">${fmtCur(curVal)}</text>
    <text x="${padL}" y="${H - 5}" fill="#8494ad" font-size="10.5">Game 1</text>
    <text x="${W - padR}" y="${H - 5}" text-anchor="end" fill="#8494ad" font-size="10.5">Game ${n}</text>
  `;
}

function renderAll() {
  const S = APP.seasons[state.season];
  const cum = getSeasonCum(state.season);
  const cp = S.checkpoints[state.step];

  document.getElementById("season-pill").textContent = S.season;
  const badge = document.getElementById("split-badge");
  const isLive = S.split === "live";
  badge.textContent = isLive ? "LIVE" : "PAST";
  badge.className = `split-badge ${isLive ? "live" : "past"}`;

  document.getElementById("header-meta").innerHTML =
    `<strong>G ${cp.g}/${S.max_step}</strong> &middot; ${cp.date} &middot; ${cp.pct}%`;
  document.getElementById("replay-clock").innerHTML =
    `Game ${cp.g} / ${S.max_step} <span>&middot; ${cp.date} &middot; ${cp.reg} NBA games</span>`;
  document.getElementById("slider-max-lbl").textContent = `G ${S.max_step}`;

  const s2All = S.s2_all.map(r => enrichRow(r, cum, cp));
  const s2Top150 = s2All.slice(0, 150);
  const mktTop150 = s2All.filter(r => r.mrk !== null && r.mrk <= 150).sort((a, b) => a.mrk - b.mrk);
  const deepPool = s2All.filter(r => r.mrk !== null && r.mrk > 150);
  const s1Top150 = S.s1_150.map(r => enrichRow(r, cum, cp));

  const s2_150_diff = s2Top150.reduce((a, r) => a + r.fpts, 0) - mktTop150.reduce((a, r) => a + r.fpts, 0);
  const s2_swaps = s2Top150.filter(r => r.mrk > 150).length;

  const byE = [...mktTop150].sort((a, b) => b.e - a.e);
  const t15 = byE.slice(0, 15);
  const f15 = byE.slice(-15);
  const t15Diff = t15.reduce((a, r) => a + r.fpts, 0) - f15.reduce((a, r) => a + r.fpts, 0);
  const t15VsAvg = t15.reduce((a, r) => a + r.act_vs_adp, 0) / 15;
  const f15VsAvg = f15.reduce((a, r) => a + r.act_vs_adp, 0) / 15;
  const t15VsSpread = t15VsAvg - f15VsAvg;

  let flipsTotal = 0, flipsWon = 0;
  for (let i = 0; i < mktTop150.length; i++) {
    for (let j = i + 1; j < mktTop150.length; j++) {
      if (mktTop150[j].pr > mktTop150[i].pr) {
        flipsTotal++;
        if (mktTop150[j].act_adj > mktTop150[i].act_adj) flipsWon++;
      }
    }
  }
  const flipsPct = flipsTotal ? (100 * flipsWon / flipsTotal) : 50;
  const curBands = computeBandsFromRows(s2All);
  renderSeasonHero(S, cp, curBands, flipsWon, flipsTotal, flipsPct);

  const posDeep = deepPool.filter(r => r.e > 0);
  const negDeep = deepPool.filter(r => r.e <= 0);
  const posDeepHit = posDeep.filter(r => r.in_t150).length;
  const negDeepHit = negDeep.filter(r => r.in_t150).length;
  const posDeepPct = posDeep.length ? (100 * posDeepHit / posDeep.length) : 0;
  const negDeepPct = negDeep.length ? (100 * negDeepHit / negDeep.length) : 0;
  const s1Cap = s1Top150.filter(r => r.in_t150).length;
  const mktCap = mktTop150.filter(r => r.in_t150).length;
  const s2Cap = s2Top150.filter(r => r.in_t150).length;

  document.getElementById("breakdown-teaser").innerHTML = `
    <span>Biggest Calls: <strong class="${t15VsSpread >= 0 ? "pos" : "neg"}">${fmtSigned(t15VsSpread, 1)} FPPG vs. ADP</strong></span>
    <span>vs. ADP Calls: <strong class="${flipsPct >= 50 ? "pos" : "neg"}">${flipsPct.toFixed(1)}% right</strong></span>
    <span>Late Sleepers: <strong class="${posDeepPct >= negDeepPct ? "pos" : "neg"}">${posDeepPct.toFixed(0)}% hit</strong> vs. ${negDeepPct.toFixed(0)}% passed on</span>
  `;

  const cards = [
    {
      key: "top150",
      label: "Our Top 150 vs. ADP Top 150",
      val: `${fmtSignedInt(s2_150_diff)} FPTS`,
      cls: s2_150_diff >= 0 ? "pos" : "neg",
      sub: `Our 150 players outscored ADP&rsquo;s 150 (<strong>${s2Cap}</strong> vs. <strong>${mktCap}</strong> finished Top 150)`,
    },
    {
      key: "t15",
      label: "15 Most Liked vs. 15 Most Faded",
      val: `${fmtSigned(t15VsSpread, 1)} FPPG vs. ADP`,
      cls: t15VsSpread >= 0 ? "pos" : "neg",
      sub: `Each player vs. their own draft slot: Liked <strong>${fmtSigned(t15VsAvg, 1)}</strong> vs. Faded <strong>${fmtSigned(f15VsAvg, 1)}</strong> FPPG`,
    },
    {
      key: "flips",
      label: "When We Disagree With ADP",
      val: `${flipsPct.toFixed(1)}% Right`,
      cls: flipsPct >= 50 ? "pos" : "neg",
      sub: `Picked the better player in <strong>${flipsWon.toLocaleString()}/${flipsTotal.toLocaleString()}</strong> disagreements (50% = coin flip)`,
    },
    {
      key: "sleepers",
      label: "Late-Round Sleepers (Picks 151&ndash;225)",
      val: `${posDeepPct.toFixed(0)}% Hit Rate`,
      cls: posDeepPct >= negDeepPct ? "pos" : "neg",
      sub: `Our sleeper picks reaching Top 150: <strong>${posDeepHit}/${posDeep.length} (${posDeepPct.toFixed(0)}%)</strong> &middot; Passed-on picks: only <strong>${negDeepHit}/${negDeep.length} (${negDeepPct.toFixed(0)}%)</strong>`,
    },
  ];
  document.getElementById("hero-cards").innerHTML = cards.map(c => {
    const isAct = state.compareBox === c.key;
    return `
    <div class="stat-box ${isAct ? "active" : ""}" data-compare="${c.key}">
      <div class="stat-head">
        <div class="stat-label">${c.label}</div>
        <span class="stat-hint">${isAct ? "Close &#9652;" : "Inspect &#9662;"}</span>
      </div>
      <div class="stat-val ${c.cls}">${c.val}</div>
      <div class="stat-sub">${c.sub}</div>
    </div>`;
  }).join("");

  renderCompareDrawer({
    s2Top150, mktTop150, t15, f15, posDeep, negDeep,
    s2_150_diff, t15Diff, t15VsAvg, f15VsAvg, t15VsSpread,
    flipsWon, flipsTotal, flipsPct,
    posDeepHit, negDeepHit, posDeepPct, negDeepPct,
  });

  document.getElementById("band-strip").innerHTML =
    `<strong>More Context:</strong>` +
    `<span>Pure Stats (0% ADP): <strong>${s1Cap}/150</strong> Top-150 hits</span>` +
    `<span class="sep">|</span>` +
    `<span>Top-150 Swaps vs. ADP: <strong>${s2_swaps}</strong> players</span>`;

  renderChart();

  const baseRows = state.tab === "stage2_top150" ? s2Top150 : (state.tab === "stage1_top150" ? s1Top150 : deepPool);
  const q = state.search.trim().toLowerCase();
  const filtered = baseRows.filter(r => {
    if (q && !(r.p.toLowerCase().includes(q) || r.pos.toLowerCase().includes(q))) return false;
    if (state.filter === "top15" && !(r.rk <= 15)) return false;
    if (state.filter === "pos" && !(r.e >= 0.25)) return false;
    if (state.filter === "neg" && !(r.e <= -0.25)) return false;
    if (state.filter === "rook" && r.t !== "ROOK") return false;
    return true;
  });

  const col = state.sortCol, dir = state.sortDir;
  filtered.sort((a, b) => {
    const va = col === "call_score" ? evalCall(a).score : a[col];
    const vb = col === "call_score" ? evalCall(b).score : b[col];
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    if (typeof va === "string") return dir * va.localeCompare(vb);
    return dir * (va - vb);
  });

  document.querySelectorAll("thead th").forEach(th => {
    const arrow = th.querySelector(".sort-arrow");
    if (!arrow) return;
    arrow.textContent = th.dataset.col === state.sortCol ? (state.sortDir === 1 ? "▲" : "▼") : "";
  });

  const hits = filtered.filter(r => evalCall(r).score === 1).length;
  const calls = filtered.filter(r => evalCall(r).score !== 0).length;
  const top150Now = filtered.filter(r => r.in_t150).length;

  document.getElementById("row-count").innerHTML =
    `<strong>${filtered.length}</strong> players &middot; <strong>${top150Now}</strong> Top-150`;
  document.getElementById("filtered-summary").innerHTML =
    calls ? `Calls: <strong>${hits}/${calls} (${(100 * hits / calls).toFixed(0)}%)</strong>` : "";

  document.getElementById("board-body").innerHTML = filtered.map(r => {
    const call = evalCall(r);
    const typeTag = r.t === "ROOK" ? '<span class="tag tag-rook">ROOK</span>' : (r.t === "RET" ? '<span class="tag tag-ret">RET</span>' : "");
    const injTag = r.al ? `<span class="tag tag-inj">${r.al}</span>` : "";
    const adpStr = r.mrk !== null
      ? `#${r.mrk}<span class="sub-num">(${fmtNum(r.adp, 1)})</span>`
      : '<span class="muted">UNR</span>';
    const deltaCls = r.drk === null ? "muted" : (r.drk > 0 ? "pos" : (r.drk < 0 ? "neg" : "muted"));
    const deltaStr = r.drk === null ? "&mdash;" : (r.drk > 0 ? "+" + r.drk : String(r.drk));
    const edgeCls = r.e >= 0.25 ? "edge-pos" : (r.e <= -0.25 ? "edge-neg" : "edge-flat");
    const actVsCls = r.act_vs_adp > 0.25 ? "pos" : (r.act_vs_adp < -0.25 ? "neg" : "muted");
    const gpCls = r.gp === 0 ? "neg" : "";
    const projTip = r.mp !== null ? `ADP Proj: ${fmtNum(r.mp, 2)}` : "";
    const actTip = `FPPG: ${fmtNum(r.act_fppg, 1)} | Act Adj: ${fmtNum(r.act_adj, 2)}`;

    return `<tr>
      <td><strong>#${r.rk}</strong></td>
      <td class="left player-cell">${r.p}${typeTag}${injTag}</td>
      <td class="left muted">${r.pos}</td>
      <td>${adpStr}</td>
      <td class="${deltaCls}">${deltaStr}</td>
      <td><span class="edge-pill ${edgeCls}">${fmtSigned(r.e, 2)}</span></td>
      <td title="${projTip}"><strong>${fmtNum(r.pr, 1)}</strong></td>
      <td class="group-sep ${gpCls}">${r.gp}</td>
      <td>${fmtInt(r.fpts)}</td>
      <td title="${actTip}">#${r.act_rank}</td>
      <td class="${actVsCls}">${fmtSigned(r.act_vs_adp, 2)}</td>
      <td><span class="call-badge ${call.cls}">${call.label}</span></td>
    </tr>`;
  }).join("");
}

function stopPlayback() {
  if (state.timer !== null) { clearInterval(state.timer); state.timer = null; }
  const btn = document.getElementById("play-btn");
  btn.classList.remove("playing");
  btn.innerHTML = "&#9654; Play";
}

function startPlayback() {
  const S = APP.seasons[state.season];
  if (state.step >= S.max_step - 1) {
    state.step = 0;
    document.getElementById("game-slider").value = 1;
    renderAll();
  }
  const btn = document.getElementById("play-btn");
  btn.classList.add("playing");
  btn.innerHTML = "&#10074;&#10074; Pause";
  state.timer = setInterval(() => {
    const curS = APP.seasons[state.season];
    if (state.step >= curS.max_step - 1) { stopPlayback(); return; }
    state.step++;
    document.getElementById("game-slider").value = state.step + 1;
    renderAll();
  }, state.speedMs);
}

function selectSeason(seasonKey) {
  stopPlayback();
  state.season = seasonKey;
  const S = APP.seasons[seasonKey];
  state.step = S.max_step - 1;
  const slider = document.getElementById("game-slider");
  slider.min = 1;
  slider.max = S.max_step;
  slider.value = S.max_step;
  renderSeasonBar();
  renderAll();
}

document.querySelectorAll(".info-pill").forEach(btn => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.info;
    const drawer = document.getElementById("info-drawer");
    if (state.activeInfo === key) {
      state.activeInfo = null;
      btn.classList.remove("active");
      drawer.className = "info-drawer";
      return;
    }
    state.activeInfo = key;
    document.querySelectorAll(".info-pill").forEach(b => b.classList.toggle("active", b.dataset.info === key));
    const info = getInfoContent(key);
    drawer.innerHTML = info.html;
    drawer.className = `info-drawer open ${info.tone}`;
  });
});

document.getElementById("breakdown-toggle").addEventListener("click", () => {
  const body = document.getElementById("breakdown-body");
  const open = body.classList.toggle("open");
  document.getElementById("breakdown-toggle").classList.toggle("is-open", open);
  document.getElementById("breakdown-btn").innerHTML = open ? "Hide &#9652;" : "Season Breakdown &#9662;";
});
document.getElementById("hero-cards").addEventListener("click", e => {
  const box = e.target.closest(".stat-box[data-compare]");
  if (!box) return;
  const k = box.dataset.compare;
  state.compareBox = (state.compareBox === k) ? null : k;
  renderAll();
});
document.getElementById("compare-drawer").addEventListener("click", e => {
  if (e.target.closest(".compare-close")) {
    state.compareBox = null;
    renderAll();
  }
});

document.getElementById("play-btn").addEventListener("click", () => {
  if (state.timer !== null) stopPlayback();
  else startPlayback();
});
document.getElementById("reset-btn").addEventListener("click", () => {
  stopPlayback(); state.step = 0; document.getElementById("game-slider").value = 1; renderAll();
});
document.getElementById("end-btn").addEventListener("click", () => {
  stopPlayback();
  const S = APP.seasons[state.season];
  state.step = S.max_step - 1;
  document.getElementById("game-slider").value = S.max_step;
  renderAll();
});
document.getElementById("game-slider").addEventListener("input", e => {
  stopPlayback();
  state.step = Math.max(0, parseInt(e.target.value, 10) - 1);
  renderAll();
});
document.querySelectorAll("[data-speed]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-speed]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.speedMs = parseInt(btn.dataset.speed, 10);
    if (state.timer !== null) { stopPlayback(); startPlayback(); }
  });
});
document.querySelectorAll("[data-chart]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-chart]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.chartMode = btn.dataset.chart;
    renderChart();
  });
});
document.getElementById("edge-chart").addEventListener("click", e => {
  stopPlayback();
  const rect = e.currentTarget.getBoundingClientRect();
  const xFrac = Math.max(0, Math.min(1, (e.clientX - rect.left - rect.width * 0.056) / (rect.width * 0.928)));
  const S = APP.seasons[state.season];
  state.step = Math.round(xFrac * (S.max_step - 1));
  document.getElementById("game-slider").value = state.step + 1;
  renderAll();
});
document.getElementById("guide-toggle").addEventListener("click", e => {
  const drawer = document.getElementById("guide-drawer");
  const open = drawer.classList.toggle("open");
  e.currentTarget.classList.toggle("active", open);
});
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.tab = btn.dataset.tab;
    state.sortCol = state.tab === "deep_sleepers" ? "e" : "rk";
    state.sortDir = state.tab === "deep_sleepers" ? -1 : 1;
    renderAll();
  });
});
document.querySelectorAll(".pill[data-filter]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".pill[data-filter]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.filter = btn.dataset.filter;
    renderAll();
  });
});
document.getElementById("search").addEventListener("input", e => { state.search = e.target.value; renderAll(); });
document.querySelectorAll("thead th").forEach(th => {
  th.addEventListener("click", () => {
    const col = th.dataset.col;
    if (!col) return;
    if (state.sortCol === col) state.sortDir *= -1;
    else {
      state.sortCol = col;
      state.sortDir = (col === "rk" || col === "mrk" || col === "adp" || col === "act_rank" || col === "p" || col === "pos") ? 1 : -1;
    }
    renderAll();
  });
});

renderMultiYearHero();
selectSeason(APP.default_season);
</script>
</body>
</html>
"""
)


CAT9_HTML_TEMPLATE = (
    r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Draft Model &mdash; 9-Cat</title>
<style>"""
    + SHARED_DARK_CSS.replace("__MAX_WIDTH__", "1360").replace("__TABLE_FS__", "12.5").replace("__TH_PAD__", "7")
    + r"""</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand-row">
      <span class="brand-dot"></span>
      <h1>The Draft Model</h1>
      <span class="fmt-pill">9-CAT &middot; ROTO + H2H</span>
      <span class="season-pill" id="season-pill">2026&ndash;27</span>
      <span class="split-badge live" id="split-badge">LIVE</span>
    </div>
    <div class="header-meta" id="header-meta"></div>
  </header>

  <section class="info-strip">
    <div class="info-pills">
      <button class="info-pill" data-info="why"><span class="dot accent"></span>Why This Exists <span class="chev">&#9662;</span></button>
      <button class="info-pill" data-info="what"><span class="dot pos"></span>What This Is <span class="chev">&#9662;</span></button>
      <button class="info-pill" data-info="not"><span class="dot neg"></span>Not Covered <span class="chev">&#9662;</span></button>
      <button class="info-pill" data-info="edge"><span class="dot amber"></span>How Edge Is Measured <span class="chev">&#9662;</span></button>
    </div>
    <div class="info-drawer" id="info-drawer"></div>
  </section>

  <section class="edge-hero" id="edge-hero">
    <div id="edge-hero-multi"></div>
    <div class="edge-hero-div" id="edge-hero-season"></div>
  </section>

  <section class="replay-panel">
    <div class="season-strip">
      <div class="season-group"><span class="season-group-lbl">Past Seasons</span><span id="btns-past"></span></div>
      <div class="season-group"><span class="season-group-lbl">Live</span><span id="btns-live"></span></div>
    </div>
    <div class="replay-top">
      <div class="replay-controls">
        <button class="play-btn" id="play-btn">&#9654; Play 4x</button>
        <button class="mini-btn" id="btn-start">G1</button>
        <button class="mini-btn" id="btn-end">End</button>
        <button class="mini-btn active" data-speed="250">4&times;</button>
        <button class="mini-btn" data-speed="125">8&times;</button>
      </div>
      <div class="replay-clock" id="replay-clock"></div>
    </div>
    <div class="slider-row">
      <span class="slider-end">G1</span>
      <input type="range" id="game-slider" min="0" max="13" value="13">
      <span class="slider-end right" id="slider-max-lbl">G14</span>
    </div>
    <div class="chart-header">
      <div class="chart-title">Season Trajectory (click chart to jump)</div>
      <div class="chart-modes">
        <button class="mini-btn active" data-chart="t15_vs">15 Most Liked vs. Faded</button>
        <button class="mini-btn" data-chart="top150_z">Top 150 Swaps (Z)</button>
        <button class="mini-btn" data-chart="flips_pct">% Calls Right</button>
      </div>
    </div>
    <svg id="edge-chart" viewBox="0 0 1000 110" preserveAspectRatio="none"></svg>

    <div class="breakdown-bar is-open" id="breakdown-toggle">
      <span class="breakdown-title-open">Season Breakdown</span>
      <div class="breakdown-teaser" id="breakdown-teaser"></div>
      <span class="breakdown-btn" id="breakdown-btn">Hide &#9652;</span>
    </div>
    <div class="breakdown-body open" id="breakdown-body">
      <section class="stats-grid" id="hero-cards">
        <div class="stat-box" data-compare="top150">
          <div class="stat-head">
            <div class="stat-label">Our Top 150 vs. ADP Top 150</div>
            <span class="stat-hint" data-hint="top150">Inspect &#9662;</span>
          </div>
          <div class="stat-val" id="kpi-top150">&mdash;</div>
          <div class="stat-sub" id="kpi-top150-sub">&mdash;</div>
        </div>
        <div class="stat-box" data-compare="t15">
          <div class="stat-head">
            <div class="stat-label">15 Most Liked vs. 15 Most Faded</div>
            <span class="stat-hint" data-hint="t15">Inspect &#9662;</span>
          </div>
          <div class="stat-val" id="kpi-t15">&mdash;</div>
          <div class="stat-sub" id="kpi-t15-sub">&mdash;</div>
        </div>
        <div class="stat-box" data-compare="flips">
          <div class="stat-head">
            <div class="stat-label">When We Disagree With ADP</div>
            <span class="stat-hint" data-hint="flips">Inspect &#9662;</span>
          </div>
          <div class="stat-val" id="kpi-flips">&mdash;</div>
          <div class="stat-sub" id="kpi-flips-sub">&mdash;</div>
        </div>
        <div class="stat-box" data-compare="sleepers">
          <div class="stat-head">
            <div class="stat-label">Late-Round Sleepers (Picks 151&ndash;225)</div>
            <span class="stat-hint" data-hint="sleepers">Inspect &#9662;</span>
          </div>
          <div class="stat-val" id="kpi-sleepers">&mdash;</div>
          <div class="stat-sub" id="kpi-sleepers-sub">&mdash;</div>
        </div>
      </section>
      <div class="compare-drawer" id="compare-drawer"></div>
    </div>
  </section>

  <div class="controls">
    <div class="tabs">
      <button class="tab-btn active" data-tab="top150">Top 150</button>
      <button class="tab-btn" data-tab="targets">Targets</button>
      <button class="tab-btn" data-tab="fades">Fades</button>
      <button class="tab-btn" data-tab="sleepers">Sleepers (151&ndash;225)</button>
      <button class="tab-btn" data-tab="all">All 225</button>
    </div>
    <div class="filters">
      <input type="text" id="search" placeholder="Search player / pos...">
      <button class="pill" id="punt-toggle">Punt &#9662;</button>
      <button class="pill active" id="cat-mode-btn" title="Switch category columns between Projected Z and Actual Z">Cats: Proj</button>
      <button class="pill" data-filter="hits">Hits</button>
      <button class="pill" data-filter="misses">Misses</button>
      <button class="pill" data-filter="rookies">Rookies</button>
      <button class="pill" id="guide-toggle">? Columns</button>
    </div>
  </div>

  <div class="drawer-panel" id="punt-drawer">
    <div class="punt-row">
      <span class="punt-lbl">Punt Categories:</span>
      <span id="punt-chips" style="display:inline-flex;flex-wrap:wrap;gap:6px"></span>
      <button class="punt-reset" id="punt-reset">Reset Balanced</button>
      <span class="punt-hint" id="punt-hint">Click any category to punt &amp; re-rank instantly</span>
    </div>
  </div>

  <div class="drawer-panel" id="guide-drawer">
    <div class="guide-items">
      <span><code>&Delta;Rk</code> ADP rank minus model rank</span>
      <span><code>Proj Z</code> Pre-draft 9-Cat Z-score (0.75&times; TO)</span>
      <span><code>Edge</code> Projected Z-score gain vs. ADP peers</span>
      <span><code>PTS..TO</code> Category Z-scores (green = strength, red = drag)</span>
      <span><code>vs ADP</code> Actual 9-Cat Z minus ADP expectation</span>
      <span><code>Call</code> Grades |Edge| &ge; 0.30 or |&Delta;Rk| &ge; 3</span>
    </div>
  </div>

  <div class="table-meta">
    <div id="row-count">Showing 150 players</div>
    <div class="inline-key">
      <span class="call-badge call-hit">TGT &#10003;</span><span class="call-badge call-miss">TGT &#10007;</span> Target
      <span class="call-badge call-hit">FADE &#10003;</span><span class="call-badge call-miss">FADE &#10007;</span> Fade
      <span class="call-badge call-agree">AGREE</span> Matches ADP
    </div>
  </div>

  <div class="table-wrap">
    <table id="board-table">
      <thead><tr id="thead-row"></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>

  <footer>
    <span>The Draft Model &middot; Strict out-of-sample 9-Cat projections (0.75&times; TO, Roto + H2H)</span>
    <span id="footer-edge-summary"></span>
  </footer>
</div>

<script>
const APP = __PAYLOAD_JSON__;
const CATS = APP.cat_names;
const BASE_W = [1, 1, 1, 1, 1, 1, 1, 1, APP.to_weight];
const PICK_BANDS = [
  { lbl: "Picks 1\u201325", lo: 1, hi: 25 },
  { lbl: "Picks 26\u201360", lo: 26, hi: 60 },
  { lbl: "Picks 61\u2013100", lo: 61, hi: 100 },
  { lbl: "Picks 101\u2013150", lo: 101, hi: 150 },
  { lbl: "Picks 151\u2013225", lo: 151, hi: 225 },
];
let multiYearCache = null;

let state = {
  season: APP.default_season,
  step: 0,
  tab: "top150",
  filter: "all",
  search: "",
  sortKey: "dyn_rk",
  sortDir: 1,
  chartMode: "t15_vs",
  catView: "proj",
  punted: new Set(),
  playing: false,
  speedMs: 250,
  timer: null,
  activeInfo: null,
  compareBox: null,
};

function fmtSigned(v, dp = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "&mdash;";
  return (v >= 0 ? "+" : "") + v.toFixed(dp);
}

function interpAnchor(anchors, adp) {
  const rk = APP.anchor_ranks;
  if (!anchors || !anchors.length) return 0;
  if (adp <= rk[0]) return anchors[0];
  if (adp >= rk[rk.length - 1]) return anchors[anchors.length - 1];
  for (let i = 0; i < rk.length - 1; i++) {
    if (adp >= rk[i] && adp <= rk[i + 1]) {
      const f = (adp - rk[i]) / (rk[i + 1] - rk[i]);
      return anchors[i] + f * (anchors[i + 1] - anchors[i]);
    }
  }
  return anchors[anchors.length - 1];
}

function estimateRankFromAnchors(anchors, zVal) {
  const rk = APP.anchor_ranks;
  if (!anchors || !anchors.length) return 251;
  if (zVal >= anchors[0]) return 1;
  const last = anchors.length - 1;
  if (zVal <= anchors[last]) return rk[last] + 1;
  for (let i = 0; i < last; i++) {
    const v0 = anchors[i], v1 = anchors[i + 1];
    if (zVal <= v0 && zVal >= v1) {
      const span = Math.max(v0 - v1, 1e-6);
      const t = (v0 - zVal) / span;
      return Math.max(1, Math.round(rk[i] + t * (rk[i + 1] - rk[i])));
    }
  }
  return rk[last] + 1;
}

function activeWeights() {
  return BASE_W.map((w, i) => state.punted.has(i) ? 0 : w);
}

function getInfoContent(key) {
  if (key === "why") {
    return {
      tone: "",
      html: "Fantasy leagues are already <strong>well projected in general</strong>&mdash;consensus ADP gets most players roughly right, and throwing it away loses badly. The edge comes from repeatable <strong>human biases on a small group of players</strong>: overpaying for rookie hype, aging/injury-prone names, and flashy scoring while overlooking durable, efficient multi-category starters. Avoiding a few landmines and hitting a few undervalued targets builds league-winning teams.",
    };
  }
  if (key === "what") {
    return {
      tone: "tone-pos",
      html: "<strong>Balanced pre-draft 9-Cat projections</strong> built on top of ADP (Z-scores across all 9 categories, 0.75&times; TO) that we believe outperform consensus ADP on average in both Roto and H2H&mdash;backed by strict out-of-sample historical replay and live tracking below. Use <strong>Punt</strong> in the toolbar to re-rank instantly if you punt a category.",
    };
  }
  if (key === "not") {
    return {
      tone: "tone-neg",
      html: "Anything after draft day&mdash;<strong>no</strong> waiver streaming, trades, mid-season injuries, or rotation changes. This is strictly a pre-draft projection to help you leave the draft with a stronger roster; managing the season is up to you.",
    };
  }
  return {
    tone: "tone-amber",
    html: "Instead of tying results to one fixed league size, we grade every pre-draft disagreement with ADP across <strong>five pick ranges (1&ndash;25 through 151&ndash;225)</strong>. <strong>Calls Right %</strong> asks: when we rank Player A above Player B against ADP, how often does A finish with higher 9-Cat value? (50% = coin flip.) <strong>Tgt vs. Fade</strong> compares how much more 9-Cat Z-score our liked targets produced than our fades drafted in the exact same pick range.",
  };
}

function buildSeasonButtons() {
  const pastEl = document.getElementById("btns-past");
  const liveEl = document.getElementById("btns-live");
  pastEl.innerHTML = "";
  liveEl.innerHTML = "";
  for (const item of APP.season_order) {
    const isLive = item.split === "live";
    const btn = document.createElement("button");
    btn.className = "season-btn " + (isLive ? "live" : "past") + (item.season === state.season ? " active" : "");
    btn.textContent = item.season;
    btn.onclick = () => selectSeason(item.season);
    (isLive ? liveEl : pastEl).appendChild(btn);
  }
}

function buildPuntChips() {
  const wrap = document.getElementById("punt-chips");
  wrap.innerHTML = "";
  CATS.forEach((name, idx) => {
    const btn = document.createElement("button");
    btn.className = "punt-chip" + (state.punted.has(idx) ? " punted" : "");
    btn.textContent = name;
    btn.onclick = () => {
      if (state.punted.has(idx)) state.punted.delete(idx);
      else if (state.punted.size < 4) state.punted.add(idx);
      buildPuntChips();
      renderTable();
    };
    wrap.appendChild(btn);
  });
  const hint = document.getElementById("punt-hint");
  const toggleBtn = document.getElementById("punt-toggle");
  if (state.punted.size === 0) {
    hint.innerHTML = "Active: <strong>Balanced 9-Cat</strong>";
    toggleBtn.innerHTML = "Punt &#9662;";
    toggleBtn.classList.remove("punt-on");
  } else {
    const names = Array.from(state.punted).sort().map(i => CATS[i]).join(", ");
    hint.innerHTML = `Punting: <strong style="color:var(--neg)">${names}</strong>`;
    toggleBtn.innerHTML = `Punt: ${names} &#9662;`;
    toggleBtn.classList.add("punt-on");
  }
}

function stopPlay() {
  state.playing = false;
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  const pb = document.getElementById("play-btn");
  pb.innerHTML = state.speedMs === 125 ? "&#9654; Play 8x" : "&#9654; Play 4x";
  pb.classList.remove("playing");
}

function togglePlay() {
  if (state.playing) { stopPlay(); return; }
  const S = APP.seasons[state.season];
  if (state.step >= S.max_step - 1) state.step = 0;
  state.playing = true;
  const pb = document.getElementById("play-btn");
  pb.innerHTML = "&#10074;&#10074; Pause";
  pb.classList.add("playing");
  renderStep();
  state.timer = setInterval(() => {
    const curS = APP.seasons[state.season];
    if (state.step >= curS.max_step - 1) { stopPlay(); return; }
    state.step += 1;
    renderStep();
  }, state.speedMs);
}

function computeSeasonBands(S, stepIdx) {
  const cp = S.checkpoints[stepIdx];
  return PICK_BANDS.map(b => {
    const sub = [];
    for (const r of S.rows) {
      if (r.mrk < b.lo || r.mrk > b.hi) continue;
      const act = S.zs[r.id][stepIdx];
      const slot = interpAnchor(cp.anchors, r.adp);
      sub.push({ mrk: r.mrk, pr: r.pr, e: r.e, drk: r.drk, act, vs: act - slot });
    }
    sub.sort((a, c) => a.mrk - c.mrk);
    let wins = 0, pairs = 0, tgtSum = 0, tgtCnt = 0, fadeSum = 0, fadeCnt = 0;
    for (let i = 0; i < sub.length; i++) {
      for (let j = i + 1; j < sub.length; j++) {
        if (sub[j].pr > sub[i].pr) {
          pairs++;
          if (sub[j].act > sub[i].act) wins++;
        }
      }
      const r = sub[i];
      const big = Math.abs(r.e) >= 0.30 || Math.abs(r.drk) >= 3;
      if (big) {
        if (r.e > 0) { tgtSum += r.vs; tgtCnt++; }
        else { fadeSum += r.vs; fadeCnt++; }
      }
    }
    const pct = pairs > 0 ? (100 * wins / pairs) : null;
    const gapZ = (tgtCnt > 0 && fadeCnt > 0) ? ((tgtSum / tgtCnt) - (fadeSum / fadeCnt)) : null;
    return { lbl: b.lbl, wins, pairs, pct, tgtSum, tgtCnt, fadeSum, fadeCnt, gapZ };
  });
}

function getMultiYearStats() {
  if (multiYearCache) return multiYearCache;
  const past = APP.season_order.filter(o => o.split !== "live");
  const agg = PICK_BANDS.map(b => ({ lbl: b.lbl, wins: 0, pairs: 0, tgtSum: 0, tgtCnt: 0, fadeSum: 0, fadeCnt: 0 }));
  let totalWins = 0, totalPairs = 0, winYrs = 0;
  for (const item of past) {
    const S = APP.seasons[item.season];
    const lastStep = S.max_step - 1;
    const bands = computeSeasonBands(S, lastStep);
    bands.forEach((b, idx) => {
      agg[idx].wins += b.wins;
      agg[idx].pairs += b.pairs;
      agg[idx].tgtSum += b.tgtSum;
      agg[idx].tgtCnt += b.tgtCnt;
      agg[idx].fadeSum += b.fadeSum;
      agg[idx].fadeCnt += b.fadeCnt;
    });
    const mkt150 = S.rows.filter(r => r.mrk <= 150).map(r => ({ mrk: r.mrk, pr: r.pr, act: S.zs[r.id][lastStep] })).sort((a, b) => a.mrk - b.mrk);
    let sw = 0, st = 0;
    for (let i = 0; i < mkt150.length; i++) {
      for (let j = i + 1; j < mkt150.length; j++) {
        if (mkt150[j].pr > mkt150[i].pr) {
          st++;
          if (mkt150[j].act > mkt150[i].act) sw++;
        }
      }
    }
    totalWins += sw;
    totalPairs += st;
    if (st > 0 && sw * 2 >= st) winYrs++;
  }
  const bands = agg.map(b => {
    const pct = b.pairs > 0 ? (100 * b.wins / b.pairs) : null;
    const gapZ = (b.tgtCnt > 0 && b.fadeCnt > 0) ? ((b.tgtSum / b.tgtCnt) - (b.fadeSum / b.fadeCnt)) : null;
    return { ...b, pct, gapZ };
  });
  multiYearCache = {
    nYrs: past.length,
    winYrs,
    totalWins,
    totalPairs,
    overallPct: totalPairs > 0 ? (100 * totalWins / totalPairs) : 50,
    bands,
  };
  return multiYearCache;
}

function renderBandTableHtml(bands) {
  const head = `<div class="band-head"><span>Draft Range</span><span class="c-mid">vs. 50% Coin Flip</span><span class="c-r">Right</span><span class="c-r">Tgt vs Fade</span></div>`;
  const rows = bands.map(b => {
    let fillHtml = "";
    let pctCls = "muted", pctStr = "&mdash;";
    if (b.pct !== null) {
      pctCls = b.pct >= 50 ? "pos" : "neg";
      pctStr = `${b.pct.toFixed(1)}%`;
      const w = Math.min(50, (Math.abs(b.pct - 50) / 20) * 50);
      const style = b.pct >= 50 ? `left:50%;width:${w.toFixed(1)}%` : `left:${(50 - w).toFixed(1)}%;width:${w.toFixed(1)}%`;
      fillHtml = `<div class="band-fill ${pctCls}" style="${style}"></div>`;
    }
    const gapCls = b.gapZ === null ? "muted" : (b.gapZ >= 0 ? "pos" : "neg");
    const gapStr = b.gapZ === null ? "&mdash;" : `${b.gapZ >= 0 ? "+" : ""}${b.gapZ.toFixed(2)} Z`;
    const tip = `${b.wins.toLocaleString()}/${b.pairs.toLocaleString()} disagreements right (${b.tgtCnt} targets vs ${b.fadeCnt} fades)`;
    return `<div class="band-row" title="${tip}">
      <span class="band-lbl">${b.lbl}</span>
      <div class="band-track"><div class="band-midline"></div>${fillHtml}</div>
      <span class="band-pct ${pctCls}">${pctStr}</span>
      <span class="band-gap ${gapCls}">${gapStr}</span>
    </div>`;
  }).join("");
  return `<div class="band-table">${head}${rows}</div>`;
}

function renderMultiYearHero() {
  const my = getMultiYearStats();
  const el = document.getElementById("edge-hero-multi");
  if (!el || !my.nYrs) return;
  const cls = my.overallPct >= 50 ? "pos" : "neg";
  const yrCls = my.winYrs * 2 >= my.nYrs ? "pos" : "neg";
  el.innerHTML = `
    <div class="edge-hero-lbl">${my.nYrs}-Year Track Record &middot; All League Sizes</div>
    <div class="edge-hero-big">
      <span class="edge-hero-num ${cls}">${my.overallPct.toFixed(1)}%</span>
      <span class="edge-hero-unit">Calls Right</span>
      <span class="edge-hero-pill ${yrCls}" title="Completed seasons where overall Picks 1&ndash;150 Calls Right finished above 50%">Beat ADP ${my.winYrs}/${my.nYrs} Yrs</span>
    </div>
    <div class="edge-hero-sub">Picks 1&ndash;150 overall (<strong>${my.totalWins.toLocaleString()}/${my.totalPairs.toLocaleString()}</strong> right &middot; 50% = coin flip)</div>
    ${renderBandTableHtml(my.bands)}
  `;
  const ft = document.getElementById("footer-edge-summary");
  const deep = my.bands[4];
  const deepStr = deep && deep.gapZ !== null ? ` &middot; ${deep.gapZ >= 0 ? "+" : ""}${deep.gapZ.toFixed(2)} Z on Picks 151&ndash;225` : "";
  if (ft) {
    ft.innerHTML = `${my.nYrs}-Yr Out-of-Sample: ${my.overallPct.toFixed(1)}% Calls Right (1&ndash;150)${deepStr} &middot; Beat ADP ${my.winYrs}/${my.nYrs} yrs`;
  }
}

function renderSeasonHero(S, stepIdx, flipsWon, flipsTotal, flipsPct) {
  const el = document.getElementById("edge-hero-season");
  if (!el) return;
  const cp = S.checkpoints[stepIdx];
  const isLive = S.split === "live";
  const cls = flipsPct >= 50 ? "pos" : "neg";
  const pill = isLive ? `${cp.g} Equalized G` : (cp.g === S.max_step ? "Full Season" : `Through G ${cp.g}`);
  const curBands = computeSeasonBands(S, stepIdx);
  el.innerHTML = `
    <div class="edge-hero-lbl">Selected Season &middot; ${S.season} (${isLive ? "LIVE" : "PAST"} &middot; G ${cp.g}/${S.max_step})</div>
    <div class="edge-hero-big">
      <span class="edge-hero-num ${cls}">${flipsPct.toFixed(1)}%</span>
      <span class="edge-hero-unit">Calls Right</span>
      <span class="edge-hero-pill">${pill}</span>
    </div>
    <div class="edge-hero-sub">Picks 1&ndash;150: <strong>${flipsWon.toLocaleString()}/${flipsTotal.toLocaleString()}</strong> disagreements right &middot; Scrub slider to replay</div>
    ${renderBandTableHtml(curBands)}
  `;
}

function selectSeason(seasonKey) {
  stopPlay();
  state.season = seasonKey;
  const S = APP.seasons[seasonKey];
  state.step = S.max_step - 1;
  buildSeasonButtons();

  document.getElementById("season-pill").textContent = seasonKey;
  const badge = document.getElementById("split-badge");
  const isLive = S.split === "live";
  badge.className = "split-badge " + (isLive ? "live" : "past");
  badge.textContent = isLive ? "LIVE" : "PAST";

  const slider = document.getElementById("game-slider");
  slider.min = 0;
  slider.max = S.max_step - 1;
  slider.value = state.step;
  document.getElementById("slider-max-lbl").textContent = `G${S.max_step}`;

  renderStep();
}

function drawChart() {
  const S = APP.seasons[state.season];
  const svg = document.getElementById("edge-chart");
  const arr = S.ts[state.chartMode] || [];
  const n = arr.length;
  if (!n) { svg.innerHTML = ""; return; }

  const isPct = state.chartMode === "flips_pct";
  const baseRef = isPct ? 50 : 0;
  const W = 1000, H = 110, padL = 52, padR = 16, padT = 13, padB = 20;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  let minV = Math.min(baseRef, ...arr);
  let maxV = Math.max(baseRef, ...arr);
  if (maxV - minV < (isPct ? 4 : 0.05)) { maxV += (isPct ? 3 : 0.25); minV -= (isPct ? 3 : 0.25); }
  const margin = (maxV - minV) * 0.14;
  minV -= margin; maxV += margin;

  const xOf = i => padL + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
  const yOf = v => padT + ((maxV - v) / (maxV - minV)) * plotH;
  const y0 = yOf(baseRef);

  const fullPts = arr.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const curSlice = arr.slice(0, state.step + 1);
  const curPts = curSlice.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const curV = arr[state.step];
  const cx = xOf(state.step), cy = yOf(curV);
  const col = curV >= baseRef ? "#34d399" : "#fb7185";
  const fill = curV >= baseRef ? "rgba(52,211,153,0.16)" : "rgba(251,113,133,0.14)";
  const area = `${xOf(0).toFixed(1)},${y0.toFixed(1)} ${curPts} ${cx.toFixed(1)},${y0.toFixed(1)}`;
  const fmtTick = v => isPct ? `${v.toFixed(0)}%` : (v >= 0 ? "+" : "") + v.toFixed(1);
  const fmtCur = v => isPct ? `${v.toFixed(1)}%` : (v >= 0 ? "+" : "") + v.toFixed(1);
  const refLbl = isPct ? "50%" : "0";

  svg.innerHTML = `
    <line x1="${padL}" y1="${padT}" x2="${W - padR}" y2="${padT}" stroke="rgba(148,163,184,0.10)" stroke-width="1"/>
    <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="rgba(148,163,184,0.10)" stroke-width="1"/>
    <line x1="${padL}" y1="${y0.toFixed(1)}" x2="${W - padR}" y2="${y0.toFixed(1)}" stroke="rgba(148,163,184,0.32)" stroke-dasharray="4,4" stroke-width="1"/>
    <text x="${padL - 8}" y="${padT + 9}" text-anchor="end" fill="#8494ad" font-size="10.5">${fmtTick(maxV)}</text>
    <text x="${padL - 8}" y="${(y0 + 3).toFixed(1)}" text-anchor="end" fill="#8494ad" font-size="10.5">${refLbl}</text>
    <text x="${padL - 8}" y="${H - padB}" text-anchor="end" fill="#8494ad" font-size="10.5">${fmtTick(minV)}</text>
    <polyline points="${fullPts}" fill="none" stroke="rgba(148,163,184,0.28)" stroke-width="1.4"/>
    <polygon points="${area}" fill="${fill}"/>
    <polyline points="${curPts}" fill="none" stroke="${col}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="${cx.toFixed(1)}" y1="${padT}" x2="${cx.toFixed(1)}" y2="${H - padB}" stroke="#38bdf8" stroke-width="1.2" stroke-dasharray="2,3"/>
    <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="4.5" fill="${col}" stroke="#0a101d" stroke-width="2"/>
    <text x="${Math.min(W - padR - 4, Math.max(padL + 6, cx + 9)).toFixed(1)}" y="${Math.max(padT + 12, cy - 7).toFixed(1)}" fill="${col}" font-size="11.5" font-weight="700">${fmtCur(curV)}</text>
    <text x="${padL}" y="${H - 5}" fill="#8494ad" font-size="10.5">Game 1</text>
    <text x="${W - padR}" y="${H - 5}" text-anchor="end" fill="#8494ad" font-size="10.5">Game ${n}</text>
  `;
}

function renderKPIs() {
  const S = APP.seasons[state.season];
  const cp = S.checkpoints[state.step];
  const top150Z = S.ts.top150_z[state.step];
  const t15Vs = S.ts.t15_vs[state.step];
  const rVal = S.ts.corr_r[state.step];

  document.getElementById("header-meta").innerHTML =
    `<strong>G ${cp.g}/${S.max_step}</strong> &middot; ${cp.date} &middot; ${cp.pct.toFixed(0)}%`;
  document.getElementById("replay-clock").innerHTML =
    `Game ${cp.g} / ${S.max_step} <span>&middot; ${cp.date} &middot; ${cp.reg.toLocaleString()} NBA games</span>`;
  document.getElementById("game-slider").value = state.step;

  let modHits = 0, mktHits = 0;
  const mktTop150 = [];
  let posDeep = 0, posDeepHit = 0, negDeep = 0, negDeepHit = 0;
  for (const r of S.rows) {
    const az = S.zs[r.id][state.step];
    const gp = S.gp[r.id][state.step];
    const hit = az >= cp.t150_cut && gp > 0;
    if (r.rk <= 150 && hit) modHits++;
    if (r.mrk <= 150) {
      if (hit) mktHits++;
      mktTop150.push({ mrk: r.mrk, pr: r.pr, az });
    } else {
      if (r.e > 0) { posDeep++; if (hit) posDeepHit++; }
      else { negDeep++; if (hit) negDeepHit++; }
    }
  }
  mktTop150.sort((a, b) => a.mrk - b.mrk);
  let flipsTotal = 0, flipsWon = 0;
  for (let i = 0; i < mktTop150.length; i++) {
    for (let j = i + 1; j < mktTop150.length; j++) {
      if (mktTop150[j].pr > mktTop150[i].pr) {
        flipsTotal++;
        if (mktTop150[j].az > mktTop150[i].az) flipsWon++;
      }
    }
  }
  const flipsPct = flipsTotal ? (100 * flipsWon / flipsTotal) : 50;
  renderSeasonHero(S, state.step, flipsWon, flipsTotal, flipsPct);
  const posDeepPct = posDeep ? (100 * posDeepHit / posDeep) : 0;
  const negDeepPct = negDeep ? (100 * negDeepHit / negDeep) : 0;

  document.getElementById("breakdown-teaser").innerHTML = `
    <span>Biggest Calls: <strong class="${t15Vs >= 0 ? "pos" : "neg"}">${fmtSigned(t15Vs, 1)} Z</strong></span>
    <span>vs. ADP Calls: <strong class="${flipsPct >= 50 ? "pos" : "neg"}">${flipsPct.toFixed(1)}% right</strong></span>
    <span>Late Sleepers: <strong class="${posDeepPct >= negDeepPct ? "pos" : "neg"}">${posDeepPct.toFixed(0)}% hit</strong> vs. ${negDeepPct.toFixed(0)}% passed on</span>
  `;

  const k1 = document.getElementById("kpi-top150");
  k1.innerHTML = `${fmtSigned(top150Z, 1)} Z`;
  k1.className = "stat-val " + (top150Z >= 0 ? "pos" : "neg");
  document.getElementById("kpi-top150-sub").innerHTML = `Our 150 players beat ADP&rsquo;s 150 (<strong>${modHits}</strong> vs. <strong>${mktHits}</strong> finished Top 150)`;

  const k2 = document.getElementById("kpi-t15");
  k2.innerHTML = `${fmtSigned(t15Vs, 1)} Z`;
  k2.className = "stat-val " + (t15Vs >= 0 ? "pos" : "neg");
  document.getElementById("kpi-t15-sub").innerHTML = `Our 15 highest-edge targets minus 15 steepest fades below ADP`;

  const k3 = document.getElementById("kpi-flips");
  k3.innerHTML = `${flipsPct.toFixed(1)}% Right`;
  k3.className = "stat-val " + (flipsPct >= 50 ? "pos" : "neg");
  document.getElementById("kpi-flips-sub").innerHTML = `Picked the better player in <strong>${flipsWon.toLocaleString()}/${flipsTotal.toLocaleString()}</strong> disagreements (50% = coin flip)`;

  const k4 = document.getElementById("kpi-sleepers");
  k4.innerHTML = `${posDeepPct.toFixed(0)}% Hit Rate`;
  k4.className = "stat-val " + (posDeepPct >= negDeepPct ? "pos" : "neg");
  document.getElementById("kpi-sleepers-sub").innerHTML = `Our sleeper picks reaching Top 150: <strong>${posDeepHit}/${posDeep} (${posDeepPct.toFixed(0)}%)</strong> &middot; Passed-on picks: only <strong>${negDeepHit}/${negDeep} (${negDeepPct.toFixed(0)}%)</strong>`;

  document.querySelectorAll(".stat-box[data-compare]").forEach(b => {
    const isAct = state.compareBox === b.dataset.compare;
    b.classList.toggle("active", isAct);
    const h = b.querySelector(".stat-hint");
    if (h) h.innerHTML = isAct ? "Close &#9652;" : "Inspect &#9662;";
  });
  renderCat9CompareDrawer(S, cp, { top150Z, t15Vs, flipsWon, flipsTotal, flipsPct, posDeepHit, posDeep, posDeepPct, negDeepHit, negDeep, negDeepPct });
}

function buildCat9CompareCol(title, tone, rows, opts = {}) {
  const showCall = !!opts.showCall;
  const showT150 = !!opts.showT150;
  const totZ = rows.reduce((a, r) => a + r.act_z, 0);
  const totVs = rows.reduce((a, r) => a + r.vs_adp, 0);
  const avgZ = rows.length ? totZ / rows.length : 0;
  const avgGp = rows.length ? rows.reduce((a, r) => a + r.act_g, 0) / rows.length : 0;
  const avgAdp = rows.length ? rows.reduce((a, r) => a + r.mrk, 0) / rows.length : null;
  const avgFin = rows.length ? rows.reduce((a, r) => a + r.act_rank, 0) / rows.length : null;
  const hits150 = rows.filter(r => r.in_t150).length;
  const colMeta = showT150
    ? `<strong>${hits150}/${rows.length}</strong> Top-150 &middot; Avg <strong>${fmtSigned(avgZ, 2)}</strong> Z`
    : `Total <strong>${fmtSigned(totZ, 1)}</strong> Z &middot; Avg <strong>${fmtSigned(avgZ, 2)}</strong> Z`;
  const midHdr = showCall ? "ADP&rarr;Rk" : "Edge";
  const bodyHtml = rows.length ? rows.map(r => {
    const callTag = showCall ? `<span class="mini-call ${r.call_score === 1 ? "hit" : "miss"}">${r.call_label}</span>` : "";
    const t150Tag = (showT150 && r.in_t150) ? `<span class="t150-tag">T150</span>` : "";
    const midCell = showCall
      ? `<td class="${r.drk >= 0 ? "pos" : "neg"}">#${r.mrk}&rarr;#${r.rk}</td>`
      : `<td class="${r.e >= 0.30 ? "pos" : (r.e <= -0.30 ? "neg" : "muted")}">${fmtSigned(r.e, 2)}</td>`;
    const vsCls = r.vs_adp >= 0.5 ? "pos" : (r.vs_adp <= -0.5 ? "neg" : "muted");
    const finTxt = r.act_rank > 250 ? "250+" : `#${r.act_rank}`;
    const finCls = (r.act_rank <= r.mrk - 5) ? "pos" : ((r.act_rank >= r.mrk + 5) ? "neg" : "");
    return `<tr>
      <td class="left p-name">${r.p}${callTag}${t150Tag}</td>
      <td class="muted">#${r.mrk}</td>
      ${midCell}
      <td class="${r.act_g === 0 ? "neg" : "muted"}">${r.act_g}</td>
      <td><strong>${fmtSigned(r.act_z, 2)}</strong></td>
      <td class="${vsCls}">${fmtSigned(r.vs_adp, 2)}</td>
      <td class="${finCls}"><strong>${finTxt}</strong></td>
    </tr>`;
  }).join("") : `<tr><td colspan="7" class="left muted">No players in this group</td></tr>`;

  const avgAdpTxt = avgAdp !== null ? `#${Math.round(avgAdp)}` : "&mdash;";
  const avgFinTxt = avgFin !== null ? `#${Math.round(avgFin)}` : "&mdash;";
  const avgFinCls = (avgAdp !== null && avgFin !== null) ? (avgFin <= avgAdp ? "pos" : "neg") : "";

  return `<div class="compare-col">
    <div class="compare-col-head">
      <span><span class="c-dot ${tone}"></span>${title} (${rows.length})</span>
      <span class="compare-col-meta">${colMeta}</span>
    </div>
    <div class="compare-tbl-wrap">
      <table class="compare-tbl">
        <thead><tr>
          <th class="left">Player</th>
          <th>ADP</th>
          <th>${midHdr}</th>
          <th>GP</th>
          <th>Act Z</th>
          <th>vs ADP</th>
          <th>Finish</th>
        </tr></thead>
        <tbody>${bodyHtml}</tbody>
        <tfoot><tr>
          <td class="left">Total / Avg</td>
          <td class="muted">${avgAdpTxt}</td>
          <td class="muted">&mdash;</td>
          <td>${avgGp.toFixed(1)}</td>
          <td>${fmtSigned(avgZ, 2)} avg</td>
          <td class="${totVs >= 0 ? "pos" : "neg"}">${fmtSigned(totVs, 1)}</td>
          <td class="${avgFinCls}">${avgFinTxt}</td>
        </tr></tfoot>
      </table>
    </div>
  </div>`;
}

function renderCat9CompareDrawer(S, cp, ctx) {
  const drawer = document.getElementById("compare-drawer");
  if (!drawer) return;
  const key = state.compareBox;
  if (!key) {
    drawer.className = "compare-drawer";
    drawer.innerHTML = "";
    return;
  }

  const enriched = S.rows.map(r => {
    const act_g = S.gp[r.id][state.step];
    const act_z = S.zs[r.id][state.step];
    const slot_z = interpAnchor(cp.anchors, r.adp);
    const vs_adp = act_z - slot_z;
    const act_rank = act_g > 0 ? estimateRankFromAnchors(cp.anchors, act_z) : (APP.anchor_ranks[APP.anchor_ranks.length - 1] + 1);
    const in_t150 = act_z >= cp.t150_cut && act_g > 0;
    const big = Math.abs(r.e) >= 0.30 || Math.abs(r.drk) >= 3;
    let call_score = 0, call_label = "AGREE";
    if (big) {
      const liked = r.e !== 0 ? (r.e > 0) : (r.drk > 0);
      const hit = liked ? (vs_adp > 0) : (vs_adp < 0);
      call_score = hit ? 1 : -1;
      call_label = liked ? (hit ? "TGT \u2713" : "TGT \u2717") : (hit ? "FADE \u2713" : "FADE \u2717");
    }
    return { ...r, act_g, act_z, vs_adp, act_rank, in_t150, call_score, call_label };
  });

  let title = "", sub = "", leftHtml = "", rightHtml = "";
  if (key === "top150") {
    const swapsIn = enriched.filter(r => r.rk <= 150 && r.mrk > 150).sort((a, b) => b.act_z - a.act_z);
    const swapsOut = enriched.filter(r => r.mrk <= 150 && r.rk > 150).sort((a, b) => b.act_z - a.act_z);
    const inAvg = swapsIn.length ? swapsIn.reduce((a, r) => a + r.act_z, 0) / swapsIn.length : 0;
    const outAvg = swapsOut.length ? swapsOut.reduce((a, r) => a + r.act_z, 0) / swapsOut.length : 0;
    title = `Our Top 150 Swaps In vs. Dropped from ADP Top 150`;
    sub = `Net difference: <strong class="${ctx.top150Z >= 0 ? "pos" : "neg"}">${fmtSigned(ctx.top150Z, 1)} Z</strong> (Avg <strong>${fmtSigned(inAvg, 2)}</strong> vs. <strong>${fmtSigned(outAvg, 2)}</strong> Z/player) &middot; The other ${150 - swapsIn.length} players are on both lists`;
    leftHtml = buildCat9CompareCol("Added to Our Top 150", "pos", swapsIn, { showT150: true });
    rightHtml = buildCat9CompareCol("Dropped from ADP Top 150", "neg", swapsOut, { showT150: true });
  } else if (key === "t15") {
    const mkt150 = enriched.filter(r => r.mrk <= 150);
    const t15 = [...mkt150].sort((a, b) => b.e - a.e).slice(0, 15);
    const f15 = [...mkt150].sort((a, b) => a.e - b.e).slice(0, 15);
    const tAvg = t15.reduce((a, r) => a + r.vs_adp, 0) / 15;
    const fAvg = f15.reduce((a, r) => a + r.vs_adp, 0) / 15;
    title = `15 Most Liked Targets vs. 15 Most Faded Below ADP`;
    sub = `Targets vs. Fades (vs. ADP): <strong class="${ctx.t15Vs >= 0 ? "pos" : "neg"}">${fmtSigned(ctx.t15Vs, 1)} Z</strong> total (Avg <strong>${fmtSigned(tAvg, 2)}</strong> vs. <strong>${fmtSigned(fAvg, 2)}</strong> Z/player)`;
    leftHtml = buildCat9CompareCol("15 Most Liked (Highest Edge)", "pos", t15);
    rightHtml = buildCat9CompareCol("15 Most Faded (Steepest Below ADP)", "neg", f15);
  } else if (key === "flips") {
    const graded = enriched
      .filter(r => r.mrk <= 150 && r.call_score !== 0)
      .sort((a, b) => (Math.abs(b.drk) - Math.abs(a.drk)) || (Math.abs(b.e) - Math.abs(a.e)));
    const bigHits = graded.filter(r => r.call_score === 1).slice(0, 15);
    const bigMiss = graded.filter(r => r.call_score === -1).slice(0, 15);
    title = `Biggest Disagreements With ADP: Top Hits vs. Top Misses`;
    sub = `Overall Picks 1&ndash;150: <strong class="${ctx.flipsPct >= 50 ? "pos" : "neg"}">${ctx.flipsWon.toLocaleString()}/${ctx.flipsTotal.toLocaleString()} (${ctx.flipsPct.toFixed(1)}%)</strong> pairwise disagreements right &middot; Sorted by biggest rank shift vs. ADP`;
    leftHtml = buildCat9CompareCol("15 Biggest Calls Right", "pos", bigHits, { showCall: true });
    rightHtml = buildCat9CompareCol("15 Biggest Calls Missed", "neg", bigMiss, { showCall: true });
  } else if (key === "sleepers") {
    const deep = enriched.filter(r => r.mrk > 150);
    const posSorted = deep.filter(r => r.e > 0).sort((a, b) => b.act_z - a.act_z);
    const negSorted = deep.filter(r => r.e <= 0).sort((a, b) => b.act_z - a.act_z);
    title = `Late-Round Sleepers (Picks 151&ndash;225): Targeted vs. Passed On`;
    sub = `Targeted sleepers reaching Top 150: <strong class="pos">${ctx.posDeepHit}/${ctx.posDeep} (${ctx.posDeepPct.toFixed(0)}%)</strong> vs. Passed-on picks: <strong>${ctx.negDeepHit}/${ctx.negDeep} (${ctx.negDeepPct.toFixed(0)}%)</strong>`;
    leftHtml = buildCat9CompareCol("Our Sleeper Targets (Edge > 0)", "pos", posSorted, { showT150: true });
    rightHtml = buildCat9CompareCol("Passed-On Late Picks (Edge \u2264 0)", "neg", negSorted, { showT150: true });
  }

  drawer.className = "compare-drawer open";
  drawer.innerHTML = `
    <div class="compare-head">
      <div>
        <div class="compare-title">${title}</div>
        <div class="compare-sub">${sub}</div>
      </div>
      <button class="compare-close" type="button">&times; Close</button>
    </div>
    <div class="compare-grid">${leftHtml}${rightHtml}</div>
  `;
}

function catClass(v) {
  if (v >= 1.2) return "cat-cell cat-hi";
  if (v >= 0.35) return "cat-cell cat-midpos";
  if (v <= -1.2) return "cat-cell cat-lo";
  if (v <= -0.35) return "cat-cell cat-midneg";
  return "cat-cell muted";
}

function buildRowsWithPunt() {
  const S = APP.seasons[state.season];
  const cp = S.checkpoints[state.step];
  const w = activeWeights();
  const isPunted = state.punted.size > 0;

  const enriched = S.rows.map(r => {
    let projZ = r.pr;
    if (isPunted) {
      projZ = 0;
      for (let c = 0; c < 9; c++) projZ += r.pz[c] * w[c];
    }
    const actG = S.gp[r.id][state.step];
    const actZ = S.zs[r.id][state.step];
    const slotZ = interpAnchor(cp.anchors, r.adp);
    const vsAdp = actZ - slotZ;
    return { ...r, dyn_pr: projZ, act_g: actG, act_z: actZ, vs_adp: vsAdp };
  });

  const sorted = [...enriched].sort((a, b) => (b.dyn_pr - a.dyn_pr) || (a.mrk - b.mrk));
  const diffs150 = sorted.filter(r => r.mrk <= 150).map(r => r.dyn_pr - r.mp).sort((a, b) => a - b);
  const medDiff = diffs150.length ? diffs150[Math.floor(diffs150.length / 2)] : 0;

  sorted.forEach((r, i) => {
    r.dyn_rk = i + 1;
    r.dyn_drk = r.mrk - r.dyn_rk;
    r.dyn_e = isPunted ? (r.dyn_pr - r.mp - medDiff) : r.e;
    const bigEdge = Math.abs(r.dyn_e) >= 0.30;
    const bigShift = Math.abs(r.dyn_drk) >= 3;
    if (!bigEdge && !bigShift) {
      r.call_label = "AGREE";
      r.call_cls = "call-agree";
      r.call_score = 0;
    } else {
      const liked = r.dyn_e !== 0 ? (r.dyn_e > 0) : (r.dyn_drk > 0);
      const hit = liked ? (r.vs_adp > 0) : (r.vs_adp < 0);
      r.call_label = liked ? (hit ? "TGT \u2713" : "TGT \u2717") : (hit ? "FADE \u2713" : "FADE \u2717");
      r.call_cls = hit ? "call-hit" : "call-miss";
      r.call_score = hit ? 1 : -1;
    }
  });
  return sorted;
}

function renderTable() {
  const rowsAll = buildRowsWithPunt();
  const isActCat = state.catView === "act";

  const cols = [
    { k: "dyn_rk", l: "Rk" },
    { k: "p", l: "Player", left: true },
    { k: "pos", l: "Pos", left: true },
    { k: "adp", l: "ADP" },
    { k: "dyn_drk", l: "&Delta;Rk" },
    { k: "dyn_pr", l: "Proj Z" },
    { k: "dyn_e", l: "Edge" },
    ...CATS.map((c, idx) => ({ k: `c_${idx}`, l: c, sep: idx === 0, catIdx: idx })),
    { k: "act_g", l: "G", sep: true },
    { k: "act_z", l: "Act Z" },
    { k: "vs_adp", l: "vs ADP" },
    { k: "call_score", l: "Call" },
  ];
  const thead = document.getElementById("thead-row");
  thead.innerHTML = cols.map(c => {
    const cls = [c.left ? "left" : "", c.sep ? "group-sep" : "", (c.catIdx !== undefined && state.punted.has(c.catIdx)) ? "punted-col" : ""].filter(Boolean).join(" ");
    const arr = state.sortKey === c.k ? (state.sortDir === 1 ? " &#9650;" : " &#9660;") : "";
    return `<th class="${cls}" data-key="${c.k}">${c.l}<span class="sort-arrow">${arr}</span></th>`;
  }).join("");
  thead.querySelectorAll("th").forEach(th => {
    th.onclick = () => {
      const k = th.dataset.key;
      if (state.sortKey === k) state.sortDir *= -1;
      else { state.sortKey = k; state.sortDir = (k === "dyn_rk" || k === "adp" || k === "p" || k === "pos") ? 1 : -1; }
      renderTable();
    };
  });

  let view = rowsAll;
  if (state.tab === "top150") view = rowsAll.filter(r => r.dyn_rk <= 150);
  else if (state.tab === "targets") view = rowsAll.filter(r => r.mrk <= 180 && r.dyn_e > 0).sort((a, b) => b.dyn_e - a.dyn_e).slice(0, 40);
  else if (state.tab === "fades") view = rowsAll.filter(r => r.mrk <= 150 && r.dyn_e < 0).sort((a, b) => a.dyn_e - b.dyn_e).slice(0, 40);
  else if (state.tab === "sleepers") view = rowsAll.filter(r => r.mrk > 150);

  if (state.filter === "hits") view = view.filter(r => r.call_score === 1);
  else if (state.filter === "misses") view = view.filter(r => r.call_score === -1);
  else if (state.filter === "rookies") view = view.filter(r => r.t === "ROOK");

  if (state.search.trim()) {
    const q = state.search.trim().toLowerCase();
    view = view.filter(r => r.p.toLowerCase().includes(q) || r.pos.toLowerCase().includes(q));
  }

  const sk = state.sortKey, sd = state.sortDir;
  view = [...view].sort((a, b) => {
    let va, vb;
    if (sk.startsWith("c_")) {
      const ci = parseInt(sk.slice(2), 10);
      va = (isActCat ? a.az : a.pz)[ci];
      vb = (isActCat ? b.az : b.pz)[ci];
    } else {
      va = a[sk]; vb = b[sk];
    }
    if (typeof va === "string") return sd * va.localeCompare(vb);
    return sd * ((va ?? 0) - (vb ?? 0));
  });

  const hits = view.filter(r => r.call_score === 1).length;
  const calls = view.filter(r => r.call_score !== 0).length;
  document.getElementById("row-count").innerHTML =
    `<strong>${view.length}</strong> players (${isActCat ? "Actual" : "Projected"} Z)` +
    (calls ? ` &middot; Calls: <strong>${hits}/${calls} (${(100 * hits / calls).toFixed(0)}%)</strong>` : "");

  const tbody = document.getElementById("tbody");
  tbody.innerHTML = view.map(r => {
    let tags = "";
    if (r.t === "ROOK") tags += `<span class="tag tag-rook">ROOK</span>`;
    else if (r.t === "RET") tags += `<span class="tag tag-ret">RET</span>`;
    if (r.al) tags += `<span class="tag tag-inj">INJ?</span>`;
    const drkCls = r.dyn_drk >= 5 ? "pos" : (r.dyn_drk <= -5 ? "neg" : "muted");
    const eCls = r.dyn_e >= 0.30 ? "edge-pill edge-pos" : (r.dyn_e <= -0.30 ? "edge-pill edge-neg" : "edge-pill edge-flat");
    const vsCls = r.vs_adp >= 0.5 ? "pos" : (r.vs_adp <= -0.5 ? "neg" : "muted");
    const cArr = isActCat ? r.az : r.pz;
    const catTds = cArr.map((v, ci) => {
      const cls = [ci === 0 ? "group-sep" : "", state.punted.has(ci) ? "punted-col" : ""].filter(Boolean).join(" ");
      return `<td class="${cls}"><span class="${catClass(v)}">${fmtSigned(v, 1)}</span></td>`;
    }).join("");
    return `<tr>
      <td><strong>#${r.dyn_rk}</strong></td>
      <td class="left player-cell">${r.p}${tags}</td>
      <td class="left muted">${r.pos}</td>
      <td>${r.adp.toFixed(1)}</td>
      <td class="${drkCls}">${r.dyn_drk >= 0 ? "+" : ""}${r.dyn_drk}</td>
      <td><strong>${fmtSigned(r.dyn_pr, 2)}</strong></td>
      <td><span class="${eCls}">${fmtSigned(r.dyn_e, 2)}</span></td>
      ${catTds}
      <td class="group-sep muted">${r.act_g}</td>
      <td>${fmtSigned(r.act_z, 2)}</td>
      <td class="${vsCls}">${fmtSigned(r.vs_adp, 2)}</td>
      <td><span class="call-badge ${r.call_cls}">${r.call_label}</span></td>
    </tr>`;
  }).join("");
}

function renderStep() {
  renderKPIs();
  drawChart();
  renderTable();
}

document.querySelectorAll(".info-pill").forEach(btn => {
  btn.onclick = () => {
    const key = btn.dataset.info;
    const drawer = document.getElementById("info-drawer");
    if (state.activeInfo === key) {
      state.activeInfo = null;
      btn.classList.remove("active");
      drawer.className = "info-drawer";
      return;
    }
    state.activeInfo = key;
    document.querySelectorAll(".info-pill").forEach(b => b.classList.toggle("active", b.dataset.info === key));
    const info = getInfoContent(key);
    drawer.innerHTML = info.html;
    drawer.className = `info-drawer open ${info.tone}`;
  };
});

document.getElementById("breakdown-toggle").onclick = () => {
  const body = document.getElementById("breakdown-body");
  const open = body.classList.toggle("open");
  document.getElementById("breakdown-toggle").classList.toggle("is-open", open);
  document.getElementById("breakdown-btn").innerHTML = open ? "Hide &#9652;" : "Season Breakdown &#9662;";
};
document.querySelectorAll(".stat-box[data-compare]").forEach(box => {
  box.onclick = () => {
    const k = box.dataset.compare;
    state.compareBox = (state.compareBox === k) ? null : k;
    renderKPIs();
  };
});
document.getElementById("compare-drawer").onclick = e => {
  if (e.target.closest(".compare-close")) {
    state.compareBox = null;
    renderKPIs();
  }
};
document.getElementById("punt-toggle").onclick = () => {
  const d = document.getElementById("punt-drawer");
  const open = d.classList.toggle("open");
  document.getElementById("punt-toggle").classList.toggle("active", open);
};
document.getElementById("guide-toggle").onclick = e => {
  const d = document.getElementById("guide-drawer");
  const open = d.classList.toggle("open");
  e.currentTarget.classList.toggle("active", open);
};
document.getElementById("play-btn").onclick = togglePlay;
document.getElementById("btn-start").onclick = () => { stopPlay(); state.step = 0; renderStep(); };
document.getElementById("btn-end").onclick = () => { stopPlay(); state.step = APP.seasons[state.season].max_step - 1; renderStep(); };
document.getElementById("game-slider").oninput = e => { stopPlay(); state.step = parseInt(e.target.value, 10); renderStep(); };
document.getElementById("punt-reset").onclick = () => { state.punted.clear(); buildPuntChips(); renderTable(); };
document.getElementById("cat-mode-btn").onclick = () => {
  state.catView = state.catView === "proj" ? "act" : "proj";
  document.getElementById("cat-mode-btn").textContent = state.catView === "proj" ? "Cats: Proj" : "Cats: Act";
  renderTable();
};
document.querySelectorAll("[data-speed]").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll("[data-speed]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.speedMs = parseInt(btn.dataset.speed, 10);
    if (state.playing) { stopPlay(); togglePlay(); }
    else document.getElementById("play-btn").innerHTML = state.speedMs === 125 ? "&#9654; Play 8x" : "&#9654; Play 4x";
  };
});
document.querySelectorAll("[data-chart]").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll("[data-chart]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.chartMode = btn.dataset.chart;
    drawChart();
  };
});
document.getElementById("edge-chart").onclick = e => {
  const S = APP.seasons[state.season];
  const rect = e.currentTarget.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left - rect.width * 0.052) / (rect.width * 0.932)));
  stopPlay();
  state.step = Math.round(frac * (S.max_step - 1));
  renderStep();
};
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.tab = btn.dataset.tab;
    state.sortKey = state.tab === "targets" ? "dyn_e" : (state.tab === "fades" ? "dyn_e" : "dyn_rk");
    state.sortDir = state.tab === "targets" ? -1 : 1;
    renderTable();
  };
});
document.querySelectorAll("[data-filter]").forEach(btn => {
  btn.onclick = () => {
    const f = btn.dataset.filter;
    if (state.filter === f) { state.filter = "all"; btn.classList.remove("active"); }
    else {
      document.querySelectorAll("[data-filter]").forEach(b => b.classList.remove("active"));
      state.filter = f; btn.classList.add("active");
    }
    renderTable();
  };
});
document.getElementById("search").oninput = e => { state.search = e.target.value; renderTable(); };

renderMultiYearHero();
buildPuntChips();
selectSeason(APP.default_season);
</script>
</body>
</html>
"""
)
