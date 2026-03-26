"""
March Madness Head-to-Head Dashboard — Streamlit
Pulls data directly from ESPN API (no hoopR dependency)
Run: streamlit run march_madness_app.py
Install: pip install streamlit pandas numpy requests pyarrow
"""

import os
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="March Madness — Head-to-Head Predictor",
    page_icon="🏀",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .topbar {
    background: #1a2744; padding: 14px 32px;
    border-radius: 8px; margin-bottom: 24px;
  }
  .topbar h2 { color: #fff; font-size: 22px; font-weight: 700; margin: 0; }
  .topbar p  { color: #8899bb; font-size: 12px; margin: 4px 0 0; }

  .mcard {
    background: #fff; border: 1px solid #ccc;
    border-radius: 8px; display: flex;
    margin-bottom: 16px; overflow: hidden;
  }
  .tcol {
    flex: 1; padding: 20px 24px; text-align: center;
  }
  .midcol {
    width: 120px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: #f8f8f8;
    border-left: 1px solid #eee; border-right: 1px solid #eee;
  }
  .tlogo    { width: 64px; height: 64px; object-fit: contain; }
  .tname    { font-size: 18px; font-weight: 800; color: #1a2744; }
  .trec     { font-size: 13px; color: #777; margin-top: 4px; }
  .tmarg-pos{ font-size: 12px; font-weight: 600; color: #2d8a4e; margin-top: 4px; }
  .tmarg-neg{ font-size: 12px; font-weight: 600; color: #c93534; margin-top: 4px; }
  .midvs    { font-size: 22px; font-weight: 300; color: #bbb; }
  .midlbl   { font-size: 10px; color: #aaa; text-transform: uppercase; letter-spacing: .08em; }

  .otable {
    width: 100%; border-collapse: collapse; background: #fff;
    border: 1px solid #ccc; border-radius: 8px;
    overflow: hidden; margin-bottom: 16px;
  }
  .otable thead tr { background: #1a2744; }
  .otable th {
    padding: 10px 14px; color: #fff;
    font-size: 13px; font-weight: 600; text-align: center;
  }
  .otable td {
    padding: 11px 14px; border-bottom: 1px solid #f0f0f0; vertical-align: middle;
  }
  .otable tr:last-child td { border-bottom: none; }
  .pcell { font-size: 20px; font-weight: 800; text-align: center; width: 80px; }
  .pw    { color: #e87722; }
  .pl    { color: #bbb; }
  .mname { font-size: 13px; font-weight: 700; color: #1a2744; text-align: center; }
  .mdesc { font-size: 11px; color: #999; text-align: center; margin-top: 2px; }
  .wt    { font-size: 10px; color: #bbb; text-align: center; margin-top: 2px; }

  .sgrid {
    background: #fff; border: 1px solid #ccc;
    border-radius: 8px; overflow: hidden; margin-bottom: 16px;
  }
  .shead {
    background: #1a2744; color: #fff; padding: 10px 16px;
    font-size: 13px; font-weight: 700; display: grid;
    grid-template-columns: 1fr 180px 1fr;
  }
  .shead .sa { text-align: right; padding-right: 10px; }
  .shead .sb { text-align: left;  padding-left:  10px; }
  .shead .sc { text-align: center; }
  .srow {
    display: grid; grid-template-columns: 1fr 180px 1fr;
    padding: 8px 16px; border-bottom: 1px solid #f2f2f2; align-items: center;
  }
  .srow:last-child { border-bottom: none; }
  .slbl { text-align: center; font-size: 12px; color: #555; font-weight: 500; }
  .sva  { text-align: right; font-size: 14px; font-weight: 700; padding-right: 10px; }
  .svb  { text-align: left;  font-size: 14px; font-weight: 700; padding-left:  10px; }
  .hi   { color: #e87722; }
  .lo   { color: #aaa; }
  .sec-title {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: #888;
    padding: 12px 16px 4px; border-top: 1px solid #eee; background: #fafafa;
  }
  .fdot {
    display: inline-block; width: 11px; height: 11px;
    border-radius: 50%; margin: 1px;
  }
  .fw { background: #2d8a4e; }
  .fl { background: #c93534; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 1. Data loading — ESPN API directly (no hoopR)
# ─────────────────────────────────────────────────────────────
def _parse_events(events: list) -> list:
    rows = []
    for event in events:
        for comp in event.get("competitions", []):
            date = comp.get("date", "")[:10]
            teams = comp.get("competitors", [])
            if len(teams) < 2:
                continue
            # Only include completed games
            status = comp.get("status", {}).get("type", {}).get("completed", False)
            if not status:
                continue
            for i, team in enumerate(teams):
                opp = teams[1 - i]
                stats = {
                    s["name"]: s.get("displayValue", "0")
                    for s in team.get("statistics", [])
                }
                row = {
                    "game_date":                           date,
                    "team_display_name":                   team.get("team", {}).get("displayName", ""),
                    "team_logo":                           team.get("team", {}).get("logo", ""),
                    "team_score":                          team.get("score", 0),
                    "opponent_team_display_name":          opp.get("team", {}).get("displayName", ""),
                    "opponent_team_score":                 opp.get("score", 0),
                    "team_winner":                         team.get("winner", False),
                    "field_goals_made":                    stats.get("fieldGoalsMade", 0),
                    "field_goals_attempted":               stats.get("fieldGoalsAttempted", 0),
                    "three_point_field_goals_made":        stats.get("threePointFieldGoalsMade", 0),
                    "three_point_field_goals_attempted":   stats.get("threePointFieldGoalsAttempted", 0),
                    "free_throws_made":                    stats.get("freeThrowsMade", 0),
                    "free_throws_attempted":               stats.get("freeThrowsAttempted", 0),
                    "offensive_rebounds":                  stats.get("offensiveRebounds", 0),
                    "defensive_rebounds":                  stats.get("defensiveRebounds", 0),
                    "total_turnovers":                     stats.get("turnovers", 0),
                    "assists":                             stats.get("assists", 0),
                    "blocks":                              stats.get("blocks", 0),
                    "steals":                              stats.get("steals", 0),
                    "points_in_paint":                     stats.get("pointsInPaint", 0),
                    "fast_break_points":                   stats.get("fastBreakPoints", 0),
                }
                rows.append(row)
    return rows


@st.cache_data(show_spinner="Loading game data from ESPN…")
def load_box_cached(season: int = 2026) -> pd.DataFrame:
    cache_file = f"mbb_box_{season}.pkl"
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)

    from datetime import date, timedelta
    all_games = []
    start = date(season - 1, 11, 1)   # Nov 1 of prior year
    end   = date(season,     4, 8)    # Apr 8 covers full NCAA tournament
    current = start

    progress = st.progress(0, text="Fetching games from ESPN…")
    total_days = (end - start).days + 1
    day_num = 0

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/basketball"
            f"/mens-college-basketball/scoreboard"
            f"?dates={date_str}&limit=300&groups=50"
        )
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                all_games.extend(_parse_events(data.get("events", [])))
        except Exception:
            pass

        day_num += 1
        progress.progress(min(day_num / total_days, 1.0),
                          text=f"Fetching games… {date_str}")
        current += timedelta(days=1)

    progress.empty()

    if not all_games:
        st.error("Could not fetch game data from ESPN API.")
        return pd.DataFrame()

    df = pd.DataFrame(all_games).drop_duplicates()
    df.to_pickle(cache_file)
    return df


# ─────────────────────────────────────────────────────────────
# 2. Advanced metrics
# ─────────────────────────────────────────────────────────────
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


@st.cache_data(show_spinner="Computing advanced metrics…")
def build_advanced_metrics(box: pd.DataFrame) -> pd.DataFrame:
    b = box.copy()
    b = b[b["team_display_name"].notna() & b["team_score"].notna()]

    num_cols = [
        "team_score", "opponent_team_score", "field_goals_made",
        "field_goals_attempted", "three_point_field_goals_made",
        "three_point_field_goals_attempted", "free_throws_attempted",
        "free_throws_made", "total_turnovers", "offensive_rebounds",
        "defensive_rebounds", "assists", "blocks", "steals",
        "points_in_paint", "fast_break_points",
    ]
    for col in num_cols:
        if col in b.columns:
            b[col] = _to_num(b[col])

    b["team_winner"] = (
        b["team_winner"].astype(str).str.lower()
        .isin(["true", "1", "yes", "t"])
        .astype(int)
    )

    # Get the first non-empty logo per team separately so logo differences
    # across games don't split a team's records into multiple rows
    if "team_logo" in b.columns:
        logo_map = (
            b[b["team_logo"].notna() & (b["team_logo"] != "")]
            .groupby("team_display_name")["team_logo"]
            .first()
        )
    else:
        logo_map = pd.Series(dtype=str)

    grp = (
        b.groupby("team_display_name")
        .agg(
            games        = ("team_score",                        "count"),
            wins         = ("team_winner",                       "sum"),
            pts          = ("team_score",                        "mean"),
            pts_allowed  = ("opponent_team_score",               "mean"),
            fgm          = ("field_goals_made",                  "mean"),
            fga          = ("field_goals_attempted",             "mean"),
            fg3m         = ("three_point_field_goals_made",      "mean"),
            fg3a         = ("three_point_field_goals_attempted", "mean"),
            fta          = ("free_throws_attempted",             "mean"),
            ftm          = ("free_throws_made",                  "mean"),
            tov          = ("total_turnovers",                   "mean"),
            oreb         = ("offensive_rebounds",                "mean"),
            dreb         = ("defensive_rebounds",                "mean"),
            ast          = ("assists",                           "mean"),
            blk          = ("blocks",                            "mean"),
            stl          = ("steals",                            "mean"),
            paint_pts    = ("points_in_paint",                   "mean"),
            fast_break   = ("fast_break_points",                 "mean"),
        )
        .reset_index()
        .rename(columns={"team_display_name": "team"})
    )
    grp["logo"] = grp["team"].map(logo_map).fillna("")

    grp["losses"]         = grp["games"] - grp["wins"]
    grp["win_pct"]        = grp["wins"] / grp["games"].clip(lower=1)
    grp["scoring_margin"] = grp["pts"] - grp["pts_allowed"]
    grp["eff_fg_pct"]     = (grp["fgm"] + 0.5 * grp["fg3m"]) / grp["fga"].clip(lower=1)
    grp["tov_rate"]       = grp["tov"] / (grp["fga"] + 0.44 * grp["fta"] + grp["tov"]).clip(lower=1)
    grp["ft_rate"]        = grp["fta"] / grp["fga"].clip(lower=1)
    grp["fg3_pct"]        = grp["fg3m"] / grp["fg3a"].clip(lower=1)
    grp["fg3_rate"]       = grp["fg3a"] / grp["fga"].clip(lower=1)
    grp["ft_pct"]         = grp["ftm"]  / grp["fta"].clip(lower=1)

    return grp[grp["games"] >= 1].reset_index(drop=True)


@st.cache_data(show_spinner="Computing strength of schedule…")
def build_sos(box: pd.DataFrame) -> pd.DataFrame:
    b = box.copy()
    b["team_winner"] = (
        b["team_winner"].astype(str).str.lower()
        .isin(["true", "1", "yes", "t"])
        .astype(int)
    )
    team_wins = (
        b[b["team_display_name"].notna()]
        .groupby("team_display_name")["team_winner"]
        .mean()
        .reset_index()
        .rename(columns={"team_display_name": "opp", "team_winner": "opp_win_pct"})
    )
    return (
        b[b["team_display_name"].notna() & b["opponent_team_display_name"].notna()]
        [["team_display_name", "opponent_team_display_name"]]
        .rename(columns={
            "team_display_name": "team",
            "opponent_team_display_name": "opp",
        })
        .merge(team_wins, on="opp", how="left")
        .groupby("team")["opp_win_pct"]
        .mean()
        .reset_index()
        .rename(columns={"opp_win_pct": "sos"})
    )


def recent_form(box: pd.DataFrame, team_name: str, n: int = 10) -> pd.DataFrame:
    b = box.copy()
    b["team_winner"] = (
        b["team_winner"].astype(str).str.lower()
        .isin(["true", "1", "yes", "t"])
        .astype(int)
    )
    b["game_date"] = pd.to_datetime(b["game_date"], errors="coerce")
    return (
        b[b["team_display_name"] == team_name]
        .sort_values("game_date", ascending=False)
        .head(n)
    )


# ─────────────────────────────────────────────────────────────
# 3. Win probability model
# ─────────────────────────────────────────────────────────────
def _safe(val) -> float:
    try:
        v = float(val)
        return 0.0 if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return 0.0


def _norm(av: float, bv: float, inv: bool = False):
    if abs(av - bv) < 1e-12:
        return 0.5, 0.5
    lo, hi = min(av, bv), max(av, bv)
    if hi == lo:
        return 0.5, 0.5
    na, nb = (av - lo) / (hi - lo), (bv - lo) / (hi - lo)
    return (1 - na, 1 - nb) if inv else (na, nb)


def compute_adv_prob(sa: pd.DataFrame, sb: pd.DataFrame,
                     fa: pd.DataFrame, fb: pd.DataFrame) -> dict:
    def g(df, col):
        return _safe(df[col].iloc[0]) if col in df.columns and not df.empty else 0.0

    a_form = fa["team_winner"].mean() if not fa.empty else 0.0
    b_form = fb["team_winner"].mean() if not fb.empty else 0.0
    if np.isnan(a_form): a_form = 0.0
    if np.isnan(b_form): b_form = 0.0

    # ── Evidence-based weights ────────────────────────────────────────────────
    # Sources: KenPom, BartTorvik, empirical Four Factors re-analyses,
    # Harvard Sports Analysis, ThePowerRank research
    #
    # Scoring margin (net efficiency) is the #1 proven predictor of future wins.
    # Empirical Four Factors weights: eFG% ~44%, TOV% ~37%, OREB% ~12%, FTR ~7%
    # 3P% has only 14.5% year-to-year repeatability — dropped as noisy.
    # Defensive efficiency tracked separately to reward balanced teams.
    # Recent form (hot streaks) has near-zero predictive value per research.

    nMGN     = _norm(g(sa, "scoring_margin"),         g(sb, "scoring_margin"))         # 30%
    nEFG     = _norm(g(sa, "eff_fg_pct"),             g(sb, "eff_fg_pct"))             # 20%
    nTOV     = _norm(g(sa, "tov_rate"),               g(sb, "tov_rate"),  inv=True)    # 15%
    nDEF_EFF = _norm(g(sa, "pts_allowed"),            g(sb, "pts_allowed"), inv=True)  # 10%
    nSOS     = _norm(g(sa, "sos"),                    g(sb, "sos"))                    # 8%
    nOREB    = _norm(g(sa, "oreb"),                   g(sb, "oreb"))                   # 6%
    nFTPCT   = _norm(g(sa, "ft_pct"),                 g(sb, "ft_pct"))                 # 5%
    nDEF_ACT = _norm(g(sa, "blk") + g(sa, "stl"),    g(sb, "blk") + g(sb, "stl"))    # 4%
    nFRM     = _norm(a_form,                           b_form)                          # 2%

    sca = (0.30*nMGN[0]  + 0.20*nEFG[0]  + 0.15*nTOV[0] +
           0.10*nDEF_EFF[0] + 0.08*nSOS[0] + 0.06*nOREB[0] +
           0.05*nFTPCT[0] + 0.04*nDEF_ACT[0] + 0.02*nFRM[0])
    scb = (0.30*nMGN[1]  + 0.20*nEFG[1]  + 0.15*nTOV[1] +
           0.10*nDEF_EFF[1] + 0.08*nSOS[1] + 0.06*nOREB[1] +
           0.05*nFTPCT[1] + 0.04*nDEF_ACT[1] + 0.02*nFRM[1])

    tot = sca + scb
    pa  = sca / tot if (tot and not np.isnan(tot)) else 0.5

    def pct(t): return (round(t[0]*100, 1), round(t[1]*100, 1))

    return dict(
        prob_a=pa, prob_b=1-pa,
        nMGN=pct(nMGN), nEFG=pct(nEFG), nTOV=pct(nTOV),
        nDEF_EFF=pct(nDEF_EFF), nSOS=pct(nSOS), nOREB=pct(nOREB),
        nFTPCT=pct(nFTPCT), nDEF_ACT=pct(nDEF_ACT), nFRM=pct(nFRM),
    )


# ─────────────────────────────────────────────────────────────
# 4. HTML helpers
# ─────────────────────────────────────────────────────────────
def odds_row(na: float, nb: float, name: str, desc: str, wt: str) -> str:
    ca = "pw" if na >= nb else "pl"
    cb = "pw" if nb >  na else "pl"
    return f"""<tr>
      <td><div class="pcell {ca}">{na}%</div></td>
      <td>
        <div class="mname">{name}</div>
        <div class="mdesc">{desc}</div>
        <div class="wt">Weight: {wt}</div>
      </td>
      <td><div class="pcell {cb}">{nb}%</div></td>
    </tr>"""


def stat_row(va, vb, lbl: str, inv: bool = False) -> str:
    try:
        an = float(str(va).replace("%", "").replace("+", ""))
        bn = float(str(vb).replace("%", "").replace("+", ""))
        if abs(an - bn) < 1e-9:
            ca = cb = ""
        elif not inv:
            ca = "hi" if an > bn else "lo"
            cb = "hi" if bn > an else "lo"
        else:
            ca = "hi" if an < bn else "lo"
            cb = "hi" if bn < an else "lo"
    except Exception:
        ca = cb = ""
    return f"""<div class="srow">
      <div class="sva {ca}">{va}</div>
      <div class="slbl">{lbl}</div>
      <div class="svb {cb}">{vb}</div>
    </div>"""


def form_dots(df: pd.DataFrame) -> str:
    if df.empty:
        return "—"
    return "".join(
        f'<span class="fdot {"fw" if w == 1 else "fl"}"></span>'
        for w in reversed(df["team_winner"].tolist())
    )


def sv(df: pd.DataFrame, col: str, fmt: str = ".1f") -> str:
    if col not in df.columns or df.empty:
        return "N/A"
    try:
        return f"{float(df[col].iloc[0]):{fmt}}"
    except Exception:
        return "N/A"


def pct_str(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns or df.empty:
        return "N/A"
    try:
        return f"{float(df[col].iloc[0])*100:.1f}%"
    except Exception:
        return "N/A"


def safe_float(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns or df.empty:
        return default
    try:
        return float(df[col].iloc[0])
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────
# 5. App layout
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <h2>🏀 March Madness — Head-to-Head Predictor</h2>
  <p>2025–26 Men's College Basketball · Advanced Stats · ESPN API</p>
</div>
""", unsafe_allow_html=True)

# Load & prepare data
box_raw = load_box_cached(2026)
if box_raw.empty:
    st.stop()

metrics_df = build_advanced_metrics(box_raw)
sos_df     = build_sos(box_raw)
metrics_df = metrics_df.merge(sos_df, on="team", how="left")

team_list = sorted(metrics_df["team"].tolist())
default_a = "Duke Blue Devils"  if "Duke Blue Devils"  in team_list else team_list[0]
default_b = "Kentucky Wildcats" if "Kentucky Wildcats" in team_list else team_list[1]

# Team selectors
c1, c2 = st.columns(2)
with c1:
    team_a = st.selectbox("Team A", team_list, index=team_list.index(default_a))
with c2:
    team_b = st.selectbox("Team B", team_list, index=team_list.index(default_b))

if team_a == team_b:
    st.warning("Please select two different teams.")
    st.stop()

sa = metrics_df[metrics_df["team"] == team_a]
sb = metrics_df[metrics_df["team"] == team_b]
fa = recent_form(box_raw, team_a)
fb = recent_form(box_raw, team_b)

if sa.empty or sb.empty:
    st.error("Could not find stats for one of the selected teams.")
    st.stop()

pred = compute_adv_prob(sa, sb, fa, fb)
pa   = round(pred["prob_a"] * 100, 1)
pb   = round(pred["prob_b"] * 100, 1)

wins_a   = int(safe_float(sa, "wins"))
losses_a = int(safe_float(sa, "losses"))
wins_b   = int(safe_float(sb, "wins"))
losses_b = int(safe_float(sb, "losses"))
mgn_a    = round(safe_float(sa, "scoring_margin"), 1)
mgn_b    = round(safe_float(sb, "scoring_margin"), 1)
logo_a   = sa["logo"].iloc[0] if "logo" in sa.columns and not sa.empty else ""
logo_b   = sb["logo"].iloc[0] if "logo" in sb.columns and not sb.empty else ""

logo_a_html = f'<img src="{logo_a}" class="tlogo"><br>' if logo_a and str(logo_a) not in ("nan", "") else ""
logo_b_html = f'<img src="{logo_b}" class="tlogo"><br>' if logo_b and str(logo_b) not in ("nan", "") else ""
mgn_a_sign  = "+" if mgn_a > 0 else ""
mgn_b_sign  = "+" if mgn_b > 0 else ""
mgn_a_cls   = "tmarg-pos" if mgn_a >= 0 else "tmarg-neg"
mgn_b_cls   = "tmarg-pos" if mgn_b >= 0 else "tmarg-neg"

# ── Matchup header ──────────────────────────────────────────
st.markdown(f"""
<div class="mcard">
  <div class="tcol">
    {logo_a_html}
    <div class="tname">{team_a}</div>
    <div class="trec">{wins_a}–{losses_a}</div>
    <div class="{mgn_a_cls}">{mgn_a_sign}{mgn_a} pt margin</div>
  </div>
  <div class="midcol">
    <div class="midvs">VS</div>
    <div class="midlbl">2025–26</div>
  </div>
  <div class="tcol">
    {logo_b_html}
    <div class="tname">{team_b}</div>
    <div class="trec">{wins_b}–{losses_b}</div>
    <div class="{mgn_b_cls}">{mgn_b_sign}{mgn_b} pt margin</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Win probability table ───────────────────────────────────
st.markdown(f"""
<table class="otable">
  <thead>
    <tr>
      <th style="width:85px">{team_a}</th>
      <th>Win Probability Model</th>
      <th style="width:85px">{team_b}</th>
    </tr>
  </thead>
  <tbody>
    {odds_row(pa,                    pb,                    "Overall Prediction",      "Evidence-based composite model",             "—")}
    {odds_row(pred["nMGN"][0],      pred["nMGN"][1],      "Scoring Margin",          "#1 predictor of future wins (KenPom/BPI)",   "30%")}
    {odds_row(pred["nEFG"][0],      pred["nEFG"][1],      "Effective FG%",           "(FGM + 0.5×3PM) / FGA — top Four Factor",    "20%")}
    {odds_row(pred["nTOV"][0],      pred["nTOV"][1],      "Turnover Rate",           "TOV / possessions (lower = better)",         "15%")}
    {odds_row(pred["nDEF_EFF"][0],  pred["nDEF_EFF"][1],  "Defensive Efficiency",    "Points allowed/game — rewards balanced teams","10%")}
    {odds_row(pred["nSOS"][0],      pred["nSOS"][1],      "Strength of Schedule",    "Validates whether efficiency nums are real",  "8%")}
    {odds_row(pred["nOREB"][0],     pred["nOREB"][1],      "Off. Rebounding",         "Extra possessions via offensive boards",      "6%")}
    {odds_row(pred["nFTPCT"][0],    pred["nFTPCT"][1],    "Free Throw %",            "Most stable shooting metric (70% repeatable)","5%")}
    {odds_row(pred["nDEF_ACT"][0],  pred["nDEF_ACT"][1],  "Defensive Activity",      "Blocks + steals — shot quality defense",      "4%")}
    {odds_row(pred["nFRM"][0],      pred["nFRM"][1],      "Recent Form",             "Win rate last 10 games (weak signal)",        "2%")}
  </tbody>
</table>
""", unsafe_allow_html=True)

# ── Stats comparison grid ───────────────────────────────────
sos_a = round(safe_float(sa, "sos"), 3)
sos_b = round(safe_float(sb, "sos"), 3)

st.markdown(f"""
<div class="sgrid">
  <div class="shead">
    <span class="sa">{team_a}</span>
    <span class="sc">STAT</span>
    <span class="sb">{team_b}</span>
  </div>

  <div class="sec-title">SCORING</div>
  {stat_row(f"{wins_a}–{losses_a}",  f"{wins_b}–{losses_b}",  "Record")}
  {stat_row(sv(sa,"pts"),            sv(sb,"pts"),             "Points / Game")}
  {stat_row(sv(sa,"pts_allowed"),    sv(sb,"pts_allowed"),     "Points Allowed",        inv=True)}
  {stat_row(f"{mgn_a_sign}{mgn_a}", f"{mgn_b_sign}{mgn_b}",   "Scoring Margin")}
  {stat_row(sv(sa,"paint_pts"),      sv(sb,"paint_pts"),       "Points in Paint")}
  {stat_row(sv(sa,"fast_break"),     sv(sb,"fast_break"),      "Fast Break Pts")}

  <div class="sec-title">FOUR FACTORS</div>
  {stat_row(pct_str(sa,"eff_fg_pct"), pct_str(sb,"eff_fg_pct"), "Effective FG%")}
  {stat_row(pct_str(sa,"tov_rate"),   pct_str(sb,"tov_rate"),   "Turnover Rate",  inv=True)}
  {stat_row(pct_str(sa,"ft_rate"),    pct_str(sb,"ft_rate"),    "Free Throw Rate")}
  {stat_row(sv(sa,"oreb"),            sv(sb,"oreb"),             "Off Rebounds / Game")}

  <div class="sec-title">SHOOTING</div>
  {stat_row(pct_str(sa,"fg3_pct"),  pct_str(sb,"fg3_pct"),  "3-Point %")}
  {stat_row(pct_str(sa,"fg3_rate"), pct_str(sb,"fg3_rate"), "3-Point Rate (3PA/FGA)")}
  {stat_row(pct_str(sa,"ft_pct"),   pct_str(sb,"ft_pct"),   "Free Throw %")}

  <div class="sec-title">DEFENSE &amp; PLAYMAKING</div>
  {stat_row(sv(sa,"ast"),  sv(sb,"ast"),  "Assists / Game")}
  {stat_row(sv(sa,"blk"),  sv(sb,"blk"),  "Blocks / Game")}
  {stat_row(sv(sa,"stl"),  sv(sb,"stl"),  "Steals / Game")}
  {stat_row(sv(sa,"dreb"), sv(sb,"dreb"), "Def Rebounds / Game")}
  {stat_row(str(sos_a),    str(sos_b),    "Strength of Schedule")}

  <div class="sec-title">FORM</div>
  <div class="srow">
    <div class="sva" style="text-align:right;padding-right:10px;">{form_dots(fa)}</div>
    <div class="slbl">Last 10 Games</div>
    <div class="svb" style="padding-left:10px;">{form_dots(fb)}</div>
  </div>
</div>
""", unsafe_allow_html=True)
