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
from datetime import date, timedelta

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="March Madness — Head-to-Head Predictor",
    page_icon="🏀",
    layout="wide",
)

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
  .tcol { flex: 1; padding: 20px 24px; text-align: center; }
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
  .otable th { padding: 10px 14px; color: #fff; font-size: 13px; font-weight: 600; text-align: center; }
  .otable td { padding: 11px 14px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }
  .otable tr:last-child td { border-bottom: none; }
  .pcell { font-size: 20px; font-weight: 800; text-align: center; width: 80px; }
  .pw { color: #e87722; } .pl { color: #bbb; }
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
  .shead .sb { text-align: left; padding-left: 10px; }
  .shead .sc { text-align: center; }
  .srow {
    display: grid; grid-template-columns: 1fr 180px 1fr;
    padding: 8px 16px; border-bottom: 1px solid #f2f2f2; align-items: center;
  }
  .srow:last-child { border-bottom: none; }
  .slbl { text-align: center; font-size: 12px; color: #555; font-weight: 500; }
  .sva  { text-align: right; font-size: 14px; font-weight: 700; padding-right: 10px; }
  .svb  { text-align: left;  font-size: 14px; font-weight: 700; padding-left:  10px; }
  .hi { color: #e87722; } .lo { color: #aaa; }
  .sec-title {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: #888;
    padding: 12px 16px 4px; border-top: 1px solid #eee; background: #fafafa;
  }
  .fdot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin: 1px; }
  .fw { background: #2d8a4e; } .fl { background: #c93534; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 1. ESPN API — fetch day by day for complete, accurate data
# ─────────────────────────────────────────────────────────────
def _parse_events(events: list) -> list:
    rows = []
    for event in events:
        for comp in event.get("competitions", []):
            date_str = comp.get("date", "")[:10]
            teams = comp.get("competitors", [])
            if len(teams) < 2:
                continue
            completed = comp.get("status", {}).get("type", {}).get("completed", False)
            if not completed:
                continue
            for i, team in enumerate(teams):
                opp = teams[1 - i]
                stats = {s["name"]: s.get("displayValue", "0") for s in team.get("statistics", [])}
                rows.append({
                    "game_date":                           date_str,
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
                })
    return rows


@st.cache_data(show_spinner=False)
def load_box_cached(season: int = 2026) -> pd.DataFrame:
    cache_file = f"mbb_box_{season}.pkl"
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)

    all_games = []
    start     = date(season - 1, 11, 1)   # Nov 1
    end       = date(season,      3, 31)   # Mar 31
    current   = start
    total_days = (end - start).days + 1
    day_num   = 0

    progress = st.progress(0, text="Fetching season data from ESPN…")

    while current <= end:
        ds  = current.strftime("%Y%m%d")
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/basketball"
            f"/mens-college-basketball/scoreboard?dates={ds}&limit=300"
        )
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                all_games.extend(_parse_events(r.json().get("events", [])))
        except Exception:
            pass

        day_num += 1
        progress.progress(
            min(day_num / total_days, 1.0),
            text=f"Fetching games… {current.strftime('%b %d, %Y')}",
        )
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

    for col in ["team_score","opponent_team_score","field_goals_made","field_goals_attempted",
                "three_point_field_goals_made","three_point_field_goals_attempted",
                "free_throws_attempted","free_throws_made","total_turnovers",
                "offensive_rebounds","defensive_rebounds","assists","blocks","steals",
                "points_in_paint","fast_break_points"]:
        if col in b.columns:
            b[col] = _to_num(b[col])

    b["team_winner"] = (
        b["team_winner"].astype(str).str.lower().isin(["true","1","yes","t"]).astype(int)
    )

    logo_col = "team_logo" if "team_logo" in b.columns else "team_display_name"

    grp = (
        b.groupby(["team_display_name", logo_col])
        .agg(
            games       =("team_score",                        "count"),
            wins        =("team_winner",                       "sum"),
            pts         =("team_score",                        "mean"),
            pts_allowed =("opponent_team_score",               "mean"),
            fgm         =("field_goals_made",                  "mean"),
            fga         =("field_goals_attempted",             "mean"),
            fg3m        =("three_point_field_goals_made",      "mean"),
            fg3a        =("three_point_field_goals_attempted", "mean"),
            fta         =("free_throws_attempted",             "mean"),
            ftm         =("free_throws_made",                  "mean"),
            tov         =("total_turnovers",                   "mean"),
            oreb        =("offensive_rebounds",                "mean"),
            dreb        =("defensive_rebounds",                "mean"),
            ast         =("assists",                           "mean"),
            blk         =("blocks",                            "mean"),
            stl         =("steals",                            "mean"),
            paint_pts   =("points_in_paint",                   "mean"),
            fast_break  =("fast_break_points",                 "mean"),
        )
        .reset_index()
        .rename(columns={"team_display_name": "team", logo_col: "logo"})
    )

    grp["losses"]         = grp["games"] - grp["wins"]
    grp["win_pct"]        = grp["wins"] / grp["games"].clip(lower=1)
    grp["scoring_margin"] = grp["pts"] - grp["pts_allowed"]
    grp["eff_fg_pct"]     = (grp["fgm"] + 0.5*grp["fg3m"]) / grp["fga"].clip(lower=1)
    grp["tov_rate"]       = grp["tov"] / (grp["fga"] + 0.44*grp["fta"] + grp["tov"]).clip(lower=1)
    grp["ft_rate"]        = grp["fta"] / grp["fga"].clip(lower=1)
    grp["fg3_pct"]        = grp["fg3m"] / grp["fg3a"].clip(lower=1)
    grp["fg3_rate"]       = grp["fg3a"] / grp["fga"].clip(lower=1)
    grp["ft_pct"]         = grp["ftm"]  / grp["fta"].clip(lower=1)

    return grp[grp["games"] >= 5].reset_index(drop=True)


@st.cache_data(show_spinner="Computing strength of schedule…")
def build_sos(box: pd.DataFrame) -> pd.DataFrame:
    b = box.copy()
    b["team_winner"] = (
        b["team_winner"].astype(str).str.lower().isin(["true","1","yes","t"]).astype(int)
    )
    team_wins = (
        b[b["team_display_name"].notna()]
        .groupby("team_display_name")["team_winner"].mean().reset_index()
        .rename(columns={"team_display_name":"opp","team_winner":"opp_win_pct"})
    )
    return (
        b[b["team_display_name"].notna() & b["opponent_team_display_name"].notna()]
        [["team_display_name","opponent_team_display_name"]]
        .rename(columns={"team_display_name":"team","opponent_team_display_name":"opp"})
        .merge(team_wins, on="opp", how="left")
        .groupby("team")["opp_win_pct"].mean().reset_index()
        .rename(columns={"opp_win_pct":"sos"})
    )


def recent_form(box: pd.DataFrame, team_name: str, n: int = 10) -> pd.DataFrame:
    b = box.copy()
    b["team_winner"] = (
        b["team_winner"].astype(str).str.lower().isin(["true","1","yes","t"]).astype(int)
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
    na, nb = (av - lo) / (hi - lo), (bv - lo) / (hi - lo)
    return (1-na, 1-nb) if inv else (na, nb)


def compute_adv_prob(sa, sb, fa, fb) -> dict:
    def g(df, col):
        return _safe(df[col].iloc[0]) if col in df.columns and not df.empty else 0.0

    a_form = fa["team_winner"].mean() if not fa.empty else 0.0
    b_form = fb["team_winner"].mean() if not fb.empty else 0.0
    if np.isnan(a_form): a_form = 0.0
    if np.isnan(b_form): b_form = 0.0

    nEFG = _norm(g(sa,"eff_fg_pct"),           g(sb,"eff_fg_pct"))
    nTOV = _norm(g(sa,"tov_rate"),              g(sb,"tov_rate"),     inv=True)
    nFT  = _norm(g(sa,"ft_rate"),               g(sb,"ft_rate"))
    nFG3 = _norm(g(sa,"fg3_pct"),               g(sb,"fg3_pct"))
    nMGN = _norm(g(sa,"scoring_margin"),        g(sb,"scoring_margin"))
    nSOS = _norm(g(sa,"sos"),                   g(sb,"sos"))
    nPNT = _norm(g(sa,"paint_pts"),             g(sb,"paint_pts"))
    nDEF = _norm(g(sa,"blk")+g(sa,"stl"),       g(sb,"blk")+g(sb,"stl"))
    nFRM = _norm(a_form,                         b_form)

    sca = 0.28*nEFG[0]+0.18*nTOV[0]+0.10*nFT[0]+0.08*nFG3[0]+0.10*nMGN[0]+0.08*nSOS[0]+0.06*nPNT[0]+0.04*nDEF[0]+0.04*nFRM[0]
    scb = 0.28*nEFG[1]+0.18*nTOV[1]+0.10*nFT[1]+0.08*nFG3[1]+0.10*nMGN[1]+0.08*nSOS[1]+0.06*nPNT[1]+0.04*nDEF[1]+0.04*nFRM[1]
    tot = sca + scb
    pa  = sca/tot if (tot and not np.isnan(tot)) else 0.5

    def pct(t): return (round(t[0]*100,1), round(t[1]*100,1))
    return dict(prob_a=pa, prob_b=1-pa,
                nEFG=pct(nEFG), nTOV=pct(nTOV), nFT=pct(nFT),
                nFG3=pct(nFG3), nMGN=pct(nMGN), nSOS=pct(nSOS),
                nPNT=pct(nPNT), nDEF=pct(nDEF), nFRM=pct(nFRM))


# ─────────────────────────────────────────────────────────────
# 4. HTML helpers
# ─────────────────────────────────────────────────────────────
def odds_row(na, nb, name, desc, wt):
    ca = "pw" if na >= nb else "pl"
    cb = "pw" if nb >  na else "pl"
    return f"""<tr>
      <td><div class="pcell {ca}">{na}%</div></td>
      <td><div class="mname">{name}</div><div class="mdesc">{desc}</div><div class="wt">Weight: {wt}</div></td>
      <td><div class="pcell {cb}">{nb}%</div></td>
    </tr>"""


def stat_row(va, vb, lbl, inv=False):
    try:
        an = float(str(va).replace("%","").replace("+",""))
        bn = float(str(vb).replace("%","").replace("+",""))
        if abs(an-bn) < 1e-9: ca = cb = ""
        elif not inv: ca="hi" if an>bn else "lo"; cb="hi" if bn>an else "lo"
        else:         ca="hi" if an<bn else "lo"; cb="hi" if bn<an else "lo"
    except Exception: ca = cb = ""
    return f'<div class="srow"><div class="sva {ca}">{va}</div><div class="slbl">{lbl}</div><div class="svb {cb}">{vb}</div></div>'


def form_dots(df):
    if df.empty: return "—"
    return "".join(
        f'<span class="fdot {"fw" if w==1 else "fl"}"></span>'
        for w in reversed(df["team_winner"].tolist())
    )


def sv(df, col, fmt=".1f"):
    if col not in df.columns or df.empty: return "N/A"
    try: return f"{float(df[col].iloc[0]):{fmt}}"
    except: return "N/A"


def pct_str(df, col):
    if col not in df.columns or df.empty: return "N/A"
    try: return f"{float(df[col].iloc[0])*100:.1f}%"
    except: return "N/A"


def safe_float(df, col, default=0.0):
    if col not in df.columns or df.empty: return default
    try: return float(df[col].iloc[0])
    except: return default


# ─────────────────────────────────────────────────────────────
# 5. App layout
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <h2>🏀 March Madness — Head-to-Head Predictor</h2>
  <p>2025–26 Men's College Basketball · Advanced Stats · ESPN API</p>
</div>
""", unsafe_allow_html=True)

box_raw    = load_box_cached(2026)
if box_raw.empty:
    st.stop()

metrics_df = build_advanced_metrics(box_raw)
sos_df     = build_sos(box_raw)
metrics_df = metrics_df.merge(sos_df, on="team", how="left")

team_list = sorted(metrics_df["team"].tolist())
default_a = "Duke Blue Devils"  if "Duke Blue Devils"  in team_list else team_list[0]
default_b = "Kentucky Wildcats" if "Kentucky Wildcats" in team_list else team_list[1]

c1, c2 = st.columns(2)
with c1: team_a = st.selectbox("Team A", team_list, index=team_list.index(default_a))
with c2: team_b = st.selectbox("Team B", team_list, index=team_list.index(default_b))

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

pred     = compute_adv_prob(sa, sb, fa, fb)
pa       = round(pred["prob_a"]*100, 1)
pb       = round(pred["prob_b"]*100, 1)
wins_a   = int(safe_float(sa,"wins"));   losses_a = int(safe_float(sa,"losses"))
wins_b   = int(safe_float(sb,"wins"));   losses_b = int(safe_float(sb,"losses"))
mgn_a    = round(safe_float(sa,"scoring_margin"),1)
mgn_b    = round(safe_float(sb,"scoring_margin"),1)
logo_a   = sa["logo"].iloc[0] if "logo" in sa.columns and not sa.empty else ""
logo_b   = sb["logo"].iloc[0] if "logo" in sb.columns and not sb.empty else ""

logo_a_html = f'<img src="{logo_a}" class="tlogo"><br>' if logo_a and str(logo_a) not in ("nan","") else ""
logo_b_html = f'<img src="{logo_b}" class="tlogo"><br>' if logo_b and str(logo_b) not in ("nan","") else ""
mgn_a_sign  = "+" if mgn_a > 0 else ""
mgn_b_sign  = "+" if mgn_b > 0 else ""
mgn_a_cls   = "tmarg-pos" if mgn_a >= 0 else "tmarg-neg"
mgn_b_cls   = "tmarg-pos" if mgn_b >= 0 else "tmarg-neg"

st.markdown(f"""
<div class="mcard">
  <div class="tcol">{logo_a_html}<div class="tname">{team_a}</div>
    <div class="trec">{wins_a}–{losses_a}</div>
    <div class="{mgn_a_cls}">{mgn_a_sign}{mgn_a} pt margin</div></div>
  <div class="midcol"><div class="midvs">VS</div><div class="midlbl">2025–26</div></div>
  <div class="tcol">{logo_b_html}<div class="tname">{team_b}</div>
    <div class="trec">{wins_b}–{losses_b}</div>
    <div class="{mgn_b_cls}">{mgn_b_sign}{mgn_b} pt margin</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<table class="otable">
  <thead><tr>
    <th style="width:85px">{team_a}</th>
    <th>Win Probability Model</th>
    <th style="width:85px">{team_b}</th>
  </tr></thead>
  <tbody>
    {odds_row(pa,              pb,              "Overall Prediction",    "Four Factors + advanced composite",  "—")}
    {odds_row(pred["nEFG"][0], pred["nEFG"][1], "Effective FG%",         "(FGM + 0.5×3PM) / FGA",             "28%")}
    {odds_row(pred["nTOV"][0], pred["nTOV"][1], "Turnover Rate",         "TOV / possessions (lower = better)", "18%")}
    {odds_row(pred["nFT"][0],  pred["nFT"][1],  "Free Throw Rate",       "FTA / FGA",                         "10%")}
    {odds_row(pred["nFG3"][0], pred["nFG3"][1], "3-Point Efficiency",    "3P%",                               "8%")}
    {odds_row(pred["nMGN"][0], pred["nMGN"][1], "Scoring Margin",        "Average point differential",        "10%")}
    {odds_row(pred["nSOS"][0], pred["nSOS"][1], "Strength of Schedule",  "Avg opponent win%",                 "8%")}
    {odds_row(pred["nPNT"][0], pred["nPNT"][1], "Points in the Paint",   "Interior scoring avg",              "6%")}
    {odds_row(pred["nDEF"][0], pred["nDEF"][1], "Defensive Activity",    "Blocks + steals per game",          "4%")}
    {odds_row(pred["nFRM"][0], pred["nFRM"][1], "Recent Form",           "Win rate last 10 games",            "4%")}
  </tbody>
</table>
""", unsafe_allow_html=True)

sos_a = round(safe_float(sa,"sos"),3)
sos_b = round(safe_float(sb,"sos"),3)

st.markdown(f"""
<div class="sgrid">
  <div class="shead">
    <span class="sa">{team_a}</span><span class="sc">STAT</span><span class="sb">{team_b}</span>
  </div>
  <div class="sec-title">SCORING</div>
  {stat_row(f"{wins_a}–{losses_a}", f"{wins_b}–{losses_b}", "Record")}
  {stat_row(sv(sa,"pts"),           sv(sb,"pts"),            "Points / Game")}
  {stat_row(sv(sa,"pts_allowed"),   sv(sb,"pts_allowed"),    "Points Allowed",    inv=True)}
  {stat_row(f"{mgn_a_sign}{mgn_a}", f"{mgn_b_sign}{mgn_b}", "Scoring Margin")}
  {stat_row(sv(sa,"paint_pts"),     sv(sb,"paint_pts"),      "Points in Paint")}
  {stat_row(sv(sa,"fast_break"),    sv(sb,"fast_break"),     "Fast Break Pts")}
  <div class="sec-title">FOUR FACTORS</div>
  {stat_row(pct_str(sa,"eff_fg_pct"), pct_str(sb,"eff_fg_pct"), "Effective FG%")}
  {stat_row(pct_str(sa,"tov_rate"),   pct_str(sb,"tov_rate"),   "Turnover Rate", inv=True)}
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
