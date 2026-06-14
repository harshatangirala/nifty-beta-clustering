import json
import os
from datetime import date, timedelta
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from study_data import DAILY_PLAN, END_DATE, START_DATE, SUBJECTS

# Streamlit Cloud has an ephemeral filesystem — detect it so we can warn the user.
ON_CLOUD = bool(os.environ.get("STREAMLIT_SHARING") or os.environ.get("HOSTNAME", "").startswith("streamlit"))

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Harsha's MBA Study Tracker",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROGRESS_FILE = Path(__file__).parent / "progress.json"
TODAY = date.today()

# ── Colors ─────────────────────────────────────────────────────────────────────
BACKGROUND   = "#f8f7f4"
CARD_BG      = "#ffffff"
BORDER       = "#e5e4df"
TEXT_PRIMARY = "#1a1a18"
TEXT_MUTED   = "#888780"
TEXT_BODY    = "#5f5e5a"
ACCENT_BLUE  = "#185fa5"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{ background: {BACKGROUND}; }}
  [data-testid="stSidebar"] {{ background: {CARD_BG}; border-right: 1px solid {BORDER}; }}
  .main-header {{
    font-size: 26px; font-weight: 600; color: {TEXT_PRIMARY};
    margin-bottom: 2px; line-height: 1.2;
  }}
  .sub-header {{
    font-size: 14px; color: {TEXT_MUTED}; margin-bottom: 20px;
  }}
  .task-card {{
    background: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
  }}
  .task-title {{
    font-size: 15px; font-weight: 600; color: {TEXT_PRIMARY}; margin: 6px 0 4px 0;
  }}
  .task-detail {{
    font-size: 13px; color: {TEXT_BODY}; margin-bottom: 6px; line-height: 1.5;
  }}
  .task-resource {{
    font-size: 12px; color: {ACCENT_BLUE};
  }}
  .subject-pill {{
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600;
  }}
  .metric-card {{
    background: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 14px 16px; text-align: center;
  }}
  .metric-val {{
    font-size: 28px; font-weight: 700; color: {TEXT_PRIMARY}; line-height: 1.1;
  }}
  .metric-label {{
    font-size: 12px; color: {TEXT_MUTED}; margin-top: 4px;
  }}
  .week-badge {{
    display: inline-block; background: {TEXT_PRIMARY}; color: #fff;
    border-radius: 20px; padding: 2px 12px; font-size: 12px; font-weight: 600;
  }}
  .day-cell {{
    background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 10px; min-height: 90px;
  }}
  .day-cell-today {{
    background: {CARD_BG}; border: 2px solid {TEXT_PRIMARY}; border-radius: 8px;
    padding: 10px; min-height: 90px;
  }}
  .day-name {{ font-size: 11px; color: {TEXT_MUTED}; text-transform: uppercase; }}
  .day-num  {{ font-size: 20px; font-weight: 700; color: {TEXT_PRIMARY}; line-height: 1.2; }}
  .progress-bar-bg {{
    background: {BORDER}; border-radius: 4px; height: 8px; margin: 4px 0 10px 0;
  }}
  .progress-bar-fill {{
    background: currentColor; border-radius: 4px; height: 8px;
  }}
  div[data-testid="stCheckbox"] label {{ font-size: 14px !important; }}
  .stRadio > label {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ── Persistence ────────────────────────────────────────────────────────────────
def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_progress(data: dict) -> None:
    try:
        PROGRESS_FILE.write_text(json.dumps(data, indent=2))
    except OSError:
        pass  # Cloud: filesystem read-only; progress lives in session_state only


def task_key(dt: date, idx: int) -> str:
    return f"{dt.isoformat()}_{idx}"


def toggle_task(key: str) -> None:
    st.session_state.progress[key] = not st.session_state.progress.get(key, False)
    save_progress(st.session_state.progress)


# ── Session init ───────────────────────────────────────────────────────────────
if "progress" not in st.session_state:
    st.session_state.progress = load_progress()


# ── Stats ──────────────────────────────────────────────────────────────────────
def compute_stats() -> dict:
    prog = st.session_state.progress
    total_tasks = sum(len(v) for v in DAILY_PLAN.values())
    completed_tasks = sum(
        1 for dt, tasks in DAILY_PLAN.items()
        for idx in range(len(tasks))
        if prog.get(task_key(dt, idx), False)
    )

    # Hours studied
    hours_studied = sum(
        tasks[idx]["duration_hrs"]
        for dt, tasks in DAILY_PLAN.items()
        for idx in range(len(tasks))
        if prog.get(task_key(dt, idx), False)
    )

    # Streak — count consecutive days with all tasks done (skip days with 0 tasks)
    streak = 0
    check = TODAY
    while check >= START_DATE:
        tasks = DAILY_PLAN.get(check, [])
        if not tasks:
            check -= timedelta(days=1)
            continue
        done = all(prog.get(task_key(check, i), False) for i in range(len(tasks)))
        if done:
            streak += 1
            check -= timedelta(days=1)
        else:
            break

    # Days remaining
    if TODAY > END_DATE:
        days_remaining = 0
    elif TODAY < START_DATE:
        days_remaining = (END_DATE - START_DATE).days + 1
    else:
        days_remaining = (END_DATE - TODAY).days + 1

    # Subject stats
    subject_stats: dict[str, dict] = {}
    for subj in SUBJECTS:
        s_tasks = [
            (dt, idx)
            for dt, tasks in DAILY_PLAN.items()
            for idx, t in enumerate(tasks)
            if t["subject"] == subj
        ]
        s_done = sum(1 for dt, idx in s_tasks if prog.get(task_key(dt, idx), False))
        s_hours = sum(
            DAILY_PLAN[dt][idx]["duration_hrs"]
            for dt, idx in s_tasks
            if prog.get(task_key(dt, idx), False)
        )
        subject_stats[subj] = {
            "total": len(s_tasks),
            "done": s_done,
            "hours": round(s_hours, 1),
            "pct": round(s_done / len(s_tasks) * 100, 1) if s_tasks else 0,
        }

    return {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "hours_studied": round(hours_studied, 1),
        "streak": streak,
        "days_remaining": days_remaining,
        "overall_pct": round(completed_tasks / total_tasks * 100, 1) if total_tasks else 0,
        "subject_stats": subject_stats,
    }


# ── Render helpers ─────────────────────────────────────────────────────────────
def render_task_card(dt: date, idx: int, task: dict, show_checkbox: bool = True) -> None:
    subj = task["subject"]
    color = SUBJECTS[subj]["color"]
    light = SUBJECTS[subj]["light"]
    icon  = SUBJECTS[subj]["icon"]
    key   = task_key(dt, idx)
    done  = st.session_state.progress.get(key, False)
    opacity = "0.5" if done else "1.0"

    col_card, col_cb = st.columns([12, 1])
    with col_card:
        st.markdown(f"""
<div class="task-card" style="border-left:4px solid {color}; opacity:{opacity}">
  <span class="subject-pill" style="background:{light};color:{color}">{icon} {subj}</span>
  <span style="font-size:11px;color:{TEXT_MUTED};float:right">{task['duration_hrs']}h</span>
  <div class="task-title">{task['title']}</div>
  <div class="task-detail">{task['details']}</div>
  <div class="task-resource">📖 {task['resources']}</div>
</div>
""", unsafe_allow_html=True)
    if show_checkbox:
        with col_cb:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            checked = st.checkbox("", value=done, key=f"cb_{key}", label_visibility="collapsed")
            if checked != done:
                toggle_task(key)
                st.rerun()


def metric_html(value, label) -> str:
    return f"""
<div class="metric-card">
  <div class="metric-val">{value}</div>
  <div class="metric-label">{label}</div>
</div>"""


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
<div style='padding:12px 0 4px 0'>
  <div style='font-size:18px;font-weight:700;color:{TEXT_PRIMARY}'>📚 MBA Study</div>
  <div style='font-size:12px;color:{TEXT_MUTED}'>Harsha · 15 Jun – 7 Aug 2026</div>
</div>
""", unsafe_allow_html=True)
    st.divider()
    nav = st.radio(
        "Navigate",
        ["📅 Today", "📆 Full Calendar", "📊 Progress", "📖 By Subject", "🔗 Google Calendar"],
        label_visibility="collapsed",
    )
    st.divider()

    if ON_CLOUD:
        st.info("☁️ Running on Streamlit Cloud — progress resets on page refresh. Run locally for persistent tracking.", icon="ℹ️")

    # Mini stats
    stats = compute_stats()
    st.markdown(f"""
<div style='font-size:12px;color:{TEXT_MUTED};margin-bottom:6px'>Overall progress</div>
<div style='font-size:22px;font-weight:700;color:{TEXT_PRIMARY}'>{stats['overall_pct']}%</div>
<div class='progress-bar-bg'>
  <div class='progress-bar-fill' style='width:{stats['overall_pct']}%;background:{ACCENT_BLUE}'></div>
</div>
<div style='font-size:12px;color:{TEXT_MUTED}'>{stats['completed_tasks']} / {stats['total_tasks']} tasks · {stats['hours_studied']}h studied</div>
<div style='font-size:12px;color:{TEXT_MUTED};margin-top:4px'>🔥 {stats['streak']}-day streak · {stats['days_remaining']} days left</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW: TODAY
# ─────────────────────────────────────────────────────────────────────────────
if nav == "📅 Today":
    if TODAY < START_DATE:
        days_until = (START_DATE - TODAY).days
        st.markdown(f'<div class="main-header">📅 Today — {TODAY.strftime("%A, %d %B %Y")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">Study plan starts in {days_until} day{"s" if days_until != 1 else ""}! Get ready 🚀</div>', unsafe_allow_html=True)

        # Show a preview of Day 1
        st.markdown(f"### 👀 Day 1 Preview — {START_DATE.strftime('%A, %d %B')}")
        for idx, task in enumerate(DAILY_PLAN.get(START_DATE, [])):
            render_task_card(START_DATE, idx, task, show_checkbox=False)

    elif TODAY > END_DATE:
        st.markdown('<div class="main-header">🎓 Study Plan Complete!</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">54 days done. You\'re MBA-ready.</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        today_tasks = DAILY_PLAN.get(TODAY, [])
        day_num = (TODAY - START_DATE).days + 1
        st.markdown(f'<div class="main-header">📅 Today — {TODAY.strftime("%A, %d %B %Y")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">Day {day_num} of 54 · {stats["days_remaining"]} days remaining in plan</div>', unsafe_allow_html=True)

        if not today_tasks:
            st.info("Rest day — no tasks scheduled. Enjoy the break! 😌")
        else:
            done_today = sum(1 for i in range(len(today_tasks)) if st.session_state.progress.get(task_key(TODAY, i), False))
            hours_today = sum(t["duration_hrs"] for t in today_tasks)
            hours_done  = sum(t["duration_hrs"] for i, t in enumerate(today_tasks) if st.session_state.progress.get(task_key(TODAY, i), False))
            pct_today   = round(done_today / len(today_tasks) * 100) if today_tasks else 0

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(metric_html(f"{done_today}/{len(today_tasks)}", "Tasks today"), unsafe_allow_html=True)
            c2.markdown(metric_html(f"{hours_done:.1f}h", "Hours done"), unsafe_allow_html=True)
            c3.markdown(metric_html(f"{hours_today:.1f}h", "Total today"), unsafe_allow_html=True)
            c4.markdown(metric_html(f"{pct_today}%", "Today's progress"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if pct_today == 100:
                st.success("🎉 All tasks done for today! Great work.")
            elif pct_today > 0:
                st.progress(pct_today / 100)

            st.markdown("### Today's Tasks")
            for idx, task in enumerate(today_tasks):
                render_task_card(TODAY, idx, task)

        # Tomorrow preview
        tomorrow = TODAY + timedelta(days=1)
        if tomorrow in DAILY_PLAN:
            st.markdown("---")
            st.markdown(f"### 👀 Tomorrow — {tomorrow.strftime('%A, %d %B')}")
            for task in DAILY_PLAN[tomorrow]:
                subj  = task["subject"]
                color = SUBJECTS[subj]["color"]
                light = SUBJECTS[subj]["light"]
                icon  = SUBJECTS[subj]["icon"]
                st.markdown(f"""
<div class="task-card" style="border-left:4px solid {color}; opacity:0.7">
  <span class="subject-pill" style="background:{light};color:{color}">{icon} {subj}</span>
  <span style="font-size:11px;color:{TEXT_MUTED};float:right">{task['duration_hrs']}h</span>
  <div class="task-title">{task['title']}</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW: FULL CALENDAR
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "📆 Full Calendar":
    st.markdown('<div class="main-header">📆 Full Calendar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">54-day study plan overview</div>', unsafe_allow_html=True)

    # Build list of weeks
    total_days = (END_DATE - START_DATE).days + 1
    weeks = []
    cur = START_DATE
    while cur <= END_DATE:
        week_end = min(cur + timedelta(days=6), END_DATE)
        weeks.append((cur, week_end))
        cur += timedelta(days=7)

    week_labels = [f"Week {i+1} — {w[0].strftime('%b %d')} to {w[1].strftime('%b %d')}" for i, w in enumerate(weeks)]

    # Default to current week
    default_week = 0
    for i, (ws, we) in enumerate(weeks):
        if ws <= TODAY <= we:
            default_week = i
            break

    selected_week = st.selectbox("Select week", week_labels, index=default_week)
    week_idx = week_labels.index(selected_week)
    ws, we = weeks[week_idx]

    st.markdown(f'<span class="week-badge">Week {week_idx + 1}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # 7-column day grid
    cols = st.columns(7)
    check = ws
    for col_idx, col in enumerate(cols):
        day = ws + timedelta(days=col_idx)
        if day > END_DATE:
            continue
        tasks = DAILY_PLAN.get(day, [])
        done_count = sum(1 for i in range(len(tasks)) if st.session_state.progress.get(task_key(day, i), False))
        is_today = (day == TODAY)
        all_done = tasks and done_count == len(tasks)
        is_past  = day < TODAY

        cell_class = "day-cell-today" if is_today else "day-cell"

        # Subject dots
        subjects_today = list(dict.fromkeys(t["subject"] for t in tasks))
        dots = "".join(f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{SUBJECTS[s]["color"]};margin-right:2px"></span>' for s in subjects_today[:4])

        status = "✅" if all_done else ("" if not is_past else ("⚠️" if tasks and done_count < len(tasks) else ""))

        with col:
            st.markdown(f"""
<div class="{cell_class}">
  <div class="day-name">{day.strftime('%a')}</div>
  <div class="day-num">{day.day} {status}</div>
  <div style="margin-top:6px">{dots}</div>
  <div style="font-size:11px;color:{TEXT_MUTED};margin-top:4px">{len(tasks)} task{'s' if len(tasks)!=1 else ''} · {done_count} done</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### View a Day's Tasks")

    # Build valid dates in this week
    valid_dates = [ws + timedelta(days=i) for i in range(7) if (ws + timedelta(days=i)) <= END_DATE]
    default_date = TODAY if TODAY in valid_dates else valid_dates[0]

    chosen = st.date_input("Pick a date", value=default_date, min_value=START_DATE, max_value=END_DATE)
    tasks = DAILY_PLAN.get(chosen, [])
    if not tasks:
        st.info(f"No tasks on {chosen.strftime('%A, %d %B')} — rest day.")
    else:
        day_num = (chosen - START_DATE).days + 1
        st.markdown(f"**{chosen.strftime('%A, %d %B %Y')}** — Day {day_num}")
        for idx, task in enumerate(tasks):
            render_task_card(chosen, idx, task)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW: PROGRESS
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "📊 Progress":
    st.markdown('<div class="main-header">📊 Progress</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Track your 54-day journey</div>', unsafe_allow_html=True)

    stats = compute_stats()

    # Top 5 metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(metric_html(f"{stats['overall_pct']}%", "Overall"), unsafe_allow_html=True)
    c2.markdown(metric_html(f"{stats['completed_tasks']}", "Tasks done"), unsafe_allow_html=True)
    c3.markdown(metric_html(f"{stats['hours_studied']}h", "Hours studied"), unsafe_allow_html=True)
    c4.markdown(metric_html(f"🔥 {stats['streak']}", "Day streak"), unsafe_allow_html=True)
    c5.markdown(metric_html(f"{stats['days_remaining']}", "Days left"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Subject progress bars
    st.markdown("### By Subject")
    for subj, ss in stats["subject_stats"].items():
        color = SUBJECTS[subj]["color"]
        icon  = SUBJECTS[subj]["icon"]
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
<div style="margin-bottom:12px">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="font-size:13px;font-weight:600;color:{TEXT_PRIMARY}">{icon} {subj}</span>
    <span style="font-size:12px;color:{TEXT_MUTED}">{ss['done']}/{ss['total']} tasks · {ss['hours']}h</span>
  </div>
  <div class="progress-bar-bg">
    <div style="width:{ss['pct']}%;height:8px;border-radius:4px;background:{color}"></div>
  </div>
</div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='text-align:right;font-size:18px;font-weight:700;color:{color};margin-top:4px'>{ss['pct']}%</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Hours Studied by Subject")
        labels = [f"{SUBJECTS[s]['icon']} {s}" for s in stats["subject_stats"] if stats["subject_stats"][s]["hours"] > 0]
        values = [stats["subject_stats"][s]["hours"] for s in stats["subject_stats"] if stats["subject_stats"][s]["hours"] > 0]
        colors = [SUBJECTS[s]["color"] for s in stats["subject_stats"] if stats["subject_stats"][s]["hours"] > 0]

        if values:
            fig_donut = go.Figure(go.Pie(
                labels=labels, values=values,
                marker_colors=colors,
                hole=0.55,
                textinfo="percent",
                textfont_size=12,
            ))
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0), height=280,
                legend=dict(font_size=11, orientation="v"),
                showlegend=True,
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Complete some tasks to see the chart.")

    with col_right:
        st.markdown("#### Daily Completion %")
        plan_dates = sorted(DAILY_PLAN.keys())
        past_dates = [d for d in plan_dates if d <= TODAY]
        pcts = []
        for d in past_dates:
            tasks = DAILY_PLAN[d]
            if not tasks:
                pcts.append(None)
                continue
            done = sum(1 for i in range(len(tasks)) if st.session_state.progress.get(task_key(d, i), False))
            pcts.append(round(done / len(tasks) * 100, 1))

        if past_dates:
            fig_area = go.Figure()
            fig_area.add_trace(go.Scatter(
                x=past_dates, y=pcts, mode="lines",
                fill="tozeroy", fillcolor="rgba(55,138,221,0.12)",
                line=dict(color=SUBJECTS["Mathematics"]["color"], width=2),
                name="Completion %",
            ))
            if TODAY in plan_dates:
                fig_area.add_vline(x=str(TODAY), line_dash="dash", line_color=TEXT_MUTED, line_width=1)
            fig_area.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0), height=280,
                xaxis=dict(showgrid=False, tickfont_size=11),
                yaxis=dict(showgrid=True, gridcolor=BORDER, ticksuffix="%", tickfont_size=11, range=[0, 105]),
                showlegend=False,
            )
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("No data yet — start completing tasks!")

    # Cumulative hours chart
    st.markdown("#### Cumulative Study Hours")
    cum_hours = []
    running = 0.0
    for d in plan_dates:
        if d > TODAY:
            break
        tasks = DAILY_PLAN[d]
        for i, t in enumerate(tasks):
            if st.session_state.progress.get(task_key(d, i), False):
                running += t["duration_hrs"]
        cum_hours.append((d, round(running, 1)))

    if cum_hours:
        xs, ys = zip(*cum_hours)
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            fill="tozeroy", fillcolor="rgba(15,110,86,0.12)",
            line=dict(color=SUBJECTS["Quant Finance"]["color"], width=2),
        ))
        fig_cum.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0), height=220,
            xaxis=dict(showgrid=False, tickfont_size=11),
            yaxis=dict(showgrid=True, gridcolor=BORDER, ticksuffix="h", tickfont_size=11),
            showlegend=False,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    # Streak heatmap (GitHub-style)
    st.markdown("#### Completion Heatmap")
    all_dates = sorted(DAILY_PLAN.keys())
    heat_x, heat_y, heat_z, heat_text = [], [], [], []
    for d in all_dates:
        tasks = DAILY_PLAN[d]
        if not tasks:
            continue
        done = sum(1 for i in range(len(tasks)) if st.session_state.progress.get(task_key(d, i), False))
        pct = done / len(tasks)
        week_num = (d - START_DATE).days // 7
        dow = d.weekday()
        heat_x.append(week_num)
        heat_y.append(dow)
        heat_z.append(pct)
        heat_text.append(f"{d.strftime('%b %d')}: {round(pct*100)}%")

    if heat_x:
        fig_heat = go.Figure(go.Scatter(
            x=heat_x, y=heat_y,
            mode="markers",
            marker=dict(
                symbol="square",
                size=22,
                color=heat_z,
                colorscale=[[0, BORDER], [0.01, "#d4edda"], [0.5, "#5cb85c"], [1.0, "#1a6e1a"]],
                cmin=0, cmax=1,
                showscale=False,
            ),
            text=heat_text,
            hoverinfo="text",
        ))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=0, t=10, b=0), height=190,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(
                showgrid=False, zeroline=False, tickfont_size=10,
                tickvals=[0,1,2,3,4,5,6],
                ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                autorange="reversed",
            ),
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW: BY SUBJECT
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "📖 By Subject":
    st.markdown('<div class="main-header">📖 By Subject</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore and complete tasks by subject</div>', unsafe_allow_html=True)

    stats = compute_stats()

    subj_options = [f"{SUBJECTS[s]['icon']} {s}" for s in SUBJECTS]
    sel = st.selectbox("Choose subject", subj_options)
    subj = sel.split(" ", 1)[1]
    color = SUBJECTS[subj]["color"]
    ss = stats["subject_stats"][subj]

    # Subject metrics
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_html(f"{ss['done']}/{ss['total']}", "Tasks"), unsafe_allow_html=True)
    c2.markdown(metric_html(f"{ss['hours']}h", "Hours studied"), unsafe_allow_html=True)
    c3.markdown(metric_html(f"{ss['pct']}%", "Complete"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Progress bar
    st.markdown(f"""
<div class="progress-bar-bg">
  <div style="width:{ss['pct']}%;height:8px;border-radius:4px;background:{color}"></div>
</div>""", unsafe_allow_html=True)

    show_done = st.toggle("Show completed tasks", value=True)

    st.markdown("### All Tasks")

    all_tasks_for_subj = [
        (dt, idx, task)
        for dt, tasks in sorted(DAILY_PLAN.items())
        for idx, task in enumerate(tasks)
        if task["subject"] == subj
    ]

    for dt, idx, task in all_tasks_for_subj:
        done = st.session_state.progress.get(task_key(dt, idx), False)
        if not show_done and done:
            continue
        day_num = (dt - START_DATE).days + 1
        st.markdown(f"<div style='font-size:11px;color:{TEXT_MUTED};margin-top:8px'>Day {day_num} — {dt.strftime('%b %d, %Y')}</div>", unsafe_allow_html=True)
        render_task_card(dt, idx, task)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW: GOOGLE CALENDAR
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "🔗 Google Calendar":
    st.markdown('<div class="main-header">🔗 Google Calendar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sync your study plan to Google Calendar</div>', unsafe_allow_html=True)

    tab_ics, tab_oauth = st.tabs(["📥 ICS Export", "🔐 Google OAuth"])

    with tab_ics:
        st.markdown("### Export as .ics file")
        st.markdown("Download a `.ics` file and import it into Google Calendar, Apple Calendar, or Outlook.")
        st.markdown("""
**Steps:**
1. Click **Generate & Download** below
2. Open Google Calendar → Settings → Import & Export → Import
3. Select the downloaded `study_plan.ics`
4. All 54 days of tasks appear as calendar events
""")
        if st.button("📥 Generate & Download ICS", type="primary"):
            try:
                from calendar_sync import generate_ics
                ics_path = generate_ics()
                with open(ics_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download study_plan.ics",
                        data=f,
                        file_name="study_plan.ics",
                        mime="text/calendar",
                    )
                st.success(f"Generated {ics_path.name}!")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab_oauth:
        st.markdown("### Direct Google Calendar Sync")
        st.markdown("Requires `credentials.json` from Google Cloud Console (OAuth 2.0 Desktop App).")
        st.info("ℹ️ This only works when running locally on `localhost` — not on Streamlit Cloud.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Sync to Google Calendar", type="primary"):
                try:
                    from calendar_sync import sync_to_google
                    count = sync_to_google()
                    st.success(f"✅ Synced {count} events to Google Calendar!")
                except FileNotFoundError:
                    st.error("❌ `credentials.json` not found. Download from Google Cloud Console.")
                except Exception as e:
                    st.error(f"Error: {e}")
        with col2:
            if st.button("🗑️ Delete All Synced Events", type="secondary"):
                try:
                    from calendar_sync import delete_synced_events
                    count = delete_synced_events()
                    st.success(f"Deleted {count} events.")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")
        st.markdown("""
**Setup instructions:**
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project → Enable **Google Calendar API**
3. Create OAuth 2.0 credentials (Desktop App)
4. Download as `credentials.json` → place in `study_dashboard/`
5. Click **Sync** — browser will open for one-time auth
""")
