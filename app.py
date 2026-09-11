from __future__ import annotations

import json
import os
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Instagram Growth Copilot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def make_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    today = pd.Timestamp.today().normalize()
    days = pd.date_range(end=today, periods=28, freq="D")
    daily = pd.DataFrame(
        {
            "date": days,
            "Reach": [
                1120, 1180, 980, 1430, 1560, 1890, 1720, 1210, 1320, 1460, 1590, 1780,
                2210, 2080, 1760, 1880, 2010, 2330, 2490, 2180, 1960, 2110, 2450, 2710,
                2590, 2840, 3020, 3180,
            ],
            "Engagements": [
                96, 110, 83, 128, 151, 188, 170, 104, 117, 142, 155, 169, 228, 205,
                164, 180, 194, 236, 251, 225, 189, 203, 246, 275, 259, 298, 316, 342,
            ],
            "Followers": [
                8420, 8432, 8441, 8458, 8476, 8505, 8521, 8530, 8547, 8562, 8581, 8604,
                8632, 8659, 8671, 8690, 8711, 8742, 8765, 8782, 8801, 8828, 8857, 8891,
                8910, 8942, 8974, 9018,
            ],
        }
    )
    posts = pd.DataFrame(
        [
            {
                "Published": today - pd.Timedelta(days=1),
                "Type": "Reel",
                "Caption": "3 ways to make your morning routine stick",
                "Reach": 8640,
                "Likes": 612,
                "Comments": 48,
                "Saves": 231,
                "Shares": 87,
            },
            {
                "Published": today - pd.Timedelta(days=4),
                "Type": "Carousel",
                "Caption": "The creator's weekly reset checklist",
                "Reach": 6210,
                "Likes": 497,
                "Comments": 31,
                "Saves": 318,
                "Shares": 54,
            },
            {
                "Published": today - pd.Timedelta(days=7),
                "Type": "Reel",
                "Caption": "POV: you finally batch your content",
                "Reach": 5840,
                "Likes": 421,
                "Comments": 36,
                "Saves": 187,
                "Shares": 72,
            },
            {
                "Published": today - pd.Timedelta(days=10),
                "Type": "Photo",
                "Caption": "Desk setup for a focused work day",
                "Reach": 2840,
                "Likes": 226,
                "Comments": 18,
                "Saves": 91,
                "Shares": 19,
            },
            {
                "Published": today - pd.Timedelta(days=13),
                "Type": "Carousel",
                "Caption": "Five hooks for your next educational post",
                "Reach": 4180,
                "Likes": 352,
                "Comments": 24,
                "Saves": 164,
                "Shares": 41,
            },
            {
                "Published": today - pd.Timedelta(days=16),
                "Type": "Reel",
                "Caption": "A realistic day behind the scenes",
                "Reach": 3970,
                "Likes": 287,
                "Comments": 22,
                "Saves": 103,
                "Shares": 38,
            },
        ]
    )
    return daily, posts


def format_number(value: int | float) -> str:
    return f"{value:,.0f}"


def format_delta(current: float, previous: float) -> str:
    if previous == 0:
        return "—"
    return f"{((current - previous) / previous) * 100:+.1f}%"


def ask_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 8192},
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini returned an error ({error.code}). {message}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach Gemini: {error.reason}") from error

    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini did not return an answer.")
    parts = candidates[0].get("content", {}).get("parts", [])
    answer = "".join(part.get("text", "") for part in parts).strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty answer.")
    return answer


def build_context(daily: pd.DataFrame, posts: pd.DataFrame, start: date, end: date) -> str:
    selected_daily = daily[
        (daily["date"].dt.date >= start) & (daily["date"].dt.date <= end)
    ]
    selected_posts = posts[
        (posts["Published"].dt.date >= start) & (posts["Published"].dt.date <= end)
    ]
    latest_followers = int(selected_daily.iloc[-1]["Followers"]) if not selected_daily.empty else 0
    first_followers = int(selected_daily.iloc[0]["Followers"]) if not selected_daily.empty else 0
    return f"""
You are the growth strategist inside an Instagram analytics dashboard.
Use only the following account data and be clear when you are making a recommendation.
Answer in a concise, practical way with bullets when helpful. Do not invent metrics.

Period: {start.isoformat()} to {end.isoformat()}
Reach: {int(selected_daily["Reach"].sum()):,}
Engagements: {int(selected_daily["Engagements"].sum()):,}
Follower change: {latest_followers - first_followers:+,}
Posts in period: {len(selected_posts)}
Top posts:
{selected_posts.nlargest(3, "Reach")[["Type", "Caption", "Reach", "Saves", "Shares"]].to_string(index=False)}

Question from the creator:
""".strip()


daily_data, post_data = make_demo_data()
min_date = daily_data["date"].min().date()
max_date = daily_data["date"].max().date()

with st.sidebar:
    st.title("Growth Copilot")
    st.caption("Instagram analytics and next-step guidance")
    st.divider()
    st.subheader("Workspace")
    st.selectbox("Account", ["@northstar.creates"], label_visibility="collapsed")
    st.info("Demo data is active. Add your Instagram connection when you are ready to analyze a live account.")
    st.divider()
    st.subheader("Date range")
    date_range = st.date_input(
        "Choose a reporting window",
        value=(max_date - timedelta(days=13), max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed",
    )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    st.divider()
    st.caption("Insights are powered by Google Gemini.")

selected_daily = daily_data[
    (daily_data["date"].dt.date >= start_date) & (daily_data["date"].dt.date <= end_date)
]
selected_posts = post_data[
    (post_data["Published"].dt.date >= start_date) & (post_data["Published"].dt.date <= end_date)
].copy()

st.title("Instagram Growth Copilot")
st.caption(
    f"Your content performance at a glance · {start_date.strftime('%b %-d')} – {end_date.strftime('%b %-d, %Y')}"
)

if not os.getenv("GEMINI_API_KEY"):
    st.warning("Gemini chat is not configured yet. Add the GEMINI_API_KEY secret to enable it.")

reach = int(selected_daily["Reach"].sum())
engagements = int(selected_daily["Engagements"].sum())
followers_now = int(selected_daily.iloc[-1]["Followers"])
followers_then = int(selected_daily.iloc[0]["Followers"])
follower_change = followers_now - followers_then
engagement_rate = (engagements / reach * 100) if reach else 0
previous_days = max(len(selected_daily), 1)
previous_window = daily_data.iloc[max(0, len(daily_data) - previous_days * 2): max(0, len(daily_data) - previous_days)]
reach_delta = format_delta(reach, float(previous_window["Reach"].sum())) if not previous_window.empty else "—"
engagement_delta = format_delta(engagements, float(previous_window["Engagements"].sum())) if not previous_window.empty else "—"

metric_cols = st.columns(4)
metric_cols[0].metric("Accounts reached", format_number(reach), reach_delta)
metric_cols[1].metric("Engagements", format_number(engagements), engagement_delta)
metric_cols[2].metric("New followers", f"+{format_number(follower_change)}", "vs. start of period")
metric_cols[3].metric("Engagement rate", f"{engagement_rate:.1f}%", "reach-based")

st.divider()
left, right = st.columns([1.65, 1], gap="large")

with left:
    st.subheader("Momentum")
    chart_data = selected_daily.melt(
        id_vars="date", value_vars=["Reach", "Engagements"], var_name="Metric", value_name="Value"
    )
    chart = px.line(
        chart_data,
        x="date",
        y="Value",
        color="Metric",
        markers=True,
        color_discrete_map={"Reach": "#7c3aed", "Engagements": "#f97316"},
    )
    chart.update_layout(
        height=330,
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title_text="",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    chart.update_yaxes(gridcolor="#e5e7eb", title="")
    chart.update_xaxes(title="")
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

with right:
    st.subheader("What stands out")
    top_post = selected_posts.sort_values("Reach", ascending=False).iloc[0] if not selected_posts.empty else None
    if top_post is not None:
        st.success(
            f"**{top_post['Type']} is leading reach**\n\n"
            f"{top_post['Caption']} reached **{format_number(top_post['Reach'])}** people."
        )
    st.info(
        f"**Save-worthy content is working**\n\n"
        f"Your top post generated {format_number(int(selected_posts['Saves'].sum()))} saves in this period."
    )
    st.warning(
        "**Next experiment**\n\n"
        "Repeat the winning Reel format with a direct question in the first line of the caption."
    )

st.divider()
performance_tab, chat_tab = st.tabs(["Content performance", "Ask your copilot"])

with performance_tab:
    st.subheader("Top content")
    if selected_posts.empty:
        st.info("No posts were published in this reporting window.")
    else:
        display_posts = selected_posts.sort_values("Reach", ascending=False).copy()
        display_posts["Published"] = display_posts["Published"].dt.strftime("%b %-d")
        display_posts["Reach"] = display_posts["Reach"].map(format_number)
        display_posts["Likes"] = display_posts["Likes"].map(format_number)
        display_posts["Saves"] = display_posts["Saves"].map(format_number)
        display_posts["Shares"] = display_posts["Shares"].map(format_number)
        st.dataframe(
            display_posts[["Published", "Type", "Caption", "Reach", "Likes", "Saves", "Shares"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Published": "Date",
                "Type": "Format",
                "Caption": st.column_config.TextColumn("Caption", width="large"),
                "Reach": "Reach",
                "Likes": "Likes",
                "Saves": "Saves",
                "Shares": "Shares",
            },
        )

with chat_tab:
    st.subheader("Ask about your growth")
    st.caption("Ask for a diagnosis, content ideas, or a plan for your next week.")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "I’m ready to help you turn these numbers into a plan. "
                    "Try asking: “What should I post next week?”"
                ),
            }
        ]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about reach, content, or your next experiment"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Reviewing your account data..."):
                try:
                    response = ask_gemini(build_context(daily_data, post_data, start_date, end_date) + f"\n\n{prompt}")
                except RuntimeError as error:
                    response = f"I couldn't reach Gemini right now. {error}"
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
