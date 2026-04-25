import streamlit as st


# ============================================================
# GLOBAL STYLES
# ============================================================

def apply_global_styles():
    st.markdown(
        """
        <style>
        /* Hide default Streamlit page navigation */
        [data-testid="stSidebarNav"] {
            display: none;
        }

        /* App background */
        .stApp {
            background: linear-gradient(135deg, #080b12 0%, #111827 45%, #0f172a 100%);
            color: #f8fafc;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: #020617;
            border-right: 1px solid rgba(148, 163, 184, 0.15);
        }

        /* Page spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* Typography */
        h1 {
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.04em;
        }

        h2, h3 {
            font-weight: 750 !important;
            letter-spacing: -0.03em;
        }

        /* Hero card */
        .hero-card {
            padding: 2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(249,115,22,0.18), rgba(37,99,235,0.16));
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 20px 60px rgba(0,0,0,0.35);
            margin-bottom: 1.5rem;
        }

        /* Metric cards */
        .metric-card {
            padding: 1.2rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 8px 26px rgba(0,0,0,0.25);
            height: 100%;
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #f8fafc;
        }

        .metric-help {
            color: #cbd5e1;
            font-size: 0.9rem;
        }

        /* Glass cards */
        .glass-card {
            padding: 1.35rem;
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 12px 35px rgba(0,0,0,0.28);
            height: 100%;
        }

        /* Pills */
        .pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.14);
            color: #60a5fa;
            border: 1px solid rgba(96, 165, 250, 0.25);
            font-weight: 700;
            font-size: 0.8rem;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }

        /* Divider */
        .section-divider {
            margin: 2rem 0;
            border-top: 1px solid rgba(148, 163, 184, 0.16);
        }

        /* Sidebar title */
        .sidebar-title {
            font-size: 2rem;
            font-weight: 900;
            line-height: 1.1;
            color: white;
        }

        .sidebar-subtitle {
            color: #94a3b8;
            font-size: 0.85rem;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(249,115,22,0.35);
            background: linear-gradient(135deg, #f97316, #ea580c);
            color: white;
            font-weight: 700;
        }

        /* Select boxes */
        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 14px;
            background-color: rgba(15,23,42,0.9);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR (CLICKABLE NAVIGATION)
# ============================================================

def render_sidebar():
    st.sidebar.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
            <div class="sidebar-title">🏀 NBA<br>Analytics</div>
            <p class="sidebar-subtitle">Final Year Project | ML System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")

    st.sidebar.page_link("app.py", label="🏠 Home")
    st.sidebar.page_link("pages/win_predictor.py", label="📊 Win Predictor")
    st.sidebar.page_link("pages/score_predictor.py", label="📈 Score Predictor")
    st.sidebar.page_link("pages/player_predictor.py", label="👤 Player Predictor")
    st.sidebar.page_link("pages/model_insights.py", label="📉 Model Insights")
    st.sidebar.page_link("pages/explainability.py", label="🔍 Explainability")
    st.sidebar.page_link("pages/methodology.py", label="🧪 Methodology")

    st.sidebar.markdown("---")
    st.sidebar.success("All models loaded successfully")


# ============================================================
# PAGE HEADER
# ============================================================

def page_header(icon, title, subtitle, tags=None):
    tags = tags or []
    tag_html = "".join([f"<span class='pill'>{tag}</span>" for tag in tags])

    st.markdown(
        f"""
        <div class="hero-card">
            <h1>{icon} {title}</h1>
            <p style="font-size:1.05rem;color:#cbd5e1;max-width:900px;">{subtitle}</p>
            <div>{tag_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# METRIC CARD
# ============================================================

def metric_card(label, value, help_text=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# GLASS CARD
# ============================================================

def glass_card(title, body):
    st.markdown(
        f"""
        <div class="glass-card">
            <h3>{title}</h3>
            <p style="color:#cbd5e1;line-height:1.65;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DIVIDER
# ============================================================

def divider():
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)