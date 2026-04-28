# streamlit is the only dependency here — all UI components are built using its markdown renderer
import streamlit as st


# this function injects a large block of custom CSS into the page
# CSS (Cascading Style Sheets) controls how everything looks — colours, fonts, spacing, borders
# st.markdown with unsafe_allow_html=True is how streamlit lets us write raw HTML and CSS
def apply_global_styles():
    st.markdown(
        """
        <style>
        /* streamlit generates its own sidebar navigation from the pages/ folder —
           we hide it here because we've built our own custom navigation links instead */
        [data-testid="stSidebarNav"] {
            display: none;
        }

        /* dark gradient background for the whole app — goes from near-black to a deep navy */
        .stApp {
            background: linear-gradient(135deg, #080b12 0%, #111827 45%, #0f172a 100%);
            color: #f8fafc;
        }

        /* the sidebar gets a slightly different (deeper) dark background
           and a faint border on the right to separate it from the main content */
        section[data-testid="stSidebar"] {
            background: #020617;
            border-right: 1px solid rgba(148, 163, 184, 0.15);
        }

        /* control the padding around the main content area and cap its maximum width
           so very wide screens don't stretch everything out unreadably */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* make the main page title (h1) large and heavy so it commands attention */
        h1 {
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.04em;  /* tighten the spacing between letters slightly */
        }

        /* subheadings are also bold but a little lighter than the main title */
        h2, h3 {
            font-weight: 750 !important;
            letter-spacing: -0.03em;
        }

        /* the hero card is the large banner at the top of each page
           it uses a warm orange-to-blue gradient tint to give it a premium feel */
        .hero-card {
            padding: 2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(249,115,22,0.18), rgba(37,99,235,0.16));
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 20px 60px rgba(0,0,0,0.35);  /* deep shadow for depth */
            margin-bottom: 1.5rem;
        }

        /* metric cards are the small stat boxes used throughout the app
           they have a dark semi-transparent background and a subtle border */
        .metric-card {
            padding: 1.2rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 8px 26px rgba(0,0,0,0.25);
            height: 100%;  /* makes all cards in the same row the same height */
        }

        /* the small uppercase label above the number in each metric card */
        .metric-label {
            color: #94a3b8;       /* muted grey-blue */
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        /* the big number in the centre of the metric card */
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #f8fafc;       /* near-white so it stands out against the dark background */
        }

        /* the small helper text below the number */
        .metric-help {
            color: #cbd5e1;
            font-size: 0.9rem;
        }

        /* glass cards are the wider content cards used for descriptions and interpretation
           they're slightly more transparent than metric cards, giving a frosted-glass look */
        .glass-card {
            padding: 1.35rem;
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.72);  /* 72% opacity — more see-through than metric cards */
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 12px 35px rgba(0,0,0,0.28);
            height: 100%;
        }

        /* pills are the small tag chips shown in the page header (e.g. "SHAP", "Ridge Model")
           border-radius: 999px makes them fully rounded on each end */
        .pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.14);  /* faint blue background */
            color: #60a5fa;                          /* bright blue text */
            border: 1px solid rgba(96, 165, 250, 0.25);
            font-weight: 700;
            font-size: 0.8rem;
            margin-right: 0.4rem;   /* small gap between pills when there are multiple */
            margin-bottom: 0.4rem;
        }

        /* the thin horizontal line used to separate sections of a page */
        .section-divider {
            margin: 2rem 0;
            border-top: 1px solid rgba(148, 163, 184, 0.16);
        }

        /* the large bold title text at the top of the sidebar */
        .sidebar-title {
            font-size: 2rem;
            font-weight: 900;
            line-height: 1.1;   /* tighter line height so the two-line title looks compact */
            color: white;
        }

        /* the smaller subtitle line beneath the sidebar title */
        .sidebar-subtitle {
            color: #94a3b8;
            font-size: 0.85rem;
        }

        /* style all streamlit buttons with rounded corners and an orange gradient */
        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(249,115,22,0.35);
            background: linear-gradient(135deg, #f97316, #ea580c);
            color: white;
            font-weight: 700;
        }

        /* style the dropdown select boxes with a rounded dark background to match the theme */
        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 14px;
            background-color: rgba(15,23,42,0.9);
        }

        </style>
        """,
        unsafe_allow_html=True,  # this flag is required to allow raw HTML/CSS through streamlit
    )


# this function renders the left-hand sidebar that appears on every page
# it contains the app logo/title and all the navigation links
def render_sidebar():
    # inject the branding block at the top of the sidebar using raw HTML
    # unsafe_allow_html=True lets us use custom CSS classes defined above
    st.sidebar.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
            <div class="sidebar-title">🏀 NBA<br>Analytics</div>
            <p class="sidebar-subtitle">Final Year Project | ML System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # a horizontal rule to visually separate the branding from the navigation links
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")

    # page_link creates a clickable link in the sidebar that navigates to another page
    # the path is relative to the app root, matching the files in the pages/ folder
    st.sidebar.page_link("app.py", label="🏠 Home")
    st.sidebar.page_link("pages/win_predictor.py", label="📊 Win Predictor")
    st.sidebar.page_link("pages/score_predictor.py", label="📈 Score Predictor")
    st.sidebar.page_link("pages/player_predictor.py", label="👤 Player Predictor")
    st.sidebar.page_link("pages/model_insights.py", label="📉 Model Insights")
    st.sidebar.page_link("pages/explainability.py", label="🔍 Explainability")
    st.sidebar.page_link("pages/methodology.py", label="🧪 Methodology")

    # another horizontal rule at the bottom, then a green status message
    st.sidebar.markdown("---")
    st.sidebar.success("All models loaded successfully")


# this function renders the large banner card at the top of each page
# it takes an icon, a title, a subtitle line, and an optional list of tag strings
def page_header(icon, title, subtitle, tags=None):
    # default to an empty list if no tags are passed in
    tags = tags or []

    # build the HTML for each tag pill by wrapping it in a <span> with the pill CSS class
    tag_html = "".join([f"<span class='pill'>{tag}</span>" for tag in tags])

    # inject the hero card as raw HTML so we can use our custom CSS classes
    # f-strings let us drop Python variables directly into the HTML string
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


# renders a small dark stat card with a label, a large value, and an optional help line
# used across every page for things like "Best F1: 0.699" or "Top Feature: pts_roll_mean_5"
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


# renders a wider frosted-glass card with a heading and a paragraph of body text
# used for interpretation summaries, methodology descriptions, and explanation sections
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


# renders a thin horizontal divider line between sections of a page
# it uses our custom CSS class rather than streamlit's built-in st.divider()
# so the styling stays consistent with the rest of the dark theme
def divider():
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
