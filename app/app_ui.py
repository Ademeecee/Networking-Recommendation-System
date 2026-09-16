"""
CyNam Ecosystem Matchmaker
Streamlit User Interface
"""

from pathlib import Path
import html

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "artifacts"
ASSETS_DIR = BASE_DIR / "assets"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CyNam Ecosystem Matchmaker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

    /* --------------------------------------------------------
       Global
    -------------------------------------------------------- */

    .stApp {
        background: #f5f8fc;
    }

    .main {
        background: #f5f8fc;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* --------------------------------------------------------
       Sidebar
    -------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e3eaf3;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    /* --------------------------------------------------------
       Main content width
    -------------------------------------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* --------------------------------------------------------
       Header
    -------------------------------------------------------- */

    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border-bottom: 1px solid #e3eaf3;
        padding: 18px 28px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(30, 70, 120, 0.05);
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .shield {
        width: 54px;
        height: 62px;
        border-radius: 14px;
        background: linear-gradient(145deg, #1689f5, #1765c0);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 28px;
        box-shadow: 0 8px 18px rgba(22, 137, 245, 0.25);
    }

    .brand-title {
        font-size: 30px;
        font-weight: 800;
        color: #102a56;
        line-height: 1.1;
    }

    .brand-subtitle {
        font-size: 14px;
        color: #6680a5;
        margin-top: 5px;
    }

    .brand-right {
        color: #1676df;
        font-size: 14px;
        font-weight: 600;
        text-align: right;
    }

    /* --------------------------------------------------------
       Section headings
    -------------------------------------------------------- */

    .section-heading {
        color: #102a56;
        font-size: 24px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 2px;
    }

    .section-subheading {
        color: #7186a5;
        font-size: 14px;
        margin-bottom: 16px;
    }

    /* --------------------------------------------------------
       Profile card
    -------------------------------------------------------- */

    .profile-card {
        background: #ffffff;
        border: 1px solid #e0e8f2;
        border-radius: 16px;
        padding: 22px 26px;
        box-shadow: 0 4px 18px rgba(30, 70, 120, 0.06);
        margin-bottom: 26px;
    }

    .profile-avatar {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: linear-gradient(145deg, #1689f5, #2875d5);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 21px;
        font-weight: 800;
        margin-right: 16px;
    }

    .profile-name {
        font-size: 21px;
        font-weight: 800;
        color: #102a56;
        margin-bottom: 3px;
    }

    .profile-role {
        color: #607da5;
        font-size: 14px;
    }

    .profile-label {
        color: #7890b0;
        font-size: 12px;
        margin-bottom: 4px;
    }

    .profile-value {
        color: #102a56;
        font-size: 15px;
        font-weight: 700;
    }

    /* --------------------------------------------------------
       Recommendation cards
    -------------------------------------------------------- */

    .recommendation-card {
        background: #ffffff;
        border: 1px solid #dfe8f3;
        border-radius: 16px;
        padding: 20px;
        min-height: 355px;
        box-shadow: 0 5px 18px rgba(30, 70, 120, 0.07);
        transition: all 0.2s ease;
    }

    .recommendation-card:hover {
        transform: translateY(-3px);
        border-color: #82bdf7;
        box-shadow: 0 10px 28px rgba(22, 118, 223, 0.13);
    }

    .match-badge {
        display: inline-block;
        background: linear-gradient(135deg, #14c99a, #10b981);
        color: white;
        font-size: 13px;
        font-weight: 800;
        padding: 6px 13px;
        border-radius: 20px;
        margin-bottom: 14px;
    }

    .member-avatar {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background: #edf5ff;
        color: #1676df;
        border: 2px solid #d5e8fc;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 17px;
        float: left;
        margin-right: 12px;
    }

    .member-name {
        color: #102a56;
        font-size: 17px;
        font-weight: 800;
        padding-top: 3px;
        margin-bottom: 2px;
    }

    .member-job {
        color: #6680a5;
        font-size: 13px;
        min-height: 20px;
    }

    .member-org {
        clear: both;
        padding-top: 16px;
        color: #6780a2;
        font-size: 13px;
        margin-bottom: 12px;
    }

    .badge {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: 700;
        margin-right: 4px;
        margin-bottom: 8px;
    }

    .badge-blue {
        background: #eaf4ff;
        color: #1676df;
    }

    .badge-green {
        background: #e8faf4;
        color: #07966d;
    }

    .badge-purple {
        background: #f1edff;
        color: #7457d9;
    }

    .why-match {
        border-top: 1px solid #e8edf4;
        margin-top: 8px;
        padding-top: 13px;
        color: #5d7496;
        font-size: 12.5px;
        line-height: 1.55;
    }

    .why-title {
        color: #102a56;
        font-weight: 800;
    }

    /* --------------------------------------------------------
       Analytics section
    -------------------------------------------------------- */

    .analytics-card {
        background: #ffffff;
        border: 1px solid #dfe8f3;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 18px rgba(30, 70, 120, 0.06);
    }

    /* --------------------------------------------------------
       Footer
    -------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #8194af;
        font-size: 12px;
        padding: 30px 0 10px 0;
    }

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data_path = DATA_DIR / "clustered_member_profiles.csv"

    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.stop()

    return pd.read_csv(data_path)


df = load_data()


# ============================================================
# FEATURE MATRICES
# ============================================================

tfidf_cols = [
    c for c in df.columns
    if c.startswith("tfidf_")
]

sector_cols = [
    c for c in df.columns
    if c.startswith("sec_")
]

X_tfidf = df[tfidf_cols].values
X_sector = df[sector_cols].values
X_sen = df["norm_seniority"].values

orgs = (
    df["organization"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .values
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_text(value, fallback="Unknown"):
    """Convert a dataframe value to safe display text."""

    if pd.isna(value):
        return fallback

    value = str(value).strip()

    if not value:
        return fallback

    return html.escape(value)


def initials(name):
    """Return member initials."""

    if not name:
        return "?"

    parts = str(name).strip().split()

    if len(parts) == 1:
        return parts[0][:2].upper()

    return (
        parts[0][0] + parts[-1][0]
    ).upper()


def calculate_rationale(
    mode,
    overlap,
    target_sector,
    sector,
    target_seniority,
    recommended_seniority,
    target_sen,
    recommended_sen
):
    """Generate an explanation for the recommendation."""

    if mode == "Homophily (Peer-to-Peer)":

        if (
            target_seniority != "Unknown"
            and recommended_seniority != "Unknown"
        ):

            return (
                f"Strong thematic alignment with "
                f"{overlap:.0f}% shared event-theme similarity "
                f"and a similar seniority profile."
            )

        return (
            f"Strong thematic alignment with "
            f"{overlap:.0f}% shared event-theme similarity."
        )

    if mode == "Mentorship & Guidance":

        if (
            target_seniority != "Unknown"
            and recommended_seniority != "Unknown"
            and recommended_sen > target_sen
        ):

            return (
                f"Strong thematic alignment with "
                f"{overlap:.0f}% shared event-theme similarity, "
                f"combined with higher seniority "
                f"({recommended_seniority} vs "
                f"{target_seniority})."
            )

        return (
            f"Selected for mentorship based on "
            f"{overlap:.0f}% shared event-theme similarity "
            f"and the overall mentorship scoring model."
        )

    # Cross-sector
    if (
        target_sector != "Unknown"
        and sector != "Unknown"
        and target_sector != sector
    ):

        return (
            f"Cross-sector opportunity between "
            f"{target_sector} and {sector}, "
            f"with {overlap:.0f}% shared event-theme similarity."
        )

    return (
        f"Selected for cross-sector networking based on "
        f"{overlap:.0f}% shared event-theme similarity; "
        f"sector data is unavailable for a direct comparison."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="brand-header">
<div class="brand-left">
<div class="shield">🛡️</div>
<div>
<div class="brand-title">CyNam Ecosystem Matchmaker</div>
<div class="brand-subtitle">AI-Powered Collaborative Intelligence for Gloucestershire's Cyber &amp; Innovation Community</div>
</div>
</div>
<div class="brand-right">👥 Stronger Connections<br>🌐 A Safer Tomorrow</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
<div style="
    color:#102a56;
    font-size:20px;
    font-weight:800;
    margin-bottom:20px;
">
⚙️ Matchmaking Configuration
</div>
""",
    unsafe_allow_html=True
)


member_names = (
    df[df["job_title"].notna()]
    ["full_name"]
    .dropna()
    .astype(str)
    .str.title()
    .unique()
    .tolist()
)


if not member_names:
    st.error("No member profiles are available.")
    st.stop()


selected_name = st.sidebar.selectbox(
    "Select Member Profile",
    member_names,
    index=min(12, len(member_names) - 1)
)


matching_rows = df[
    df["full_name"]
    .fillna("")
    .astype(str)
    .str.lower()
    == selected_name.lower()
]


if matching_rows.empty:
    st.error("Selected member could not be found.")
    st.stop()


selected_idx = matching_rows.index[0]
target_row = df.iloc[selected_idx]


mode = st.sidebar.radio(
    "Networking Objective",
    [
        "Homophily (Peer-to-Peer)",
        "Mentorship & Guidance",
        "Cross-Sector Innovation"
    ]
)


top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=3,
    max_value=10,
    value=3
)


serendipity = st.sidebar.slider(
    "Serendipity Discovery Rate (%)",
    min_value=0,
    max_value=25,
    value=5
) / 100.0


st.sidebar.divider()


st.sidebar.markdown(
    """
<div style="
    background:#f1f7ff;
    border-radius:12px;
    padding:15px;
    color:#5c7598;
    font-size:12px;
    line-height:1.5;
">
💡 <b style="color:#173b70;">How matching works</b><br><br>
Recommendations combine thematic similarity,
sector relationships, seniority and community
clustering to identify potential connections.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# ACTIVE PROFILE
# ============================================================

st.markdown(
    '<div class="section-heading">👤 Your Active Profile</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-subheading">Current member profile used as the basis for recommendations</div>',
    unsafe_allow_html=True
)


profile_name = safe_text(
    target_row["full_name"],
    "Unknown Member"
).title()


profile_role = safe_text(
    target_row["job_title"],
    "Practitioner"
)


profile_org = safe_text(
    target_row["organization"],
    "Independent"
)


profile_seniority = safe_text(
    target_row["seniority_level_name"],
    "Unknown"
)


profile_initials = initials(
    str(target_row["full_name"])
    if pd.notna(target_row["full_name"])
    else ""
)


st.markdown(
    f"""
<div class="profile-card">
<div style="display:flex;align-items:center;">
<div class="profile-avatar">{profile_initials}</div>
<div style="min-width:230px;">
<div class="profile-name">{profile_name}</div>
<div class="profile-role">{profile_role}</div>
</div>
<div style="flex:1;border-left:1px solid #e5ebf3;padding-left:28px;margin-left:25px;">
<div class="profile-label">Role</div>
<div class="profile-value">💼 {profile_role}</div>
</div>
<div style="flex:1;border-left:1px solid #e5ebf3;padding-left:28px;">
<div class="profile-label">Organisation</div>
<div class="profile-value">🏢 {profile_org}</div>
</div>
<div style="flex:1;border-left:1px solid #e5ebf3;padding-left:28px;">
<div class="profile-label">Seniority Tier</div>
<div class="profile-value">📊 {profile_seniority}</div>
</div>
</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# CALCULATE SIMILARITIES
# ============================================================

sim_int = cosine_similarity(
    X_tfidf[selected_idx:selected_idx + 1],
    X_tfidf
).flatten()


sim_sec = cosine_similarity(
    X_sector[selected_idx:selected_idx + 1],
    X_sector
).flatten()


target_sector = (
    str(target_row["clean_sector"]).strip()
    if pd.notna(target_row["clean_sector"])
    else "Unknown"
)


# ============================================================
# SECTOR SCORING
# ============================================================

if mode == "Cross-Sector Innovation":

    if target_sector != "Unknown":

        sim_sec = 1.0 - sim_sec

    else:

        # Unknown sector = neutral sector contribution.
        sim_sec = np.zeros_like(sim_sec)


# ============================================================
# SENIORITY SCORING
# ============================================================

target_sen = X_sen[selected_idx]


if mode == "Mentorship & Guidance":

    # Higher seniority is rewarded.
    sen_score = np.maximum(
        0,
        X_sen - target_sen
    )

else:

    # Similar seniority is rewarded.
    sen_score = (
        1.0
        - np.abs(X_sen - target_sen)
    )


# ============================================================
# COMPOSITE SCORE
# ============================================================

scores = (
    0.50 * sim_int
    + 0.30 * sim_sec
    + 0.20 * sen_score
)


# ============================================================
# SERENDIPITY
# ============================================================

if serendipity > 0:

    different_cluster = (
        df["cluster"]
        != target_row["cluster"]
    ).values

    scores += (
        np.random.uniform(
            0,
            serendipity,
            size=len(scores)
        )
        * different_cluster
    )


# ============================================================
# BUSINESS CONSTRAINTS
# ============================================================

# Never recommend the active member.
scores[selected_idx] = -np.inf


# Never recommend someone from the same organisation.
target_org = orgs[selected_idx]

if target_org:

    scores[
        orgs == target_org
    ] = -np.inf


# ============================================================
# SELECT TOP RECOMMENDATIONS
# ============================================================

eligible_indices = np.where(
    np.isfinite(scores)
)[0]


top_indices = eligible_indices[
    np.argsort(
        scores[eligible_indices]
    )[::-1][:top_n]
]


# ============================================================
# RECOMMENDATION HEADER
# ============================================================

st.markdown(
    '<div class="section-heading">👥 Recommended Connections</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-subheading">Curated using AI to help identify relevant connections for your networking objective</div>',
    unsafe_allow_html=True
)


# ============================================================
# RECOMMENDATION CARDS
# ============================================================

if len(top_indices) == 0:

    st.warning(
        "No eligible recommendations were found."
    )

else:

    card_cols = st.columns(
        len(top_indices)
    )

    for i, idx in enumerate(top_indices):

        match_row = df.iloc[idx]

        # --------------------------------------------------------
        # Values
        # --------------------------------------------------------

        full_name_raw = (
            str(match_row["full_name"])
            if pd.notna(match_row["full_name"])
            else "Unknown Member"
        )

        full_name = safe_text(
            full_name_raw,
            "Unknown Member"
        ).title()

        job_title = safe_text(
            match_row["job_title"],
            "Practitioner"
        )

        organization = safe_text(
            match_row["organization"],
            "Independent"
        )

        sector = safe_text(
            match_row["clean_sector"],
            "Unknown"
        )

        seniority = safe_text(
            match_row["seniority_level_name"],
            "Unknown"
        )

        recommended_sen = X_sen[idx]

        overlap = (
            float(sim_int[idx])
            * 100
        )

        # Keep the visual score between 0 and 100.
        match_score = max(
            0,
            min(
                100,
                float(scores[idx]) * 100
            )
        )

        member_initials = initials(
            full_name_raw
        )

        # --------------------------------------------------------
        # Rationale
        # --------------------------------------------------------

        rationale = calculate_rationale(
            mode=mode,
            overlap=overlap,
            target_sector=target_sector,
            sector=str(
                match_row["clean_sector"]
            ).strip()
            if pd.notna(match_row["clean_sector"])
            else "Unknown",
            target_seniority=str(
                target_row["seniority_level_name"]
            ).strip()
            if pd.notna(
                target_row["seniority_level_name"]
            )
            else "Unknown",
            recommended_seniority=str(
                match_row["seniority_level_name"]
            ).strip()
            if pd.notna(
                match_row["seniority_level_name"]
            )
            else "Unknown",
            target_sen=target_sen,
            recommended_sen=recommended_sen
        )

        rationale = html.escape(
            rationale
        )

        # --------------------------------------------------------
        # Card
        # --------------------------------------------------------

        card_html = (
            f'<div class="recommendation-card">'
            f'<div class="match-badge">'
            f'{match_score:.0f}% Match'
            f'</div>'

            f'<div>'
            f'<div class="member-avatar">'
            f'{member_initials}'
            f'</div>'

            f'<div class="member-name">'
            f'{full_name}'
            f'</div>'

            f'<div class="member-job">'
            f'{job_title}'
            f'</div>'
            f'</div>'

            f'<div class="member-org">'
            f'🏢 {organization}'
            f'</div>'

            f'<span class="badge badge-blue">'
            f'🛡️ {seniority}'
            f'</span>'

            f'<span class="badge badge-green">'
            f'🏷️ {sector}'
            f'</span>'

            f'<div class="why-match">'
            f'<span class="why-title">'
            f'💡 Why this match?'
            f'</span><br>'
            f'{rationale}'
            f'</div>'

            f'</div>'
        )

        with card_cols[i]:

            st.markdown(
                card_html,
                unsafe_allow_html=True
            )

            if st.button(
                f"🤝 Connect with {full_name.split()[0]}",
                key=f"connect_{idx}",
                use_container_width=True
            ):

                st.success(
                    f"Connection request prepared for "
                    f"{full_name}."
                )


# ============================================================
# ECOSYSTEM HEALTH & STRATEGIC INSIGHTS
# ============================================================

st.divider()

st.subheader("📊 Ecosystem Health & Strategic Insights")

st.caption(
    "Explore the broader picture of the Gloucestershire cyber "
    "and innovation ecosystem"
)

# ============================================================
# VISUALISATIONS
# ============================================================

col1, col2 = st.columns(2)

# ------------------------------------------------------------
# ECOSYSTEM CLUSTERS
# ------------------------------------------------------------

with col1:

    st.markdown("### 🔵 Ecosystem Clusters")

    st.caption(
        "PCA 2D Member Space across 5 Community Cohorts"
    )

    pca_image = ASSETS_DIR / "cluster_pca_projection.png"

    if pca_image.exists():
        st.image(
            str(pca_image),
            caption="Member Ecosystem Segmentation"
        )
    else:
        st.warning(
            "PCA cluster visualisation could not be found."
        )


# ------------------------------------------------------------
# THEMATIC ENGAGEMENT
# ------------------------------------------------------------

with col2:

    st.markdown("### 📈 Thematic Engagement Trends")

    st.caption(
        "Thematic engagement across CyNam events"
    )

    engagement_image = (
        ASSETS_DIR / "thematic_engagement_trends.png"
    )

    if engagement_image.exists():
        st.image(
            str(engagement_image),
            caption="Aggregate Thematic Event Engagement"
        )
    else:
        st.warning(
            "Thematic engagement visualisation could not be found."
        )


# ============================================================
# ECOSYSTEM INSIGHT
# ============================================================

st.markdown("---")

st.markdown("### 💡 Ecosystem Insight")

st.info(
    "The visualisations provide an overview of member segmentation "
    "and thematic engagement across the Gloucestershire cyber and "
    "innovation ecosystem. The clustering view highlights distinct "
    "member cohorts, while the thematic view shows the areas "
    "attracting the greatest event engagement."
)
