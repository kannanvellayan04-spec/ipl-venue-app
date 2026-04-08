import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.calibration import calibration_curve



# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="IPL Venue Difficulty & Thriller Analysis",
    page_icon="🏟️",
    layout="wide"
)
# -------------------------
# Global Mode Selector
# -------------------------
app_mode = st.sidebar.radio(
    "Select Mode",
    ["Venue Intelligence", "Match Outcome Pattern"]
)

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ipl_venue_clusters_freq.csv")

df = load_data()


# -------------------------
# Helpers
# -------------------------
def pct(x):
    return float(x) * 100

def confidence_label(match_count):
    if match_count >= 60:
        return "🟢 High"
    elif match_count >= 30:
        return "🟡 Medium"
    else:
        return "🔴 Low"

def toss_takeaway(chasing_adv):
    if chasing_adv >= 8:
        return "🏃 Chase-friendly venue"
    elif chasing_adv <= -8:
        return "🛡️ Bat-first friendly venue"
    else:
        return "⚖️ Neutral toss impact"

def percentile_of_value(series, value):
    """Return percentile (0 to 100) of 'value' inside 'series'."""
    if len(series) == 0:
        return 50.0
    return float((series <= value).mean() * 100)

def badge_from_percentile(p, reverse=False):
    """
    Convert percentile to badge text.
    reverse=False: high percentile = higher tendency
    reverse=True : low percentile = better (ex stability)
    """
    if reverse:
        p = 100 - p

    if p >= 90:
        return "Elite"
    elif p >= 75:
        return "Strong"
    elif p >= 50:
        return "Above Avg"
    else:
        return "Average"

def badge_chip(label, level):
    """
    Render clean badge chips (no overflow).
    """
    colors = {
        "Elite": "#7C3AED",      # purple
        "Strong": "#2563EB",     # blue
        "Above Avg": "#16A34A",  # green
        "Average": "#6B7280"     # gray
    }
    bg = colors.get(level, "#6B7280")

    st.markdown(
        f"""
        <div style="padding:12px;border-radius:16px;background:#F9FAFB;border:1px solid #E5E7EB;">
            <div style="font-size:13px;color:#374151;margin-bottom:6px;"><b>{label}</b></div>
            <span style="
                display:inline-block;
                padding:6px 14px;
                border-radius:999px;
                background:{bg};
                color:white;
                font-weight:700;
                font-size:14px;
            ">{level}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

def build_identity_card(venue_row):
    thriller = pct(venue_row["thriller_pct"])
    dominance_var = float(venue_row["dominance_variance"])
    bat_first = pct(venue_row["bat_first_win_pct"])
    chasing = pct(venue_row["chasing_win_pct"])
    chasing_adv = chasing - bat_first

    insights = []

    # Thriller insight
    if thriller >= 40:
        insights.append("🔥 Very high thriller venue (many close games).")
    elif thriller >= 30:
        insights.append("🎯 Good chance of close matches here.")
    else:
        insights.append("📉 Relatively fewer thrillers vs other venues.")

    # Chasing / batting insight
    if chasing_adv >= 8:
        insights.append("🏃 Strong chasing advantage — chasing teams win much more.")
    elif chasing_adv <= -8:
        insights.append("🛡️ Strong batting-first advantage — defending is easier here.")
    else:
        insights.append("⚖️ Balanced venue — both chasing & batting-first are similar.")

    # Variance insight
    if dominance_var >= 1.5:
        insights.append("🎲 Highly unpredictable venue — results swing a lot.")
    elif dominance_var >= 1.0:
        insights.append("🔄 Moderate unpredictability — mixed match outcomes.")
    else:
        insights.append("📌 Predictable patterns — outcomes are more consistent.")

    return insights[:3]

# -------------------------
# Reference dataset for percentile badges
# -------------------------
MIN_MATCHES = 15
df_ref = df[df["match_count"] >= MIN_MATCHES].copy()
if app_mode == "Venue Intelligence":

    # ==============================
    # VENUE INTELLIGENCE MODULE
    # ==============================

    st.title("Global IPL Venue Behaviour Analysis (2008–Present)")
    st.write("Profiling IPL venues worldwide based on dominance structure, thrillers, and toss impact patterns.")

    # ------------------------------
    # Sidebar Control Panel
    # ------------------------------
    st.sidebar.markdown("## 🎮 Control Panel")

    stadiums = df["stadium"].sort_values().unique()
    selected_stadium = st.sidebar.selectbox("Select a stadium", stadiums)

    mode = st.sidebar.radio(
        "Choose Mode",
        ["📌 Single Venue Insights", "⚔️ Venue Comparison Mode"]
    )

    venue_row = df[df["stadium"] == selected_stadium].iloc[0]

    # ----------- Clean Cluster Rename (Non-destructive) ----------
    cluster_map = {
        "Chaotic / Thriller Venue": "High Variability / High Thriller Venue",
        "Predictable / One-Sided Venue": "Stable / Directional Venue"
    }

    cluster_label_clean = cluster_map.get(
        venue_row["cluster_label"],
        venue_row["cluster_label"]
    )

    thriller = pct(venue_row["thriller_pct"])
    blowout = pct(venue_row["blowout_pct"])
    bat_first = pct(venue_row["bat_first_win_pct"])
    chasing = pct(venue_row["chasing_win_pct"])
    chasing_adv = chasing - bat_first
    match_count = int(venue_row["match_count"])

    # Sidebar summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Venue Summary")
    st.sidebar.write(f"**Cluster:** {cluster_label_clean}")
    st.sidebar.write(f"**Confidence:** {confidence_label(match_count)}")
    st.sidebar.write(f"**Takeaway:** {toss_takeaway(chasing_adv)}")

    # ==========================================================
    # MODE 1 – SINGLE VENUE INSIGHTS
    # ==========================================================
    if mode == "📌 Single Venue Insights":

        st.markdown(f"## 🏟️ {selected_stadium}")
        st.write(f"**Cluster Type:** {cluster_label_clean}")

        # Identity Card
        st.markdown("### 🧾 Venue Identity Card")
        st.info("\n".join(build_identity_card(venue_row)))

        # ------------------------------
        # Percentile Badges
        # ------------------------------
        st.markdown("---")
        st.markdown("### 🏅 Percentile Badges (vs Frequent Venues)")

        p_thriller = percentile_of_value(df_ref["thriller_pct"], venue_row["thriller_pct"])
        p_blowout = percentile_of_value(df_ref["blowout_pct"], venue_row["blowout_pct"])
        p_var = percentile_of_value(df_ref["dominance_variance"], venue_row["dominance_variance"])

        ref_chase = df_ref["chasing_win_pct"] - df_ref["bat_first_win_pct"]
        venue_chase = venue_row["chasing_win_pct"] - venue_row["bat_first_win_pct"]
        p_chase = percentile_of_value(ref_chase, venue_chase)

        badge_thriller = badge_from_percentile(p_thriller)
        badge_blowout = badge_from_percentile(p_blowout)
        badge_var = badge_from_percentile(p_var)
        badge_stability = badge_from_percentile(p_var, reverse=True)
        badge_chase = badge_from_percentile(p_chase)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: badge_chip("Thriller Level", badge_thriller)
        with c2: badge_chip("Blowout Level", badge_blowout)
        with c3: badge_chip("Unpredictability", badge_var)
        with c4: badge_chip("Stability", badge_stability)
        with c5: badge_chip("Chasing Bias", badge_chase)

        # ------------------------------
        # Key Metrics
        # ------------------------------
        st.markdown("---")
        st.subheader("📊 Key Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Matches Analysed", match_count)
        col2.metric("Thriller %", f"{thriller:.1f}%")
        col3.metric("Blowout %", f"{blowout:.1f}%")

        col4, col5, col6 = st.columns(3)
        col4.metric("Avg Dominance", f"{venue_row['avg_dominance']:.2f}")
        col5.metric("Dominance Variance", f"{venue_row['dominance_variance']:.2f}")
        col6.metric("Chasing Advantage", f"{chasing_adv:.1f}%")

        # ------------------------------
        # Interpretation Framework
        # ------------------------------
        st.markdown("---")
        st.subheader("📘 Interpretation Framework")

        st.info(
            "• High Thriller % + High Dominance Variance → High Volatility Venue\n\n"
            "• Low Variance + High Average Dominance → Directionally Stable Venue\n\n"
            "• Balanced Dominance (near zero) → Toss-Neutral Conditions"
        )

        # ------------------------------
        # Rankings Dashboard
        # ------------------------------
        st.markdown("---")
        st.subheader("🏆 Venue Rankings Dashboard")

        ranking_option = st.selectbox(
            "Choose ranking to view:",
            ["🔥 Thriller %", "🎲 Dominance variance", "🏏 Bat-first win %", "🏃 Chasing win %"]
        )

        if "Thriller" in ranking_option:
            rank_df = df.sort_values("thriller_pct", ascending=False)[["stadium", "cluster_label", "thriller_pct"]].copy()
            rank_df["thriller_pct"] = rank_df["thriller_pct"].apply(lambda x: f"{x*100:.1f}%")
            rank_df.columns = ["Stadium", "Cluster", "Thriller %"]

        elif "Dominance" in ranking_option:
            rank_df = df.sort_values("dominance_variance", ascending=False)[["stadium", "cluster_label", "dominance_variance"]].copy()
            rank_df.columns = ["Stadium", "Cluster", "Dominance Variance"]

        elif "Bat-first" in ranking_option:
            rank_df = df.sort_values("bat_first_win_pct", ascending=False)[["stadium", "cluster_label", "bat_first_win_pct"]].copy()
            rank_df["bat_first_win_pct"] = rank_df["bat_first_win_pct"].apply(lambda x: f"{x*100:.1f}%")
            rank_df.columns = ["Stadium", "Cluster", "Bat-first win %"]

        else:
            rank_df = df.sort_values("chasing_win_pct", ascending=False)[["stadium", "cluster_label", "chasing_win_pct"]].copy()
            rank_df["chasing_win_pct"] = rank_df["chasing_win_pct"].apply(lambda x: f"{x*100:.1f}%")
            rank_df.columns = ["Stadium", "Cluster", "Chasing win %"]

        st.dataframe(rank_df, hide_index=True, use_container_width=True)

    # ==========================================================
    # MODE 2 – VENUE COMPARISON
    # ==========================================================
    else:

        st.markdown("## ⚔️ Venue Comparison Mode")

        colA, colB = st.columns(2)

        with colA:
            venueA = st.selectbox("Select Venue A", stadiums, index=0)

        with colB:
            venueB = st.selectbox("Select Venue B", stadiums, index=1)

        rowA = df[df["stadium"] == venueA].iloc[0]
        rowB = df[df["stadium"] == venueB].iloc[0]

        left, right = st.columns(2)

        def render_card(title, row):
            clean_label = cluster_map.get(row["cluster_label"], row["cluster_label"])
            st.markdown(f"### 🏟️ {title}")
            st.write(f"Cluster: {clean_label}")
            st.metric("Thriller %", f"{pct(row['thriller_pct']):.1f}%")
            st.metric("Dominance Variance", f"{row['dominance_variance']:.2f}")
            st.metric("Chasing Advantage", f"{pct(row['chasing_win_pct'] - row['bat_first_win_pct']):.1f}%")

        with left:
            render_card(venueA, rowA)

        with right:
            render_card(venueB, rowB)


# ===================================================
# MODULE 2 : MATCH OUTCOME PATTERN

# ===================================================
# MODULE 2 : MATCH OUTCOME PATTERN (FINAL CALIBRATED)
# ===================================================

if app_mode == "Match Outcome Pattern":

    st.title("IPL Venue Intelligence System")
    st.subheader("Probabilistic Score Band & Match Structure Interpretation")

    # ---------------------------------------------------
    # LOAD FILES
    # ---------------------------------------------------

    @st.cache_resource
    def load_model():
        return joblib.load("ipl_score_band_model.pkl")

    @st.cache_data
    def load_data():
        venue_df = pd.read_csv("venue_master_insights.csv")
        team_df = pd.read_csv("team_master_insights.csv")
        active_venues = joblib.load("active_venues.pkl")
        model_features = joblib.load("model_features.pkl")
        return venue_df, team_df, active_venues, model_features

    @st.cache_data
    def load_league_sim():
        sim = pd.read_csv("full_league_simulation.csv")

        means = {
            "low_mean": sim["P_Low"].mean(),
            "comp_mean": sim["P_Competitive"].mean(),
            "high_mean": sim["P_High"].mean()
        }

        venue_rank = (
            sim.groupby("Venue")["P_High"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        return sim, means, venue_rank

    model = load_model()
    venue_df, team_df, ACTIVE_VENUES, model_features = load_data()
    league_sim, league_means, venue_rank = load_league_sim()

    ACTIVE_TEAMS = [
        "Chennai Super Kings",
        "Mumbai Indians",
        "Royal Challengers Bengaluru",
        "Kolkata Knight Riders",
        "Sunrisers Hyderabad",
        "Rajasthan Royals",
        "Delhi Capitals",
        "Punjab Kings",
        "Gujarat Titans",
        "Lucknow Super Giants"
    ]

    # ---------------------------------------------------
    # USER INPUT
    # ---------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        batting_team = st.selectbox("Batting Team", ACTIVE_TEAMS)

    with col2:
        bowling_team = st.selectbox("Bowling Team", ACTIVE_TEAMS)

    with col3:
        venue = st.selectbox("Venue", ACTIVE_VENUES)

    if batting_team == bowling_team:
        st.error("Batting and Bowling teams cannot be the same.")
        st.stop()

    # ---------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------

    def build_input(bat, bowl, venue):

        team_bat = team_df[team_df["batting_team"] == bat].iloc[0]
        team_bowl = team_df[team_df["batting_team"] == bowl].iloc[0]
        venue_row = venue_df[venue_df["venue"] == venue].iloc[0]

        input_dict = {
            "bat_avg_total_20": team_bat["avg_total"],
            "bat_high_rate_20": team_bat["high_score_rate"],
            "bowl_avg_conceded_20": team_bowl["avg_total"],
            "venue_avg_total": venue_row["avg_total"],
            "venue_high_rate": venue_row["high_score_rate"],
            "strength_diff": team_bat["avg_total"] - team_bowl["avg_total"],
            "venue_adjusted_bat": team_bat["avg_total"] - venue_row["avg_total"]
        }

        df = pd.DataFrame([input_dict])
        df = df[model_features]
        return df

    # ---------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------

    if st.button("Predict Score Band"):

        input_df = build_input(batting_team, bowling_team, venue)
        probs = model.predict_proba(input_df)[0]
        predicted_class = model.predict(input_df)[0]

        band_map = {
            0: "Low (<150)",
            1: "Competitive (151–185)",
            2: "High (>185)"
        }

        prob_df = pd.DataFrame({
            "Score Band": ["Low (<150)", "Competitive (151–185)", "High (>185)"],
            "Probability": np.round(probs, 3)
        })

        st.markdown("## 🎯 Predicted Score Band Probabilities")
        st.dataframe(prob_df)
        st.success(f"Most Probable Band: {band_map[predicted_class]}")

        # ===================================================
        # LEAGUE CONTEXT
        # ===================================================

        st.markdown("## 📊 League Context")

        delta_high = probs[2] - league_means["high_mean"]

        if abs(delta_high) < 0.02:
            shift_label = "Aligned with league norms"
        elif delta_high > 0.05:
            shift_label = "Strong high-scoring shift"
        elif delta_high < -0.05:
            shift_label = "Strong suppression shift"
        else:
            shift_label = "Moderate structural shift"

        st.write(f"High-scoring shift vs league average: {round(delta_high,3)}")
        st.write(f"Structural Classification: {shift_label}")

        venue_position = venue_rank[venue_rank["Venue"] == venue].index[0] + 1
        total_venues = len(venue_rank)
        st.write(f"Venue ranks {venue_position} out of {total_venues} in high-scoring tendency.")

        # ===================================================
        # SCORING SHAPE
        # ===================================================

        st.markdown("## 📈 Scoring Shape")

        dominance_margin = max(probs) - min(probs)
        middle_strength = probs[1]

        # Regime strength classification
        if max(probs) > 0.55:
            shape = "Strong Dominant Regime"
        elif max(probs) > 0.45:
            shape = "Clear Leading Regime"
        elif dominance_margin < 0.10:
            shape = "Balanced Distribution"
        elif probs[0] > 0.35 and probs[2] > 0.30:
            shape = "Polarized (Collapse + Big Total Both Live)"
        else:
            shape = "Moderately Skewed"

        # Middle band stability
        if middle_strength > 0.35:
            middle_label = "Stable Scoring Environment"
        elif middle_strength < 0.25:
            middle_label = "Extreme Outcome Environment"
        else:
            middle_label = "Transitional Scoring Pattern"

        st.write(f"Scoring Pattern Type: {shape}")
        st.write(f"Dominance Margin: {round(dominance_margin,3)}")
        st.write(f"Middle Band Strength: {round(middle_strength,3)}")
        st.write(f"Middle Compression Classification: {middle_label}")

        # ===================================================
        # MATCH INTERPRETATION
        # ===================================================

        st.markdown("## 🧠 Match Interpretation")

        if probs[0] > 0.55:
            interpretation = "Strong structural tilt toward below-par total."
        elif probs[2] > 0.55:
            interpretation = "Strong structural tilt toward 185+ outcome."
        elif probs[0] > 0.45:
            interpretation = "Clear lean toward below-par outcome."
        elif probs[2] > 0.45:
            interpretation = "Clear lean toward high-total outcome."
        elif middle_strength > 0.35:
            interpretation = "Match structurally projects into competitive 151–185 range."
        else:
            interpretation = "Match projects near league equilibrium."

        st.write(interpretation)

        # ===================================================
        # FINAL PROJECTION
        # ===================================================

        st.markdown("## ⚖ Final Projection")

        if probs[0] > 0.55:
            st.write("Primary Risk: Below-par Total.")
        elif probs[2] > 0.55:
            st.write("Primary Risk: 185+ Upside Total.")
        elif middle_strength > 0.35:
            st.write("Primary Structure: Competitive Total Most Stable.")
        else:
            st.write("Projection indicates moderate structural skew without extreme risk.")