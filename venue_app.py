import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ipl_venue_clusters_freq.csv")

df = load_data()

st.title("IPL Venue Difficulty & Thriller Analysis")
st.write(
    "Explore how IPL venues behave in terms of dominance, thrillers, and chasing vs batting-first advantage."
)

# -------------------------
# Sidebar: stadium selector
# -------------------------
stadiums = df["stadium"].sort_values().unique()
selected_stadium = st.sidebar.selectbox("Select a stadium", stadiums)

# -------------------------
# Main: selected venue details
# -------------------------
venue_row = df[df["stadium"] == selected_stadium].iloc[0]

st.subheader(f"🏟 {selected_stadium}")
st.write(f"**Cluster type:** {venue_row['cluster_label']}")

col1, col2, col3 = st.columns(3)
col1.metric("Matches analysed", int(venue_row["match_count"]))
col2.metric("Thriller %", f"{venue_row['thriller_pct']*100:.1f}%")
col3.metric("Blowout %", f"{venue_row['blowout_pct']*100:.1f}%")

st.markdown("---")

col4, col5 = st.columns(2)
col4.metric("Avg dominance score", f"{venue_row['avg_dominance']:.2f}")
col5.metric("Dominance variance", f"{venue_row['dominance_variance']:.2f}")

st.markdown("### Batting First vs Chasing")

col6, col7, col8 = st.columns(3)
col6.metric("Bat-first win %", f"{venue_row['bat_first_win_pct']*100:.1f}%")
col7.metric("Chasing win %", f"{venue_row['chasing_win_pct']*100:.1f}%")
chasing_adv = venue_row["chasing_win_pct"] - venue_row["bat_first_win_pct"]
col8.metric("Chasing advantage", f"{chasing_adv*100:.1f}%")

# Short textual explanation
if "Chaotic" in venue_row["cluster_label"]:
    st.info(
        "This venue tends to produce more close games and higher variability in results — "
        "a thriller-friendly, chaotic ground."
    )
else:
    st.info(
        "This venue tends to produce more one-sided results and clearer dominance patterns — "
        "a more predictable ground."
    )

# -------------------------
# Cluster scatter plot
# -------------------------

st.markdown("---")
st.subheader("Venue clusters overview")

st.write(
    "Each point below is a stadium. The x-axis shows how one-sided results are "
    "(average dominance), and the y-axis shows how often thrillers happen."
)

cluster_colors = {
    "Predictable / One-Sided Venue": "tab:orange",
    "Chaotic / Thriller Venue": "tab:blue",
}

df_scatter = df.copy()
df_scatter["color"] = df_scatter["cluster_label"].map(cluster_colors)

fig, ax = plt.subplots(figsize=(8, 5))

for label, grp in df_scatter.groupby("cluster_label"):
    ax.scatter(
        grp["avg_dominance"],
        grp["thriller_pct"] * 100,
        label=label,
    )

ax.axvline(0, linestyle="--", alpha=0.3)
ax.set_xlabel("Average dominance score (higher = more one-sided)")
ax.set_ylabel("Thriller %")
ax.set_title("IPL venues clustered by dominance vs thriller tendency")
ax.legend()
ax.grid(True, alpha=0.2)

st.pyplot(fig)

# -------------------------
# Venue Rankings (full lists with dropdown)
# -------------------------

st.markdown("---")
st.subheader("Venue Rankings")

ranking_option = st.selectbox(
    "Choose ranking to view:",
    [
        "🔥 Thriller percentage",
        "🎲 Dominance variance",
        "🏏 Bat-first win %",
        "🏃 Chasing win %",
    ],
)

if "Thriller" in ranking_option:
    rank_df = df.sort_values("thriller_pct", ascending=False)[
        ["stadium", "cluster_label", "thriller_pct"]
    ]
    rank_df = rank_df.rename(
        columns={
            "stadium": "Stadium",
            "cluster_label": "Cluster",
            "thriller_pct": "Thriller %",
        }
    )
elif "Dominance" in ranking_option:
    rank_df = df.sort_values("dominance_variance", ascending=False)[
        ["stadium", "cluster_label", "dominance_variance"]
    ]
    rank_df = rank_df.rename(
        columns={
            "stadium": "Stadium",
            "cluster_label": "Cluster",
            "dominance_variance": "Dominance variance",
        }
    )
elif "Bat-first" in ranking_option:
    rank_df = df.sort_values("bat_first_win_pct", ascending=False)[
        ["stadium", "cluster_label", "bat_first_win_pct"]
    ]
    rank_df = rank_df.rename(
        columns={
            "stadium": "Stadium",
            "cluster_label": "Cluster",
            "bat_first_win_pct": "Bat-first win %",
        }
    )
else:  # Chasing win %
    rank_df = df.sort_values("chasing_win_pct", ascending=False)[
        ["stadium", "cluster_label", "chasing_win_pct"]
    ]
    rank_df = rank_df.rename(
        columns={
            "stadium": "Stadium",
            "cluster_label": "Cluster",
            "chasing_win_pct": "Chasing win %",
        }
    )

st.dataframe(rank_df)
