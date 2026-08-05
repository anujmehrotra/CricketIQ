# ==========================================================
# CricketIQ — IPL Analytics and Match Prediction Platform
# ==========================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu


# ==========================================================
# 1. Page Configuration
# ==========================================================

st.set_page_config(
    page_title="CricketIQ",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"


# ==========================================================
# 3. Data Loading
# ==========================================================

@st.cache_data
def load_data():
    matches = pd.read_csv(
        DATA_DIR / "matches_cleaned.csv"
    )

    deliveries = pd.read_csv(
        DATA_DIR / "deliveries_cleaned.csv"
    )

    # Standardise season values
    season_mapping = {
        "2007/08": 2008,
        "2009/10": 2010,
        "2020/21": 2020
    }

    matches["season"] = (
        matches["season"]
        .astype(str)
        .str.strip()
        .replace(season_mapping)
        .astype(int)
    )

    # Standardise historical franchise names
    team_name_mapping = {
        "Royal Challengers Bangalore":
            "Royal Challengers Bengaluru",

        "Rising Pune Supergiant":
            "Rising Pune Supergiants",

        "Delhi Daredevils":
            "Delhi Capitals",

        "Kings XI Punjab":
            "Punjab Kings"
    }

    match_team_columns = [
        "team1",
        "team2",
        "toss_winner",
        "winner"
    ]

    for column in match_team_columns:
        if column in matches.columns:
            matches[column] = matches[column].replace(
                team_name_mapping
            )

    delivery_team_columns = [
        "batting_team",
        "bowling_team"
    ]

    for column in delivery_team_columns:
        if column in deliveries.columns:
            deliveries[column] = deliveries[column].replace(
                team_name_mapping
            )

    venue_mapping = {
    "M Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M. Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M. Chinnaswamy Stadium",

    "Punjab Cricket Association IS Bindra Stadium, Mohali":
        "Punjab Cricket Association IS Bindra Stadium",

    "Punjab Cricket Association Stadium, Mohali":
        "Punjab Cricket Association IS Bindra Stadium",

    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh":
        "Punjab Cricket Association IS Bindra Stadium",

    "MA Chidambaram Stadium, Chepauk":
        "MA Chidambaram Stadium",

    "MA Chidambaram Stadium, Chepauk, Chennai":
        "MA Chidambaram Stadium",

    "Wankhede Stadium, Mumbai":
        "Wankhede Stadium",

    "Arun Jaitley Stadium, Delhi":
        "Arun Jaitley Stadium",

    "Feroz Shah Kotla":
        "Arun Jaitley Stadium",

    "Arun Jaitely Stadium":
        "Arun Jaitley Stadium",

    "Rajiv Gandhi International Stadium, Uppal":
        "Rajiv Gandhi International Stadium",

    "Rajiv Gandhi International Stadium, Uppal, Hyderabad":
        "Rajiv Gandhi International Stadium",

    "Eden Gardens, Kolkata":
        "Eden Gardens",

    "Brabourne Stadium, Mumbai":
        "Brabourne Stadium",

    "Dr DY Patil Sports Academy, Mumbai":
        "Dr DY Patil Sports Academy",

    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam":
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",

    "Himachal Pradesh Cricket Association Stadium, Dharamsala":
        "Himachal Pradesh Cricket Association Stadium",

    "Maharashtra Cricket Association Stadium, Pune":
        "Maharashtra Cricket Association Stadium",

    "Subrata Roy Sahara Stadium":
        "Maharashtra Cricket Association Stadium",

    "Sawai Mansingh Stadium, Jaipur":
        "Sawai Mansingh Stadium",

    "Zayed Cricket Stadium, Abu Dhabi":
       "Sheikh Zayed Stadium"
}

    matches["venue"] = (
        matches["venue"]
        .astype(str)
        .str.strip()
        .replace(venue_mapping)
    )

    return matches, deliveries


matches, deliveries = load_data()


# ==========================================================
# 4. Model Loading
# ==========================================================

@st.cache_resource
def load_model_files():
    model = joblib.load(
        MODEL_DIR / "ipl_gradient_boosting_model.pkl"
    )

    metadata = joblib.load(
        MODEL_DIR / "ipl_model_metadata.pkl"
    )

    historical_state = joblib.load(
        MODEL_DIR / "ipl_historical_state.pkl"
    )

    return model, metadata, historical_state


try:
    model, model_metadata, historical_state = (
        load_model_files()
    )

    model_available = True
    model_error = None

except Exception as error:
    model = None
    model_metadata = {}
    historical_state = {}
    model_available = False
    model_error = str(error)


# ==========================================================
# 5. Helper Functions
# ==========================================================

def safe_divide(numerator, denominator, default=0):
    if denominator == 0:
        return default

    return numerator / denominator


def get_batting_stats(deliveries_data):
    valid_balls = deliveries_data[
        deliveries_data["extras_type"].ne("wides")
        | deliveries_data["extras_type"].isna()
    ]

    batting_stats = (
        deliveries_data.groupby("batter")
        .agg(
            Runs=("batsman_runs", "sum"),
            Fours=(
                "batsman_runs",
                lambda values: (values == 4).sum()
            ),
            Sixes=(
                "batsman_runs",
                lambda values: (values == 6).sum()
            )
        )
        .reset_index()
    )

    balls_faced = (
        valid_balls.groupby("batter")
        .size()
        .reset_index(name="Balls Faced")
    )

    dismissal_types = [
        "bowled",
        "caught",
        "caught and bowled",
        "lbw",
        "stumped",
        "hit wicket"
    ]

    dismissals = (
        deliveries_data[
            deliveries_data["dismissal_kind"].isin(
                dismissal_types
            )
        ]
        .groupby("player_dismissed")
        .size()
        .reset_index(name="Dismissals")
        .rename(
            columns={"player_dismissed": "batter"}
        )
    )

    batting_stats = (
        batting_stats
        .merge(
            balls_faced,
            on="batter",
            how="left"
        )
        .merge(
            dismissals,
            on="batter",
            how="left"
        )
    )

    batting_stats["Balls Faced"] = (
        batting_stats["Balls Faced"].fillna(0)
    )

    batting_stats["Dismissals"] = (
        batting_stats["Dismissals"].fillna(0)
    )

    batting_stats["Strike Rate"] = (
        batting_stats["Runs"]
        / batting_stats["Balls Faced"].replace(0, np.nan)
        * 100
    ).round(2)

    batting_stats["Batting Average"] = (
        batting_stats["Runs"]
        / batting_stats["Dismissals"].replace(0, np.nan)
    ).round(2)

    return batting_stats


def get_bowling_stats(deliveries_data):
    wicket_types = [
        "bowled",
        "caught",
        "caught and bowled",
        "lbw",
        "stumped",
        "hit wicket"
    ]

    wickets = (
        deliveries_data[
            deliveries_data["dismissal_kind"].isin(
                wicket_types
            )
        ]
        .groupby("bowler")
        .size()
        .reset_index(name="Wickets")
    )

    legal_deliveries = deliveries_data[
        ~deliveries_data["extras_type"].isin(
            ["wides", "noballs"]
        )
        | deliveries_data["extras_type"].isna()
    ]

    balls_bowled = (
        legal_deliveries.groupby("bowler")
        .size()
        .reset_index(name="Balls Bowled")
    )

    bowler_runs = np.where(
        deliveries_data["extras_type"].isin(
            ["byes", "legbyes"]
        ),
        deliveries_data["batsman_runs"],
        deliveries_data["total_runs"]
    )

    runs_conceded = (
        deliveries_data.assign(
            Bowler_Runs=bowler_runs
        )
        .groupby("bowler")["Bowler_Runs"]
        .sum()
        .reset_index(name="Runs Conceded")
    )

    dot_balls = (
        deliveries_data[
            deliveries_data["total_runs"] == 0
        ]
        .groupby("bowler")
        .size()
        .reset_index(name="Dot Balls")
    )

    bowling_stats = (
        balls_bowled
        .merge(
            wickets,
            on="bowler",
            how="left"
        )
        .merge(
            runs_conceded,
            on="bowler",
            how="left"
        )
        .merge(
            dot_balls,
            on="bowler",
            how="left"
        )
        .fillna(0)
    )

    bowling_stats["Overs"] = (
        bowling_stats["Balls Bowled"] / 6
    ).round(1)

    bowling_stats["Economy"] = (
        bowling_stats["Runs Conceded"]
        / (bowling_stats["Balls Bowled"] / 6)
    ).round(2)

    bowling_stats["Bowling Average"] = (
        bowling_stats["Runs Conceded"]
        / bowling_stats["Wickets"].replace(0, np.nan)
    ).round(2)

    bowling_stats["Bowling Strike Rate"] = (
        bowling_stats["Balls Bowled"]
        / bowling_stats["Wickets"].replace(0, np.nan)
    ).round(2)

    return bowling_stats


def get_state_value(dictionary, key, default=0):
    return dictionary.get(key, default)


def predict_match_winner(
    team1,
    team2,
    city,
    venue,
    toss_winner,
    toss_decision
):
    if team1 == team2:
        raise ValueError(
            "Team 1 and Team 2 must be different."
        )

    elo_ratings = historical_state["elo_rating"]
    recent_results = historical_state["recent_results"]
    h2h_matches = historical_state["h2h_matches"]
    h2h_wins = historical_state["h2h_wins"]
    venue_matches = historical_state["venue_matches"]
    venue_wins = historical_state["venue_wins"]
    team_matches_state = historical_state[
        "team_matches"
    ]

    # Elo ratings
    team1_elo = get_state_value(
        elo_ratings,
        team1,
        1500
    )

    team2_elo = get_state_value(
        elo_ratings,
        team2,
        1500
    )

    # Recent form
    team1_recent_history = get_state_value(
        recent_results,
        team1,
        []
    )

    team2_recent_history = get_state_value(
        recent_results,
        team2,
        []
    )

    team1_recent_form = (
        float(np.mean(team1_recent_history))
        if len(team1_recent_history) > 0
        else 0.5
    )

    team2_recent_form = (
        float(np.mean(team2_recent_history))
        if len(team2_recent_history) > 0
        else 0.5
    )

    # Head-to-head
    h2h_key = tuple(sorted([team1, team2]))

    total_h2h_matches = get_state_value(
        h2h_matches,
        h2h_key,
        0
    )

    h2h_record = get_state_value(
        h2h_wins,
        h2h_key,
        {}
    )

    if total_h2h_matches > 0:
        team1_h2h_rate = (
            h2h_record.get(team1, 0)
            / total_h2h_matches
        )

        team2_h2h_rate = (
            h2h_record.get(team2, 0)
            / total_h2h_matches
        )

    else:
        team1_h2h_rate = 0.5
        team2_h2h_rate = 0.5

    # Venue performance
    team1_venue_key = (team1, venue)
    team2_venue_key = (team2, venue)

    team1_venue_matches = get_state_value(
        venue_matches,
        team1_venue_key,
        0
    )

    team2_venue_matches = get_state_value(
        venue_matches,
        team2_venue_key,
        0
    )

    team1_venue_rate = safe_divide(
        get_state_value(
            venue_wins,
            team1_venue_key,
            0
        ),
        team1_venue_matches,
        default=0.5
    )

    team2_venue_rate = safe_divide(
        get_state_value(
            venue_wins,
            team2_venue_key,
            0
        ),
        team2_venue_matches,
        default=0.5
    )

    team1_matches_played = get_state_value(
        team_matches_state,
        team1,
        0
    )

    team2_matches_played = get_state_value(
        team_matches_state,
        team2,
        0
    )

    input_data = pd.DataFrame([{
        "city": city,
        "venue": venue,
        "team1": team1,
        "team2": team2,
        "toss_decision": toss_decision,
        "team1_won_toss": int(
            toss_winner == team1
        ),
        "elo_difference":
            team1_elo - team2_elo,
        "recent_form_diff":
            team1_recent_form - team2_recent_form,
        "h2h_win_rate_diff":
            team1_h2h_rate - team2_h2h_rate,
        "venue_win_rate_diff":
            team1_venue_rate - team2_venue_rate,
        "season_win_rate_diff": 0.0,
        "matches_played_diff":
            team1_matches_played
            - team2_matches_played
    }])

    team1_probability = float(
        model.predict_proba(input_data)[0][1]
    )

    team2_probability = 1 - team1_probability

    predicted_winner = (
        team1
        if team1_probability >= 0.5
        else team2
    )

    return {
        "predicted_winner": predicted_winner,
        "team1_probability": team1_probability,
        "team2_probability": team2_probability,
        "confidence": max(
            team1_probability,
            team2_probability
        ),
        "team1_elo": team1_elo,
        "team2_elo": team2_elo,
        "team1_recent_form": team1_recent_form,
        "team2_recent_form": team2_recent_form,
        "team1_h2h_rate": team1_h2h_rate,
        "team2_h2h_rate": team2_h2h_rate,
        "team1_venue_rate": team1_venue_rate,
        "team2_venue_rate": team2_venue_rate
    }


# ==========================================================
# 6. Prepare Reusable Analytics Tables
# ==========================================================

batting_stats = get_batting_stats(deliveries)
bowling_stats = get_bowling_stats(deliveries)


# ==========================================================
# 7. Sidebar Navigation
# ==========================================================

with st.sidebar:
    st.title("🏏 CricketIQ")

    st.caption(
        "IPL Analytics & Prediction"
    )

    selected = option_menu(
        menu_title=None,
        options=[
            "Home",
            "Team Analytics",
            "Batting Analytics",
            "Bowling Analytics",
            "Match Predictor",
            "About"
        ],
        icons=[
            "house",
            "bar-chart",
            "activity",
            "bullseye",
            "cpu",
            "info-circle"
        ],
        default_index=0
    )

    st.divider()

    st.caption(
        "Historical IPL data: 2008–2024"
    )


# ==========================================================
# 8. Home Page
# ==========================================================

if selected == "Home":
    st.title("🏏 CricketIQ")

    st.subheader(
        "IPL Analytics and Match Prediction Platform"
    )

    st.write(
        """
        CricketIQ combines exploratory data analysis, SQL,
        interactive visualisation and machine learning to
        analyse IPL performance from 2008 to 2024.
        """
    )

    total_teams = len(
        set(matches["team1"].dropna()).union(
            set(matches["team2"].dropna())
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Seasons Covered",
        matches["season"].nunique()
    )

    col2.metric(
        "Matches Analysed",
        f"{len(matches):,}"
    )

    col3.metric(
        "Teams",
        total_teams
    )

    col4.metric(
        "Final ML Model",
        model_metadata.get(
            "model_name",
            "Gradient Boosting"
        )
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            ### 📊 Analytics Features

            - Team performance by season
            - Venue-based performance
            - Batting and bowling leaderboards
            - Strike rate, average and economy analysis
            - Interactive player selection
            """
        )

    with right:
        st.markdown(
            """
            ### 🤖 Prediction Features

            - Elo rating difference
            - Recent five-match form
            - Head-to-head performance
            - Venue record
            - Toss winner and toss decision
            """
        )

    st.info(
        "Use the sidebar to navigate through the "
        "analytics platform."
    )


# ==========================================================
# 9. Team Analytics Page
# ==========================================================

elif selected == "Team Analytics":
    st.title("📊 Team Analytics")

    teams = sorted(
        set(matches["team1"].dropna()).union(
            set(matches["team2"].dropna())
        )
    )

    selected_team = st.selectbox(
        "Select a team",
        teams
    )

    team_matches = matches[
        (matches["team1"] == selected_team)
        | (matches["team2"] == selected_team)
    ].copy()

    matches_played = len(team_matches)

    matches_won = int(
        (
            team_matches["winner"]
            == selected_team
        ).sum()
    )

    win_percentage = safe_divide(
        matches_won * 100,
        matches_played
    )

    toss_wins = int(
        (
            team_matches["toss_winner"]
            == selected_team
        ).sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Matches Played",
        matches_played
    )

    col2.metric(
        "Matches Won",
        matches_won
    )

    col3.metric(
        "Win Percentage",
        f"{win_percentage:.2f}%"
    )

    col4.metric(
        "Toss Wins",
        toss_wins
    )

    st.divider()

    season_performance = (
        team_matches.groupby("season")
        .agg(
            Matches=("id", "count"),
            Wins=(
                "winner",
                lambda values:
                (values == selected_team).sum()
            )
        )
        .reset_index()
        .sort_values("season")
    )

    season_performance["Win Percentage"] = (
        season_performance["Wins"]
        / season_performance["Matches"]
        * 100
    ).round(2)

    season_chart = px.line(
        season_performance,
        x="season",
        y="Win Percentage",
        markers=True,
        title=(
            f"{selected_team} "
            "Win Percentage by Season"
        )
    )

    season_chart.update_xaxes(
        tickmode="linear",
        dtick=1,
        title="Season"
    )

    season_chart.update_yaxes(
        title="Win Percentage (%)"
    )

    st.plotly_chart(
        season_chart,
        use_container_width=True
    )

    venue_performance = (
        team_matches.groupby("venue")
        .agg(
            Matches=("id", "count"),
            Wins=(
                "winner",
                lambda values:
                (values == selected_team).sum()
            )
        )
        .reset_index()
    )

    venue_performance["Win Percentage"] = (
        venue_performance["Wins"]
        / venue_performance["Matches"]
        * 100
    ).round(2)

    venue_performance = (
        venue_performance[
            venue_performance["Matches"] >= 3
        ]
        .sort_values(
            "Win Percentage",
            ascending=False
        )
        .head(10)
        .sort_values("Win Percentage")
    )

    venue_chart = px.bar(
        venue_performance,
        x="Win Percentage",
        y="venue",
        orientation="h",
        text="Win Percentage",
        title=f"Best Venues for {selected_team}"
    )

    venue_chart.update_yaxes(
        title="Venue"
    )

    venue_chart.update_xaxes(
        title="Win Percentage (%)"
    )

    st.plotly_chart(
        venue_chart,
        use_container_width=True
    )

    st.subheader("Season-wise Performance")

    st.dataframe(
        season_performance,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# 10. Batting Analytics Page
# ==========================================================

elif selected == "Batting Analytics":
    st.title("🏏 Batting Analytics")

    selected_batter = st.selectbox(
        "Select a batter",
        sorted(
            batting_stats["batter"]
            .dropna()
            .unique()
        )
    )

    batter_record = batting_stats[
        batting_stats["batter"]
        == selected_batter
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Runs",
        int(batter_record["Runs"])
    )

    average_value = (
        batter_record["Batting Average"]
    )

    average_display = (
        f"{average_value:.2f}"
        if pd.notna(average_value)
        else "N/A"
    )

    col2.metric(
        "Batting Average",
        average_display
    )

    col3.metric(
        "Strike Rate",
        f"{batter_record['Strike Rate']:.2f}"
    )

    col4.metric(
        "Boundaries",
        int(
            batter_record["Fours"]
            + batter_record["Sixes"]
        )
    )

    st.divider()

    top_scorers = (
        batting_stats
        .nlargest(10, "Runs")
        .sort_values("Runs")
    )

    runs_chart = px.bar(
        top_scorers,
        x="Runs",
        y="batter",
        orientation="h",
        text="Runs",
        title="Top 10 IPL Run Scorers"
    )

    st.plotly_chart(
        runs_chart,
        use_container_width=True
    )

    strike_rate_leaders = (
        batting_stats[
            batting_stats["Balls Faced"] >= 500
        ]
        .nlargest(10, "Strike Rate")
        .sort_values("Strike Rate")
    )

    strike_rate_chart = px.bar(
        strike_rate_leaders,
        x="Strike Rate",
        y="batter",
        orientation="h",
        text="Strike Rate",
        title=(
            "Highest Strike Rates "
            "— Minimum 500 Balls"
        )
    )

    st.plotly_chart(
        strike_rate_chart,
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:
        top_fours = (
            batting_stats
            .nlargest(10, "Fours")
            .sort_values("Fours")
        )

        fours_chart = px.bar(
            top_fours,
            x="Fours",
            y="batter",
            orientation="h",
            text="Fours",
            title="Most Fours"
        )

        st.plotly_chart(
            fours_chart,
            use_container_width=True
        )

    with right:
        top_sixes = (
            batting_stats
            .nlargest(10, "Sixes")
            .sort_values("Sixes")
        )

        sixes_chart = px.bar(
            top_sixes,
            x="Sixes",
            y="batter",
            orientation="h",
            text="Sixes",
            title="Most Sixes"
        )

        st.plotly_chart(
            sixes_chart,
            use_container_width=True
        )

    st.subheader("Batting Leaderboard")

    batting_leaderboard = (
        batting_stats[
            [
                "batter",
                "Runs",
                "Balls Faced",
                "Batting Average",
                "Strike Rate",
                "Fours",
                "Sixes"
            ]
        ]
        .sort_values(
            "Runs",
            ascending=False
        )
        .head(25)
    )

    st.dataframe(
        batting_leaderboard,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# 11. Bowling Analytics Page
# ==========================================================

elif selected == "Bowling Analytics":
    st.title("🎯 Bowling Analytics")

    selected_bowler = st.selectbox(
        "Select a bowler",
        sorted(
            bowling_stats["bowler"]
            .dropna()
            .unique()
        )
    )

    bowler_record = bowling_stats[
        bowling_stats["bowler"]
        == selected_bowler
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Wickets",
        int(bowler_record["Wickets"])
    )

    col2.metric(
        "Economy",
        f"{bowler_record['Economy']:.2f}"
    )

    bowling_average = (
        bowler_record["Bowling Average"]
    )

    average_display = (
        f"{bowling_average:.2f}"
        if pd.notna(bowling_average)
        else "N/A"
    )

    col3.metric(
        "Bowling Average",
        average_display
    )

    bowling_sr = (
        bowler_record["Bowling Strike Rate"]
    )

    sr_display = (
        f"{bowling_sr:.2f}"
        if pd.notna(bowling_sr)
        else "N/A"
    )

    col4.metric(
        "Strike Rate",
        sr_display
    )

    st.divider()

    top_wicket_takers = (
        bowling_stats
        .nlargest(10, "Wickets")
        .sort_values("Wickets")
    )

    wickets_chart = px.bar(
        top_wicket_takers,
        x="Wickets",
        y="bowler",
        orientation="h",
        text="Wickets",
        title="Top 10 IPL Wicket Takers"
    )

    st.plotly_chart(
        wickets_chart,
        use_container_width=True
    )

    best_economy = (
        bowling_stats[
            bowling_stats["Balls Bowled"] >= 600
        ]
        .nsmallest(10, "Economy")
        .sort_values(
            "Economy",
            ascending=False
        )
    )

    economy_chart = px.bar(
        best_economy,
        x="Economy",
        y="bowler",
        orientation="h",
        text="Economy",
        title=(
            "Best Economy Rates "
            "— Minimum 100 Overs"
        )
    )

    st.plotly_chart(
        economy_chart,
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:
        best_average = (
            bowling_stats[
                bowling_stats["Wickets"] >= 50
            ]
            .nsmallest(
                10,
                "Bowling Average"
            )
            .sort_values(
                "Bowling Average",
                ascending=False
            )
        )

        average_chart = px.bar(
            best_average,
            x="Bowling Average",
            y="bowler",
            orientation="h",
            text="Bowling Average",
            title=(
                "Best Bowling Average "
                "— Minimum 50 Wickets"
            )
        )

        st.plotly_chart(
            average_chart,
            use_container_width=True
        )

    with right:
        best_strike_rate = (
            bowling_stats[
                bowling_stats["Wickets"] >= 50
            ]
            .nsmallest(
                10,
                "Bowling Strike Rate"
            )
            .sort_values(
                "Bowling Strike Rate",
                ascending=False
            )
        )

        bowling_sr_chart = px.bar(
            best_strike_rate,
            x="Bowling Strike Rate",
            y="bowler",
            orientation="h",
            text="Bowling Strike Rate",
            title=(
                "Best Bowling Strike Rate "
                "— Minimum 50 Wickets"
            )
        )

        st.plotly_chart(
            bowling_sr_chart,
            use_container_width=True
        )

    top_dot_balls = (
        bowling_stats
        .nlargest(10, "Dot Balls")
        .sort_values("Dot Balls")
    )

    dot_ball_chart = px.bar(
        top_dot_balls,
        x="Dot Balls",
        y="bowler",
        orientation="h",
        text="Dot Balls",
        title="Top 10 Dot-Ball Bowlers"
    )

    st.plotly_chart(
        dot_ball_chart,
        use_container_width=True
    )

    st.subheader("Bowling Leaderboard")

    bowling_leaderboard = (
        bowling_stats[
            [
                "bowler",
                "Wickets",
                "Overs",
                "Runs Conceded",
                "Economy",
                "Bowling Average",
                "Bowling Strike Rate",
                "Dot Balls"
            ]
        ]
        .sort_values(
            "Wickets",
            ascending=False
        )
        .head(25)
    )

    st.dataframe(
        bowling_leaderboard,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# 12. Match Predictor Page
# ==========================================================

elif selected == "Match Predictor":
    st.title("🤖 IPL Match Predictor")

    if not model_available:
        st.error(
            "The saved model files could not be loaded."
        )

        st.code(model_error)

        st.stop()

    st.write(
        """
        Select the pre-match and toss details below.
        The Gradient Boosting model will estimate the
        winning probability of both teams.
        """
    )

    current_teams = [
        "Chennai Super Kings",
        "Delhi Capitals",
        "Gujarat Titans",
        "Kolkata Knight Riders",
        "Lucknow Super Giants",
        "Mumbai Indians",
        "Punjab Kings",
        "Rajasthan Royals",
        "Royal Challengers Bengaluru",
        "Sunrisers Hyderabad"
    ]

    venue_city_map = (
    matches[["venue", "city"]]
    .dropna()
    .drop_duplicates(subset="venue")
    .set_index("venue")["city"]
    .to_dict()
)

    venues = sorted(venue_city_map.keys())

    left, right = st.columns(2)

    with left:
        team1 = st.selectbox(
        "Team 1",
        current_teams,
        index=5
    )

    team2_options = [
        team
        for team in current_teams
        if team != team1
    ]

    team2 = st.selectbox(
        "Team 2",
        team2_options
    )

    venue = st.selectbox(
        "Venue",
        venues
    )

    with right:
        city = venue_city_map[venue]

    st.text_input(
        "City",
        value=city,
        disabled=True
    )

    toss_winner = st.selectbox(
        "Toss Winner",
        [team1, team2]
    )

    toss_decision_label = st.selectbox(
        "Toss Decision",
        ["Bat", "Field"]
    )

    toss_decision = (
        toss_decision_label.lower()
    )

    st.divider()

    if st.button(
        "🏏 Predict Match Winner",
        type="primary",
        use_container_width=True
    ):
        try:
            result = predict_match_winner(
                team1=team1,
                team2=team2,
                city=city,
                venue=venue,
                toss_winner=toss_winner,
                toss_decision=toss_decision
            )

            st.success(
                "🏆 Predicted Winner: "
                f"{result['predicted_winner']}"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
              team1,
              f"{result['team1_probability'] * 100:.2f}%"
            )

            col2.metric(
              team2,
              f"{result['team2_probability'] * 100:.2f}%"
            )

            col3.metric(
            "Model Confidence",
            f"{result['confidence'] * 100:.2f}%"
            )

            probability_data = pd.DataFrame({
                "Team": [team1, team2],
                "Winning Probability": [
                    result["team1_probability"] * 100,
                    result["team2_probability"] * 100
                ]
            })

            probability_chart = px.bar(
                probability_data,
                x="Team",
                y="Winning Probability",
                text="Winning Probability",
                title="Predicted Winning Probabilities"
            )

            probability_chart.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            probability_chart.update_yaxes(
                range=[0, 100],
                title="Winning Probability (%)"
            )

            st.plotly_chart(
                probability_chart,
                use_container_width=True
            )

            st.subheader(
                "Historical Feature Comparison"
            )

            comparison_data = pd.DataFrame({
                "Metric": [
                    "Elo Rating",
                    "Recent Form (%)",
                    "Head-to-Head Win Rate (%)",
                    "Venue Win Rate (%)"
                ],
                team1: [
                    round(
                        result["team1_elo"],
                        2
                    ),
                    round(
                        result["team1_recent_form"]
                        * 100,
                        2
                    ),
                    round(
                        result["team1_h2h_rate"]
                        * 100,
                        2
                    ),
                    round(
                        result["team1_venue_rate"]
                        * 100,
                        2
                    )
                ],
                team2: [
                    round(
                        result["team2_elo"],
                        2
                    ),
                    round(
                        result["team2_recent_form"]
                        * 100,
                        2
                    ),
                    round(
                        result["team2_h2h_rate"]
                        * 100,
                        2
                    ),
                    round(
                        result["team2_venue_rate"]
                        * 100,
                        2
                    )
                ]
            })

            st.dataframe(
                comparison_data,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:
            st.error(
                "Prediction could not be generated."
            )

            st.exception(error)

    st.warning(
        """
        **Model limitation:** The model's evaluation
        performance was close to random prediction on the
        unseen 2024 season. Results should be interpreted
        as experimental probability estimates rather than
        guaranteed outcomes.
        """
    )


# ==========================================================
# 13. About Page
# ==========================================================

elif selected == "About":
    st.title("ℹ️ About CricketIQ")

    st.markdown(
        """
        CricketIQ is an end-to-end IPL analytics and machine
        learning portfolio project.

        ### Project components

        - Python data cleaning and exploratory analysis
        - DuckDB and SQL analytics
        - Batting, bowling and team performance analysis
        - Leakage-free historical feature engineering
        - Elo-based team-strength modelling
        - Gradient Boosting classification
        - Streamlit model deployment
        - Power BI dashboard development

        ### Technologies

        - Python
        - Pandas and NumPy
        - Plotly
        - SQL and DuckDB
        - Scikit-learn
        - Streamlit
        - Joblib
        """
    )

    if model_available:
        st.subheader("Model Performance")

        performance_data = {
            "Metric": [
                "Accuracy",
                "Balanced Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            "Score": [
                model_metadata.get(
                    "accuracy",
                    np.nan
                ),
                model_metadata.get(
                    "balanced_accuracy",
                    np.nan
                ),
                model_metadata.get(
                    "precision",
                    np.nan
                ),
                model_metadata.get(
                    "recall",
                    np.nan
                ),
                model_metadata.get(
                    "f1_score",
                    np.nan
                ),
                model_metadata.get(
                    "roc_auc",
                    np.nan
                )
            ]
        }

        st.dataframe(
            pd.DataFrame(performance_data),
            use_container_width=True,
            hide_index=True
        )

    st.info(
        """
        Match predictions are based on historical team-level
        features. Playing XI, injuries, weather, pitch
        conditions and individual player availability are not
        included.
        """
    )