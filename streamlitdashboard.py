# ---- Streamlit Imports ----
import streamlit as st

# ---- Core Python and Data Libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import rgb_to_hsv

# ---- Streamlit App Configuration ----
st.set_page_config(page_title="Court Usage Dashboard", layout="wide")

# ---- Data Loading with Caching ----
@st.cache_data
def load_data():
    # Blank templates
    blank_15_day_df = pd.read_parquet("Assets/blank_15_day_df.parquet")
    blank_15_month_df = pd.read_parquet("Assets/blank_15_month_df.parquet")
    blank_52_day_df = pd.read_parquet("Assets/blank_52_day_df.parquet")
    blank_52_month_df = pd.read_parquet("Assets/blank_52_month_df.parquet")
    blank_60_day_df = pd.read_parquet("Assets/blank_60_day_df.parquet")
    blank_60_month_df = pd.read_parquet("Assets/blank_60_month_df.parquet")

    # Colormap max values
    colormap_max_df = pd.read_parquet("Assets/colormap_max_df.parquet")

    # Court-specific dataframes
    Real_Tennis_Court_df = pd.read_parquet("Assets/Real_Tennis_Court_df.parquet")
    Lawn_Tennis_Court_1_df = pd.read_parquet("Assets/Lawn_Tennis_Court_1_df.parquet")
    Lawn_Tennis_Court_2_df = pd.read_parquet("Assets/Lawn_Tennis_Court_2_df.parquet")
    Rackets_and_Padel_Court_df = pd.read_parquet("Assets/Rackets_and_Padel_Court_df.parquet")
    Sports_Hall_df = pd.read_parquet("Assets/Sports_Hall_df.parquet")
    Squash_C_Court_df = pd.read_parquet("Assets/Squash_C_Court_df.parquet")
    Squash_D_Court_df = pd.read_parquet("Assets/Squash_D_Court_df.parquet")
    Squash_E_Glass_Court_df = pd.read_parquet("Assets/Squash_E_Glass_Court_df.parquet")
    Squash_F_Court_df = pd.read_parquet("Assets/Squash_F_Court_df.parquet")

    return {
        "blank_15_day_df": blank_15_day_df,
        "blank_15_month_df": blank_15_month_df,
        "blank_52_day_df": blank_52_day_df,
        "blank_52_month_df": blank_52_month_df,
        "blank_60_day_df": blank_60_day_df,
        "blank_60_month_df": blank_60_month_df,
        "colormap_max_df": colormap_max_df,
        "Real_Tennis_Court_df": Real_Tennis_Court_df,
        "Lawn_Tennis_Court_1_df": Lawn_Tennis_Court_1_df,
        "Lawn_Tennis_Court_2_df": Lawn_Tennis_Court_2_df,
        "Rackets_and_Padel_Court_df": Rackets_and_Padel_Court_df,
        "Sports_Hall_df": Sports_Hall_df,
        "Squash_C_Court_df": Squash_C_Court_df,
        "Squash_D_Court_df": Squash_D_Court_df,
        "Squash_E_Glass_Court_df": Squash_E_Glass_Court_df,
        "Squash_F_Court_df": Squash_F_Court_df,
    }

# ---- Load all data into memory ----
data = load_data()

# ---- UI Controls (Streamlit widgets) ----
st.sidebar.header("Controls")

cell_totals = st.sidebar.selectbox("Cell Totals", ["On", "Off"], index=1)
graph_height = st.sidebar.slider("Graph Height", min_value=8, max_value=20, value=10)
graph_width = st.sidebar.slider("Graph Width", min_value=8, max_value=20, value=10)

selected_year = st.sidebar.selectbox("Year", ["All", "2022", "2023", "2024", "2025"], index=4)
selected_groupby = st.sidebar.selectbox("Group By", ["Month", "Day"], index=1)

selected_court = st.sidebar.selectbox(
    "Court",
    [
        "Real Tennis Court", 
        "Lawn Tennis Court 1", 
        "Lawn Tennis Court 2",
        "Rackets & Padel Court", 
        "Sports Hall",
        "Squash C Court", 
        "Squash D Court", 
        "Squash E (Glass) Court",
        "Squash F Court"
    ],
    index=1
)

# ---- Plotting Function ----
def plot_court_heatmap(selected_year, selected_groupby, selected_court, graph_height, graph_width, cell_totals):
    selected_year_str = str(selected_year)

    df_name = selected_court.replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and") + "_df"
    df = data[df_name]

    # Filter year
    if selected_year_str != "All":
        df = df[df["Year"].astype(str) == selected_year_str]

    # Pivot table
    heatmap_data = df.groupby(["Time Slot", selected_groupby], observed=True).size().unstack(fill_value=0)

    # Determine blank structure
    if selected_court == "Real Tennis Court":
        blank_key = "blank_60"
    elif selected_court in ["Lawn Tennis Court 1", "Lawn Tennis Court 2"]:
        blank_key = "blank_52"
    else:
        blank_key = "blank_60"

    blank_name = f"{blank_key}_{selected_groupby.lower()}_df"
    blank_df = data[blank_name]
    heatmap_data = blank_df.add(heatmap_data, fill_value=0).fillna(0).astype(int)

    # Time Labels
    n_slots = heatmap_data.shape[0]
    if selected_court == "Real Tennis Court":
        time_labels = [f"{8 + i:02d}:00" for i in range(n_slots)]
    else:
        start_hour = 9 if selected_court in ["Lawn Tennis Court 1", "Lawn Tennis Court 2"] else 8
        slot_minutes = [start_hour * 60 + 15 * i for i in range(n_slots)]
        time_labels = [f"{m // 60:02d}:{m % 60:02d}" for m in slot_minutes]
    heatmap_data.index = time_labels

    # Reorder columns
    if selected_groupby == "Day":
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        heatmap_data = heatmap_data.reindex(columns=day_order)
    elif selected_groupby == "Month":
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        existing_months = [m for m in month_order if m in heatmap_data.columns]
        heatmap_data = heatmap_data.reindex(columns=existing_months)

    # Color scale
    colormap_max_df = data["colormap_max_df"]
    year_key = selected_year_str if selected_year_str in colormap_max_df["Year"].astype(str).values else "All"
    vmax_row = colormap_max_df[colormap_max_df["Year"].astype(str) == year_key].iloc[0]
    vmax_col = df_name.replace("_df", "") + f"_{selected_groupby}"
    vmax_value = vmax_row[vmax_col]



    # ---- Plot ----
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(graph_width, graph_height))
    cmap = plt.cm.gist_heat
    sns.set(font_scale=1.0)

    heatmap = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        annot=False,
        fmt="d",
        linewidths=0.5,
        linecolor="#444444",
        cbar_kws={"label": "Booking Count"},
        ax=ax,
        vmin=0,
        vmax=vmax_value
    )

    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.label.set_color("white")
    colorbar.ax.tick_params(colors="white")

    if cell_totals == "On":
        vmin, vmax = heatmap.get_children()[0].get_clim()
        for y in range(heatmap_data.shape[0]):
            for x in range(heatmap_data.shape[1]):
                val = heatmap_data.iloc[y, x]
                normed = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
                rgb = cmap(normed)[:3]
                brightness = rgb_to_hsv([[rgb]])[0][0][2]
                text_color = 'black' if brightness > 0.8 else 'white'
                ax.text(x + 0.5, y + 0.5, str(val), ha='center', va='center', color=text_color, fontsize=10)

    ax.set_title(
        f"{selected_court} Usage by {selected_groupby} – {'All Years' if selected_year_str == 'All' else selected_year_str}",
        fontsize=14, color="white"
    )
    ax.set_xlabel(selected_groupby, color="white")
    ax.set_ylabel("Start Time", color="white")
    ax.set_xticklabels(ax.get_xticklabels(), color="white")
    ax.set_yticklabels(ax.get_yticklabels(), color="white", rotation=0)
    plt.tight_layout()

    # ---- Render to Streamlit ----
    st.pyplot(fig)

# ---- Show plot ----
plot_court_heatmap(selected_year, selected_groupby, selected_court, graph_height, graph_width, cell_totals)
