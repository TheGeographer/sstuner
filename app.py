import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import date
from climbscore import (
    unified_base_score,
    logistic_adjustment,
    blended_wind_adjustment,
    seasonal_cloud_adjustment,
    precipitation_adjustment,
    calculate_apparent_temperature
)

# supabase credentials
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]

# --- Helpers ---
def c_to_f(c): return c * 9 / 5 + 32
def ms_to_mph(ms): return ms * 2.23694
def mmphr_to_inphr(mm): return mm / 25.4

def fetch_forecast_weather(activity, area_uuid):
    url = f"{SUPABASE_URL}/rest/v1/rpc/get_activity_scores"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "p_activity": activity,
        "p_area_uuid": area_uuid
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        df = pd.DataFrame(response.json())
        df["temp"] = c_to_f(df["temp"])
        df["windsp"] = ms_to_mph(df["windsp"])
        df["precip"] = mmphr_to_inphr(df["precip"].fillna(0))
        df["chance_pcp"] = df["chance_pcp"].fillna(0)
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        return df
    else:
        st.error(f"Failed to fetch data from Supabase: {response.text}")
        return pd.DataFrame()

# --- Streamlit App ---
PLOT_SIZE = (6, 3)
st.title("StokeScore Parameter Tuner")

activity = st.text_input("Activity (e.g., climbing)", "climbing")
area_uuid = st.text_input("Area UUID", "067f2a9f-b353-79e5-8000-99b0e375aad9")
latitude = st.number_input("Latitude for Climbing Area", value=38.0, format="%.2f")
timezone = st.text_input("Time Zone (e.g., America/Denver)", "America/Denver")

# Adjustment toggles
st.sidebar.header("Enable Adjustments")
use_wind = st.sidebar.checkbox("Use Wind Adjustment", value=True)
use_cloud = st.sidebar.checkbox("Use Cloud Adjustment", value=True)
use_precip = st.sidebar.checkbox("Use Precip Adjustment", value=True)
use_humidity = st.sidebar.checkbox("Use Humidity Adjustment", value=False)

if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None

if st.button("Fetch Weather"):
    st.session_state.forecast_df = fetch_forecast_weather(activity, area_uuid)

df = st.session_state.forecast_df
if df is not None and not df.empty:
    import pytz
    try:
        local_tz = pytz.timezone(timezone)
        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is None:
            df["datetime_local"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(local_tz)
        else:
            df["datetime_local"] = df["datetime"].dt.tz_convert(local_tz)
        midnights = df["datetime_local"].dt.tz_convert(None)
    except Exception as e:
        st.warning(f"Invalid time zone. Showing UTC. ({e})")
        df["datetime_local"] = pd.to_datetime(df["datetime"])
    st.write("### Forecast Weather Data (US Units)")
    if "local" not in df.columns:
        df.insert(1, "local", df["datetime_local"])
    st.dataframe(df[["datetime", "local", "temp", "rh", "windsp", "cloud", "chance_pcp", "precip"]])

    # Sliders to adjust parameters
    st.sidebar.header("Base Score Parameters")
    temp_opt = st.sidebar.slider("Temp Optimum", 40, 100, 68)
    width_lo = st.sidebar.slider("Width Low", 5, 30, 26)
    width_hi = st.sidebar.slider("Width High", 5, 30, 23)

    st.sidebar.header("Wind Adj Parameters")
    wind_tmin = 50
    wind_tmax = 70
    wind_scale = st.sidebar.slider("Cool Wind Penalty", 0, 50, 14)
    wind_decay = st.sidebar.slider("Cool Decay", 1, 20, 4) / 100
    warm_ideal = st.sidebar.slider("Warm Wind Ideal", 0, 20, 10)
    warm_width = st.sidebar.slider("Warm Wind Width", 1, 20, 6)
    warm_scale = st.sidebar.slider("Warm Wind Scale", 0, 50, 12)

    st.sidebar.header("Cloud Adj Parameters")
    cloud_min = st.sidebar.slider("Min Cloud Adj Scale", 0, 20, 7)
    cloud_max = st.sidebar.slider("Max Cloud Adj Scale", 10, 40, 14)
    cloud_k = st.sidebar.slider("Cloud Steepness Base", 1, 50, 8) / 100

    st.sidebar.header("Precipitation Adj")
    precip_p0 = st.sidebar.slider("Precip Midpoint", 20, 80, 64)
    precip_k = st.sidebar.slider("Precip Steepness", 1, 50, 15) / 100
    precip_thresh = st.sidebar.slider("Precip Threshold (in/hr)", 0.01, 0.2, 0.05)

    st.sidebar.header("Humidity Adj Parameters")
    rh_center = st.sidebar.slider("RH Center", 20, 80, 60)
    rh_k = st.sidebar.slider("RH Steepness", 1, 50, 10) / 100
    rh_scale = st.sidebar.slider("RH Scale", 5, 20, 10)

    df["apparent_temp"] = calculate_apparent_temperature(df["temp"], df["rh"], df["windsp"])
    df["base"] = unified_base_score(df["apparent_temp"], temp_opt, width_lo, width_hi)

    df["rh_adj"] = logistic_adjustment(df["rh"], rh_center, rh_k, rh_scale) if use_humidity else 0
    df["wind_adj"] = blended_wind_adjustment(df["windsp"], df["apparent_temp"], wind_tmin, wind_tmax, wind_scale, wind_decay, warm_ideal, warm_width, warm_scale) if use_wind else 0
    df["cloud_adj"] = seasonal_cloud_adjustment(df["cloud"], df["apparent_temp"], pd.to_datetime(df["date"].iloc[0]).date(), latitude, center_temp=65.0, min_scale=cloud_min, max_scale=cloud_max, steepness=cloud_k) if use_cloud else 0
    df["precip_adj"] = precipitation_adjustment(df["chance_pcp"].to_numpy(), df["precip"].to_numpy(), 100, precip_p0, precip_k, precip_thresh) if use_precip else 0

    df["score"] = (df["base"] + df["rh_adj"] + df["wind_adj"] + df["cloud_adj"] + df["precip_adj"]).clip(upper=100)



    # --- Visualize Base Score ---
    st.subheader("Base Score: Apparent Temperature")
    import plotly.express as px
    fig_temp = px.line(df, x="datetime_local", y="temp", title="Forecast Temperature (°F) [Ambient]", labels={"datetime_local": "Local Time", "temp": "Temp (°F)"})
    fig_temp.update_traces(mode="lines+markers")
    fig_temp.update_layout(height=300)
    for dt in midnights:
        if dt.hour == 0:
            fig_temp.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
    st.plotly_chart(fig_temp, use_container_width=True)

    fig_app_temp = px.line(df, x="datetime_local", y="apparent_temp", title="Forecast Temperature (°F) [Apparent]", labels={"datetime_local": "Local Time", "apparent_temp": "Apparent Temp (°F)"})
    fig_app_temp.update_traces(mode="lines+markers")
    fig_app_temp.update_layout(height=300)
    for dt in midnights:
        if dt.hour == 0:
            fig_app_temp.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
    st.plotly_chart(fig_app_temp, use_container_width=True)

    # Plot Gaussian curve for base score over a temp range
    temp_range = np.linspace(20, 100, 300)
    apparent_temp_range = calculate_apparent_temperature(temp_range, np.full_like(temp_range, 50), np.full_like(temp_range, 5))
    base_curve = unified_base_score(apparent_temp_range, temp_opt, width_lo, width_hi)

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(apparent_temp_range, base_curve, label="Base Score Curve", color="blue")
    ax1.scatter(df["apparent_temp"], df["base"], color="black", zorder=5, label="Forecast Points")
    ax1.set_xlabel("Apparent Temp (°F)")
    ax1.set_ylabel("Base Score")
    ax1.set_title("Base Score Curve (vs. Apparent Temperature)")
    ax1.grid(True)
    st.pyplot(fig1)

    # Show actual base score values over time
    import plotly.express as px
    fig_base = px.line(df, x="datetime_local", y="base", title="Base Score Over Time", labels={"datetime_local": "Local Time", "base": "Base Score"})
    fig_base.update_traces(mode="lines+markers")
    fig_base.update_layout(height=300)
    midnights = df["datetime_local"].dt.tz_convert(None)
    for dt in midnights:
        if dt.hour == 0:
            fig_base.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
    st.plotly_chart(fig_base, use_container_width=True)


    # --- Visualize Wind Adjustment ---
    if use_wind:
        st.subheader("Wind Adjustment")
        fig_wind = px.line(df, x="datetime_local", y="windsp", title="Wind Speed Forecast", labels={"datetime_local": "Local Time", "windsp": "Wind (mph)"})
        fig_wind.update_traces(mode="lines+markers")
        fig_wind.update_layout(height=300)
        for dt in midnights:
            if dt.hour == 0:
                fig_wind.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
        st.plotly_chart(fig_wind, use_container_width=True)

        wind_range = np.linspace(0, 30, 300)

        # --- Model Curves for Multiple Apparent Temps ---
        temps = list(range(50, 71, 5))  # 30, 40, ..., 90
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))  # Gradient from blue to red

        fig3, ax3 = plt.subplots(figsize=PLOT_SIZE)

        for i, t in enumerate(temps):
            curve = blended_wind_adjustment(
                wind_range, t,
                wind_tmin, wind_tmax,
                wind_scale, wind_decay,
                warm_ideal, warm_width, warm_scale
            )
            ax3.plot(wind_range, curve, color=colors[i], label=f"{t}°F")

        # Plot actual forecast points
        ax3.scatter(df["windsp"], df["wind_adj"], color="black", zorder=5, label="Forecast Points")

        ax3.set_xlabel("Wind Speed (mph)")
        ax3.set_ylabel("Wind Adjustment")
        ax3.set_title("Wind Adjustment Curves Across Apparent Temperatures")
        ax3.grid(True)
        ax3.legend(ncol=3, fontsize="small", loc="upper right")
        st.pyplot(fig3)

        fig_wind_adj = px.line(df, x="datetime_local", y="wind_adj", title="Wind Adjustment Over Time", labels={"datetime_local": "Local Time", "wind_adj": "Adjustment"})
        fig_wind_adj.update_traces(mode="lines+markers")
        fig_wind_adj.update_layout(height=300)
        for dt in midnights:
            if dt.hour == 0:
                fig_wind_adj.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
        st.plotly_chart(fig_wind_adj, use_container_width=True)
    else:
        st.subheader("Wind Adjustment (Disabled)")
        st.caption("This section is currently disabled.")


    # --- Visualize Cloud Adjustment ---
    if use_cloud:
        st.subheader("Cloud Adjustment")
        fig_cloud = px.line(df, x="datetime_local", y="cloud", title="Cloud Cover Forecast", labels={"datetime_local": "Local Time", "cloud": "Cloud (%)"})
        fig_cloud.update_traces(mode="lines+markers")
        fig_cloud.update_layout(height=300)
        for dt in midnights:
            if dt.hour == 0:
                fig_cloud.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
        st.plotly_chart(fig_cloud, use_container_width=True)

        # --- Model Curves for Multiple Apparent Temps ---
        cloud_range = np.linspace(0, 100, 300)
        cloud_temps = [45, 55,60, 65,70, 75, 85]
        cloud_colors = plt.cm.coolwarm(np.linspace(0, 1, len(cloud_temps)))

        fig4, ax4 = plt.subplots(figsize=PLOT_SIZE)

        for i, temp in enumerate(cloud_temps):
            app_temp_array = np.full_like(cloud_range, temp)
            curve = seasonal_cloud_adjustment(
                cloud_range,
                app_temp_array,
                pd.to_datetime(df["date"].iloc[0]).date(),
                latitude,
                center_temp=65.0,
                min_scale=cloud_min,
                max_scale=cloud_max,
                steepness=cloud_k
            )
            ax4.plot(cloud_range, curve, color=cloud_colors[i], label=f"{temp}°F")

        # Plot forecast points
        ax4.scatter(df["cloud"], df["cloud_adj"], color="black", zorder=5, label="Forecast Points")

        ax4.set_xlabel("Cloud Cover (%)")
        ax4.set_ylabel("Cloud Adjustment")
        ax4.set_title("Cloud Adjustment Curves Across Apparent Temperatures")
        ax4.grid(True, linestyle="--", color="lightgray", linewidth=0.5)
        ax4.legend(ncol=3, fontsize="small", loc="upper right")
        st.pyplot(fig4)


        fig_cloud_adj = px.line(df, x="datetime_local", y="cloud_adj", title="Cloud Adjustment Over Time", labels={"datetime_local": "Local Time", "cloud_adj": "Adjustment"})
        fig_cloud_adj.update_traces(mode="lines+markers")
        fig_cloud_adj.update_layout(height=300)

        # Plot static shade cloud adjustment for 100% cloud cover
        shade_cloud_adj_static = seasonal_cloud_adjustment(100, df["apparent_temp"], pd.to_datetime(df["date"].iloc[0]).date(), latitude,
                                                        center_temp=65.0, min_scale=cloud_min, max_scale=cloud_max,
                                                        steepness=cloud_k)
        fig_cloud_static = px.line(df, x="datetime_local", y=shade_cloud_adj_static, title="Shade Adjustment (100% Cloud)", labels={"datetime_local": "Local Time", "value": "Shade Cloud Adj"})
        fig_cloud_static.update_traces(mode="lines+markers")
        fig_cloud_static.update_layout(height=300)
        for dt in midnights:
            if dt.hour == 0:
                fig_cloud_static.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
        for dt in midnights:
            if dt.hour == 0:
                fig_cloud_adj.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
        st.plotly_chart(fig_cloud_adj, use_container_width=True)
        st.plotly_chart(fig_cloud_static, use_container_width=True)
    else:
        st.subheader("Cloud Adjustment (Disabled)")
        st.caption("This section is currently disabled.")


    # --- Visualize Precipitation Adjustment ---
    if use_precip:
        st.subheader("Precipitation Adjustment")
        fig_chance = px.line(df, x="datetime_local", y="chance_pcp", title="Precipitation Chance Forecast", labels={"datetime_local": "Local Time", "chance_pcp": "Chance (%)"})
        fig_chance.update_traces(mode="lines+markers")
        fig_chance.update_layout(height=300)
        for dt in midnights:
            if dt.hour == 0:
                fig_chance.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
        st.plotly_chart(fig_chance, use_container_width=True)

        fig_precip_amt = px.line(df, x="datetime_local", y="precip", title="Precipitation Amount Forecast", labels={"datetime_local": "Local Time", "precip": "Precip (in/hr)"})
        fig_precip_amt.update_traces(mode="lines+markers")
        fig_precip_amt.update_layout(height=300)
        for dt in midnights:
            if dt.hour == 0:
                fig_precip_amt.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
        st.plotly_chart(fig_precip_amt, use_container_width=True)

        p_range = np.linspace(0, 100, 300)
        p_curve = precipitation_adjustment(p_range, precip_in_hr=0.2, scale=100, p0=precip_p0, k=precip_k, light_threshold=precip_thresh)

        fig5, ax5 = plt.subplots(figsize=PLOT_SIZE)
        ax5.plot(p_range, p_curve, label="Precip Adj Curve (0.2 in/hr)", color="red")
        ax5.scatter(df["chance_pcp"], df["precip_adj"], color="black", zorder=5, label="Forecast Points")

        # Add half-penalty reference line
        ax5.plot(p_range, p_curve / 2, linestyle="--", color="darkred", label="Half Penalty Curve")


        ax5.set_xlabel("Chance of Precipitation (%)")
        ax5.set_ylabel("Precip Adjustment")
        ax5.set_title("Precipitation Adjustment Curve")
        ax5.grid(True, linestyle="--", color="lightgray", linewidth=0.5)
        ax5.legend(fontsize="small")
        st.pyplot(fig5)

        fig_precip_adj = px.line(df, x="datetime_local", y="precip_adj", title="Precipitation Adjustment Over Time", labels={"datetime_local": "Local Time", "precip_adj": "Adjustment"})
        fig_precip_adj.update_traces(mode="lines+markers")
        fig_precip_adj.update_layout(height=300)
        for dt in midnights:
            if dt.hour == 0:
                fig_precip_adj.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
        st.plotly_chart(fig_precip_adj, use_container_width=True)
    else:
        st.subheader("Precipitation Adjustment (Disabled)")
        st.caption("This section is currently disabled.")

    # --- Visualize Humidity Adjustment ---
    if use_humidity:
        st.subheader("Humidity Adjustment")
        fig_rh = px.line(df, x="datetime_local", y="rh", title="Relative Humidity Forecast", labels={"datetime_local": "Local Time", "rh": "RH (%)"})
        fig_rh.update_traces(mode="lines+markers")
        fig_rh.update_layout(height=300)
        midnights = df["datetime_local"].dt.tz_convert(None)
        for dt in midnights:
            if dt.hour == 0:
                fig_rh.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
        st.plotly_chart(fig_rh, use_container_width=True)

        rh_range = np.linspace(0, 100, 300)
        rh_curve = logistic_adjustment(rh_range, rh_center, rh_k, rh_scale)

        fig2, ax2 = plt.subplots(figsize=PLOT_SIZE)
        ax2.plot(rh_range, rh_curve, label="RH Adj Curve", color="green")
        ax2.scatter(df["rh"], df["rh_adj"], color="black", zorder=5, label="Forecast Points")
        ax2.set_xlabel("Relative Humidity (%)")
        ax2.set_ylabel("RH Adjustment")
        ax2.set_title("Humidity Adjustment Curve")
        ax2.grid(True)
        st.pyplot(fig2)

        fig_rh_adj = px.line(df, x="datetime_local", y="rh_adj", title="Humidity Adjustment Over Time", labels={"datetime_local": "Local Time", "rh_adj": "Adjustment"})
        fig_rh_adj.update_traces(mode="lines+markers")
        fig_rh_adj.update_layout(height=300)
        for dt in midnights:
            if dt.hour == 0:
                fig_rh_adj.add_vline(x=dt, line_width=1, line_dash="dot", line_color="lightgray")
        st.plotly_chart(fig_rh_adj, use_container_width=True)
    else:
        st.subheader("Humidity Adjustment (Disabled)")
        st.caption("This section is currently disabled.")

    # --- Final Score Visualization: Sun vs Shade ---
    st.subheader("Final Scores: Sun vs. Shade")

    # Recalculate cloud adjustment with 100% cloud cover
    shade_cloud_adj = seasonal_cloud_adjustment(100, df["apparent_temp"], pd.to_datetime(df["date"].iloc[0]).date(), latitude,
                                                center_temp=65.0, min_scale=cloud_min, max_scale=cloud_max,
                                                steepness=cloud_k)

    if isinstance(shade_cloud_adj, (int, float)):
        df["score_shade"] = (df["base"] + df["rh_adj"] + df["wind_adj"] + shade_cloud_adj + df["precip_adj"]).clip(upper=100)
    else:
        df["score_shade"] = (df["base"] + df["rh_adj"] + df["wind_adj"] + shade_cloud_adj + df["precip_adj"]).clip(upper=100)

    df["score_sun"] = df["score"].clip(upper=100)

    fig6, ax6 = plt.subplots(figsize=(8, 3))
    ax6.plot(df["datetime_local"], df["score_sun"], label="Sun Score", color="gold")
    ax6.plot(df["datetime_local"], df["score_shade"], label="Shade Score (100% cloud)", color="gray")
    ax6.set_ylabel("Stoke Score")
    ax6.set_xlabel("Datetime")
    ax6.set_title("Sun vs. Shade Scores")
    ax6.grid(True)
    for dt in pd.to_datetime(df["datetime_local"]).dt.tz_localize(None):
        if dt.hour == 0:
            ax6.axvline(dt, color="black", linestyle=":", linewidth=0.5)
    ax6.legend()
    st.pyplot(fig6, use_container_width=True)
    st.write("**Sun Score:** Uses real cloud values")
    st.write("**Shade Score:** Assumes 100% cloud cover (i.e., no sun benefit)")
