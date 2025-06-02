
import numpy as np
from datetime import date

def unified_base_score(temp_app_f, temp_opt=70, width_low=20, width_high=15, value_at_optimal=100):
    width = np.where(temp_app_f < temp_opt, width_low, width_high)
    return value_at_optimal * np.exp(-((temp_app_f - temp_opt) ** 2) / (2 * width ** 2))

def logistic_adjustment(x, center, steepness, scale=10):
    return scale * (1 / (1 + np.exp(steepness * (x - center))) - 0.5)

def blended_wind_adjustment(wspd, temp_app_f, tmin=50, tmax=70,
                             cool_scale=20, decay=0.05,
                             warm_ideal=10, warm_width=5, warm_scale=20):
    weight = np.clip((temp_app_f - tmin) / (tmax - tmin), 0, 1)
    cool_adj = -cool_scale * (1 - np.exp(-decay * wspd))
    warm_adj = warm_scale * np.exp(-((wspd - warm_ideal) ** 2) / (2 * warm_width ** 2)) - warm_scale / 2
    return (1 - weight) * cool_adj + weight * warm_adj

def seasonal_cloud_adjustment(cloud_percent, temp_app_f, date_obj, lat_deg,
                              center_temp=65.0, min_scale=7.0, max_scale=25.0,
                              steepness=0.08):
    """
    Cloud adjustment model that scales score impact based on apparent temperature,
    seasonal solar angle, and logistic decay with cloud cover.

    On cool days, clear skies are beneficial (positive adjustment).
    On warm days, clear skies are detrimental (negative adjustment).
    Adjustment fades to 0 with increasing cloud cover using a logistic curve.
    """
    cloud_arr = np.asarray(cloud_percent)
    temp_arr = np.asarray(temp_app_f)

    doy = date_obj.timetuple().tm_yday
    decl = 23.44 * np.pi / 180 * np.sin(2 * np.pi * (doy - 81) / 365)
    lat_rad = np.radians(lat_deg)
    decl_max = 23.44 * np.pi / 180
    solar_angle = np.arcsin(np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl))
    solar_angle_max = np.arcsin(np.sin(lat_rad) * np.sin(decl_max) + np.cos(lat_rad) * np.cos(decl_max))
    solar_power = np.clip(np.sin(solar_angle) / np.sin(solar_angle_max), 0, 1)

    seasonal_max = min_scale + (max_scale - min_scale) * (solar_power ** 1.25)
    temp_diff = temp_arr - center_temp
    temp_scale = np.clip(np.abs(temp_diff) / 15, 0, 1)
    signed_scale = seasonal_max * temp_scale * -np.sign(temp_diff)

    # Logistic decay from 1 to 0 as cloud cover increases
    decay_factor = 1 / (1 + np.exp(steepness * (cloud_arr - 50)))
    adj = signed_scale * decay_factor
    return adj

def seasonal_cloud_adjustment_old(cloud_percent, temp_app_f, current_date, lat_deg,
                                 cloud_params):
    """
    Seasonally adjusted cloud cover score adjustment based on latitude, date, and temperature.

    This function calculates how clouds affect the stokescore, which varies by:
    - Season (clouds have stronger effects in summer than winter)
    - Temperature (clouds are beneficial in hot weather, detrimental in cold weather)
    - Latitude (sun angle affects the impact of clouds)

    The adjustment can be positive or negative:
    - Positive adjustment (bonus): When it's cold and clouds are absent (sunny)
    - Negative adjustment (penalty): When it's hot and clouds are absent (sunny)
    - Near zero adjustment: When cloud cover is around 100% or temperature is moderate

    This complex calculation accounts for the fact that sun exposure has different
    effects depending on conditions (e.g., sun feels good when it's cold, but can
    be uncomfortable when it's hot).

    Parameters:
    - cloud_percent: Cloud cover (0–100%)
    - temp_app_f: Apparent temp (°F)
    - current_date: datetime.date object for seasonal calculations
    - lat_deg: Latitude in degrees (can be a 2D array)
    - cloud_params: Activity-specific cloud parameters (center_temp, min_scale, max_scale, steepness_base)

    Returns:
    - Cloud adjustment score (positive in cold sun, negative in hot sun)
    """
    center_temp, min_scale, max_scale, steepness_base = cloud_params

    cloud_arr = np.asarray(cloud_percent)
    temp_arr = np.asarray(temp_app_f)
    lat_arr = np.asarray(lat_deg)

    if temp_arr.ndim > lat_arr.ndim:
        lat_arr = np.expand_dims(lat_arr, axis=0)

    doy = current_date.timetuple().tm_yday
    decl = 23.44 * np.pi / 180 * np.sin(2 * np.pi * (doy - 81) / 365)
    lat_rad = np.radians(lat_arr)
    solar_angle = np.arcsin(np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl))

    decl_max = 23.44 * np.pi / 180
    solar_angle_max = np.arcsin(np.sin(lat_rad) * np.sin(decl_max) + np.cos(lat_rad) * np.cos(decl_max))
    solar_angle_max = np.where(np.abs(solar_angle_max) < 1e-10, 1e-10, solar_angle_max)
    solar_power = np.clip(np.sin(solar_angle) / np.sin(solar_angle_max), 0, 1)

    steepness = steepness_base * (1 + 2.0 * solar_power)
    seasonal_max = min_scale + (max_scale - min_scale) * (solar_power ** 1.25)

    temp_diff = np.abs(temp_arr - center_temp)
    scale = np.clip(seasonal_max * (temp_diff / 15), 0, seasonal_max)

    center_cloud = 100
    signed_steepness = np.where(temp_arr < center_temp, steepness, -steepness)
    # Adjust logistic function to converge to zero adjustment at 100% cloud cover
    return scale * (1 / (1 + np.exp(signed_steepness * (cloud_arr - center_cloud))) - 0.5)

def precipitation_adjustment(p, precip_in_hr=0.0, scale=100, p0=60, k=0.1, light_threshold=0.1):
    base_adj = -scale * (1 / (1 + np.exp(-k * (p - p0))))
    if isinstance(precip_in_hr, (np.ndarray, list)):
        precip_in_hr = np.asarray(precip_in_hr)
        return np.where(precip_in_hr < light_threshold, base_adj / 2, base_adj)
    else:
        return base_adj / 2 if precip_in_hr < light_threshold else base_adj

def calculate_apparent_temperature(temp_f, rh, wind_mph):
    """
    Calculate apparent temperature using NOAA rules:
    - Wind Chill if temp ≤ 50°F and wind > 3 mph
    - Heat Index if temp ≥ 80°F
    - Otherwise, use actual temp

    Parameters:
    - temp_f: Temperature in °F
    - rh: Relative humidity in %
    - wind_mph: Wind speed in mph

    Returns:
    - Apparent temperature in °F
    """
    temp_f = np.asarray(temp_f)
    rh = np.asarray(rh)
    wind_mph = np.asarray(wind_mph)

    # Heat Index formula (Steadman's empirical formula for > 80°F)
    def heat_index(t, h):
        return (-42.379 + 2.04901523 * t + 10.14333127 * h
                - 0.22475541 * t * h - 6.83783e-3 * t**2
                - 5.481717e-2 * h**2 + 1.22874e-3 * t**2 * h
                + 8.5282e-4 * t * h**2 - 1.99e-6 * t**2 * h**2)

    # Wind Chill formula (NOAA definition valid for ≤ 50°F and wind > 3 mph)
    def wind_chill(t, w):
        return 35.74 + 0.6215 * t - 35.75 * w**0.16 + 0.4275 * t * w**0.16

    apparent = np.where(
        temp_f <= 50,
        np.where(wind_mph > 3, wind_chill(temp_f, wind_mph), temp_f),
        np.where(temp_f >= 80, heat_index(temp_f, rh), temp_f)
    )

    return apparent
