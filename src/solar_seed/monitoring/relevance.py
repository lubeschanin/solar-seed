"""
Personal Relevance
==================

Calculate user-specific solar event relevance based on location and time.

Physics:
- Ionospheric effects (SID, radio blackout, GPS): Only day side, ~8 min latency
- Geomagnetic effects (storms, aurora): Day + night, 15-96 hour latency
"""

from datetime import datetime, timezone
from typing import NamedTuple
import warnings

try:
    # Suppress ERFA/IERS warnings for future dates
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        from astropy.coordinates import EarthLocation, AltAz, get_sun
        from astropy.time import Time
        # Disable IERS auto-download and age checks
        from astropy.utils.iers import conf
        conf.auto_download = False
        conf.auto_max_age = None
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

# Use simple geometric calculation instead of astropy to avoid IERS issues
USE_SIMPLE_CALCULATION = True


class SunStatus(NamedTuple):
    """Sun visibility status for a location."""
    is_visible: bool
    altitude_deg: float
    azimuth_deg: float
    status: str  # 'DAY', 'CIVIL_TWILIGHT', 'NAUTICAL_TWILIGHT', 'NIGHT'


class PersonalRelevance(NamedTuple):
    """Personal relevance assessment for solar events."""
    location_name: str
    latitude: float
    longitude: float
    local_time: str
    timezone_name: str
    sun_status: SunStatus
    immediate_risk: str      # Risk from immediate effects (radio, GPS)
    delayed_risk: str        # Risk from delayed effects (geomag, aurora)
    daylight_window_utc: tuple[str, str]  # Approximate sunrise/sunset UTC
    aurora_possible: bool    # Based on latitude and Kp threshold


# Default locations
LOCATIONS = {
    'berlin': (52.52, 13.405, 'Europe/Berlin'),
    'london': (51.51, -0.13, 'Europe/London'),
    'new_york': (40.71, -74.01, 'America/New_York'),
    'tokyo': (35.68, 139.69, 'Asia/Tokyo'),
    'sydney': (-33.87, 151.21, 'Australia/Sydney'),
    'los_angeles': (34.05, -118.24, 'America/Los_Angeles'),
}


def get_sun_status(lat: float, lon: float, time_utc: datetime = None) -> SunStatus:
    """
    Calculate sun position and visibility for a location.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        time_utc: Time in UTC (default: now)

    Returns:
        SunStatus with altitude, azimuth, and visibility classification
    """
    import math

    if time_utc is None:
        time_utc = datetime.now(timezone.utc)

    # Use simple geometric calculation (avoids IERS/astropy issues)
    # This is accurate enough for day/night determination

    # Day of year for declination
    day_of_year = time_utc.timetuple().tm_yday

    # Solar declination (degrees)
    declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))

    # Hour angle (degrees from solar noon)
    hour_utc = time_utc.hour + time_utc.minute / 60 + time_utc.second / 3600
    solar_noon_utc = 12 - lon / 15  # When sun is highest for this longitude
    hour_angle = 15 * (hour_utc - solar_noon_utc)  # 15° per hour

    # Solar altitude using spherical geometry
    lat_rad = math.radians(lat)
    dec_rad = math.radians(declination)
    ha_rad = math.radians(hour_angle)

    sin_alt = (math.sin(lat_rad) * math.sin(dec_rad) +
               math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad))
    alt = math.degrees(math.asin(max(-1, min(1, sin_alt))))

    # Solar azimuth (simplified)
    if math.cos(math.radians(alt)) != 0:
        cos_az = ((math.sin(dec_rad) - math.sin(lat_rad) * math.sin(math.radians(alt))) /
                  (math.cos(lat_rad) * math.cos(math.radians(alt))))
        cos_az = max(-1, min(1, cos_az))
        az = math.degrees(math.acos(cos_az))
        if hour_angle > 0:
            az = 360 - az
    else:
        az = 180

    # Classify based on altitude
    if alt > 0:
        status = 'DAY'
        is_visible = True
    elif alt > -6:
        status = 'CIVIL_TWILIGHT'
        is_visible = False
    elif alt > -12:
        status = 'NAUTICAL_TWILIGHT'
        is_visible = False
    else:
        status = 'NIGHT'
        is_visible = False

    return SunStatus(
        is_visible=is_visible,
        altitude_deg=alt,
        azimuth_deg=az,
        status=status
    )


def get_daylight_window_utc(lat: float, lon: float, date: datetime = None) -> tuple[str, str]:
    """
    Get approximate sunrise/sunset times in UTC for a location.

    Returns ('HH:MM', 'HH:MM') tuple for sunrise and sunset UTC.

    Uses simple geometric approximation (no external data dependencies).
    """
    import math

    if date is None:
        date = datetime.now(timezone.utc)

    # Day of year for declination calculation
    day_of_year = date.timetuple().tm_yday

    # Solar declination (approximate)
    declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))

    # Hour angle at sunrise/sunset
    lat_rad = math.radians(lat)
    dec_rad = math.radians(declination)

    # cos(hour_angle) = -tan(lat) * tan(dec)
    cos_ha = -math.tan(lat_rad) * math.tan(dec_rad)

    # Clamp to valid range (handles polar day/night)
    if cos_ha < -1:
        return "00:00", "24:00"  # Polar day
    elif cos_ha > 1:
        return "12:00", "12:00"  # Polar night

    hour_angle = math.degrees(math.acos(cos_ha))

    # Solar noon in UTC based on longitude
    solar_noon_utc = 12 - lon / 15  # hours
    solar_noon_utc = solar_noon_utc % 24

    # Sunrise and sunset
    sunrise_hours = (solar_noon_utc - hour_angle / 15) % 24
    sunset_hours = (solar_noon_utc + hour_angle / 15) % 24

    sunrise_str = f"{int(sunrise_hours):02d}:{int((sunrise_hours % 1) * 60):02d}"
    sunset_str = f"{int(sunset_hours):02d}:{int((sunset_hours % 1) * 60):02d}"

    return sunrise_str, sunset_str


def assess_personal_relevance(
    lat: float,
    lon: float,
    location_name: str = "Unknown",
    timezone_name: str = "UTC",
    kp_index: float = None,
) -> PersonalRelevance:
    """
    Assess personal relevance of solar events for a specific location.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        location_name: Human-readable location name
        timezone_name: Timezone name (for display)
        kp_index: Current Kp index (for aurora prediction)

    Returns:
        PersonalRelevance assessment
    """
    now_utc = datetime.now(timezone.utc)
    sun_status = get_sun_status(lat, lon, now_utc)

    # Try to get local time
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(timezone_name)
        local_time = now_utc.astimezone(tz).strftime('%H:%M %Z')
    except Exception:
        local_time = now_utc.strftime('%H:%M UTC')

    # Immediate risk (radio, GPS) - only if sun is visible
    if sun_status.is_visible:
        if sun_status.altitude_deg > 30:
            immediate_risk = "HIGH - You are on the sunlit side (radio/GPS vulnerable)"
        else:
            immediate_risk = "MODERATE - Sun low, partial exposure"
    else:
        immediate_risk = "LOW - You are on the night side"

    # Delayed risk (geomagnetic) - always possible
    delayed_risk = "POSSIBLE - Geomagnetic effects in 15-48h if major event"

    # Daylight window
    daylight_window = get_daylight_window_utc(lat, lon, now_utc)

    # Aurora possibility based on latitude
    # Aurora typically visible at |lat| > 60° for Kp=5, > 50° for Kp=7, > 40° for Kp=9
    abs_lat = abs(lat)
    if kp_index is not None:
        aurora_threshold = 70 - kp_index * 5  # Rough approximation
        aurora_possible = abs_lat > aurora_threshold and not sun_status.is_visible
    else:
        aurora_possible = abs_lat > 55 and not sun_status.is_visible

    return PersonalRelevance(
        location_name=location_name,
        latitude=lat,
        longitude=lon,
        local_time=local_time,
        timezone_name=timezone_name,
        sun_status=sun_status,
        immediate_risk=immediate_risk,
        delayed_risk=delayed_risk,
        daylight_window_utc=daylight_window,
        aurora_possible=aurora_possible,
    )


def get_subsolar_point(time_utc: datetime = None) -> tuple[float, float]:
    """
    Get the subsolar point (where the sun is directly overhead).

    Returns (latitude, longitude) in degrees.
    """
    if time_utc is None:
        time_utc = datetime.now(timezone.utc)

    if not ASTROPY_AVAILABLE:
        # Rough approximation
        hour_utc = time_utc.hour + time_utc.minute / 60
        lon = -15 * hour_utc
        lon = ((lon + 180) % 360) - 180
        # Latitude varies with season (±23.5°)
        day_of_year = time_utc.timetuple().tm_yday
        lat = 23.5 * __import__('math').sin(2 * __import__('math').pi * (day_of_year - 81) / 365)
        return lat, lon

    from astropy.coordinates import GCRS
    t = Time(time_utc)
    sun = get_sun(t)
    # Convert to geographic coordinates
    # Subsolar latitude = solar declination
    lat = sun.dec.deg
    # Subsolar longitude = -15 * (hour angle from Greenwich)
    # At solar noon, sun is at the meridian
    gst = t.sidereal_time('apparent', 'greenwich').deg
    lon = (sun.ra.deg - gst + 180) % 360 - 180
    return lat, lon
