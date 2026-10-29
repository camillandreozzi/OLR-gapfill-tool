"""Solar geometry: solar zenith angle and local solar time, NOAA algorithm (accuracy ~0.1 deg).

Based on the NOAA solar calculator equations (Meeus, Astronomical Algorithms): solar declination
and the equation of time from the Julian century, then the hour angle from true solar time.
Refraction is ignored, which is irrelevant for the night / day split used here (SZA > 100 deg).
"""

import numpy as np
import pandas as pd

JULIAN_UNIX_EPOCH = 2440587.5  # Julian date of 1970-01-01T00:00Z
J2000 = 2451545.0


def _julian_date(time):
    """Julian date (UTC) of datetime-like values (array, Series or DatetimeIndex)."""
    t = pd.to_datetime(np.asarray(time).ravel()).values.astype("datetime64[ns]").astype(np.int64)
    return t / 86400e9 + JULIAN_UNIX_EPOCH


def declination_eot(time):
    """Return (declination in deg, equation of time in minutes) for each UTC time."""
    jc = (_julian_date(time) - J2000) / 36525.0  # Julian century
    l0 = np.radians((280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360)  # geometric mean longitude
    m = np.radians(357.52911 + jc * (35999.05029 - 0.0001537 * jc))  # geometric mean anomaly
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)  # orbit eccentricity
    c = (np.sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(2 * m) * (0.019993 - 0.000101 * jc) + np.sin(3 * m) * 0.000289)  # equation of centre, deg
    omega = np.radians(125.04 - 1934.136 * jc)
    app_long = np.radians(np.degrees(l0) + c - 0.00569 - 0.00478 * np.sin(omega))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = np.radians(eps0 + 0.00256 * np.cos(omega))
    decl = np.degrees(np.arcsin(np.sin(eps) * np.sin(app_long)))
    y = np.tan(eps / 2) ** 2
    eot = 4 * np.degrees(y * np.sin(2 * l0) - 2 * e * np.sin(m) + 4 * e * y * np.sin(m) * np.cos(2 * l0)
                         - 0.5 * y * y * np.sin(4 * l0) - 1.25 * e * e * np.sin(2 * m))
    return decl, eot


def local_solar_time(time, lon):
    """True local solar time in hours [0, 24) at longitude lon (deg east) for UTC time."""
    _, eot = declination_eot(time)
    t = pd.to_datetime(np.asarray(time).ravel())
    utc_min = t.hour.values * 60 + t.minute.values + t.second.values / 60 + t.microsecond.values / 6e7
    return ((utc_min + eot + 4 * np.asarray(lon, float).ravel()) % 1440) / 60


def solar_zenith(time, lat, lon):
    """Solar zenith angle (deg) at (lat, lon) in deg for UTC time; arrays broadcast elementwise."""
    decl, _ = declination_eot(time)
    tst = local_solar_time(time, lon)
    ha = np.radians(tst * 15 - 180)  # hour angle, 0 at local solar noon
    lat, decl = np.radians(np.asarray(lat, float).ravel()), np.radians(decl)
    cos_sza = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arccos(np.clip(cos_sza, -1, 1)))
