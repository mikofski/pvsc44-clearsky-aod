"""
Tools for PVSC-44 Clearysky AOD Analysis
"""

from datetime import datetime

import pytz
from tzwhere import tzwhere

# timezone lookup, force nearest tz for coords outside of polygons
WHERETZ = tzwhere.tzwhere(shapely=True, forceTZ=True)
# daylight savings time (DST) in northern hemisphere starts in March and ends
# in November and the opposite in southern hemisphere
JAN1 = datetime(2015, 1, 1)  # date with standard time in northern hemisphere
JUN1 = datetime(2015, 6, 1)  # date with standard time in southern hemisphere


def tz_latlon(lat, lon):
    """
    Timezone from latitude and longitude.

    :param lat: latitude [deg]
    :type lat: float
    :param lon: longitude [deg]
    :type lon: float
    :return: timezone
    :rtype: float
    """
    # get name of time zone using tzwhere, force to nearest tz
    tz_name = WHERETZ.tzNameAt(lat, lon, forceTZ=True)
    # check if coordinates are over international waters
    if tz_name:
        tz = pytz.timezone(tz_name)  # get tzinfo
    else:
        # coordinates over international waters only depend on longitude
        return lon//15.0
    tz_date = JAN1  # standard time in northern hemisphere
    # get the daylight savings time timedelta
    if tz.dst(tz_date):
        # if DST timedelta is not zero, then it must be southern hemisphere
        tz_date = JUN1  # a standard time in southern hemisphere
    return tz.utcoffset(tz_date).total_seconds() / 3600.0
    # alternate method using ISO8601 string repr of timezone
    #tz_str = tz.localize(tz_date).strftime('%z')  # output timezone from ISO
    # convert ISO timezone string to float, including partial timezones
    #return float(tz_str[:3]) + float(tz_str[3:]) / 60.0
