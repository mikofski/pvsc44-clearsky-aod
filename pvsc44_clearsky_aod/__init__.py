"""
Tools for PVSC-44 Clearysky AOD Analysis
"""

# imports
from datetime import datetime
from ftplib import FTP
import logging
import os
from StringIO import StringIO

import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from tzwhere import tzwhere

# set figure size
sns.set_context('notebook', rc={'figure.figsize': (16, 8)})

# logging
logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# timezone lookup, force nearest tz for coords outside of polygons
WHERETZ = tzwhere.tzwhere(shapely=True, forceTZ=True)
# daylight savings time (DST) in northern hemisphere starts in March and ends
# in November and the opposite in southern hemisphere
JAN1 = datetime(2015, 1, 1)  # date with standard time in northern hemisphere
JUN1 = datetime(2015, 6, 1)  # date with standard time in southern hemisphere

# Get SURFRAD data from NOAA
# TODO: this would work faster if ftp retreival is done in a thread target
TESTING = False  # flag for testing
try:
    BASEDIR = %pwd
except Exception:
    BASEDIR = os.path.dirname(__file__)
SAVEDATAPATH = os.path.join(BASEDIR, 'surfrad')
ECMWF_MACC_START = 2003
ECMWF_MACC_STOP = 2012
NOAA_FTP = 'aftp.cmdl.noaa.gov'
SURFRAD_PATH = '/'.join(['data', 'radiation', 'surfrad'])
SURFRAD_SITES = {'Bondville_IL': 'bon', 'Boulder_CO': 'tbl', 'Desert_Rock_NV': 'dra', 'Fort_Peck_MT': 'fpk',
                 'Goodwin_Creek_MS': 'gwn', 'Penn_State_PA': 'psu', 'Sioux_Falls_SD': 'sxf'}
# these sites have data outside of the time range of the atmospheric measurements
TOO_SOON = 'Alamosa_CO', 'Red_Lake_AZ', 'Rutland_VT', 'Wasco_OR'
DTYPES = [int, int, int, int, int, int, float, float, float, int, float, int, float, int, float, int, float, int, float, int,
          float, int, float, int, float, int, float, int, float, int, float, int, float, int, float, int, float, int, float, int,
          float, int, float, int, float, int, float, int]
NAMES = ['year', 'jday', 'month', 'day', 'hour', 'min', 'dt', 'zen', 'dw_solar', 'qc_dwsolar', 'uw_solar', 'qc_uwsolar',
         'direct_n', 'qc_direct_n', 'diffuse', 'qc_diffuse', 'dw_ir', 'qc_dwir', 'dw_casetemp', 'qc_dwcasetemp',
         'dw_dometemp', 'qc_dwdometemp', 'uw_ir', 'qc_uwir', 'uw_casetemp', 'qc_uwcasetemp', 'uw_dometemp', 'qc_uwdometemp',
         'uvb', 'qc_uvb', 'par', 'qc_par', 'netsolar', 'qc_netsolar', 'netir', 'qc_netir', 'totalnet', 'qc_totalnet',
         'temp', 'qc_temp', 'rh', 'qc_rh', 'windspd', 'qc_windspd', 'winddir', 'qc_winddir', 'pressure', 'qc_pressure']
USE_COLS = ['year', 'month', 'day', 'hour', 'min', 'zen',
            'dw_solar', 'qc_dwsolar', 'direct_n', 'qc_direct_n', 'diffuse', 'qc_diffuse',
            'temp', 'qc_temp', 'rh', 'qc_rh', 'pressure', 'qc_pressure']

noaa_ftp_conn = FTP(NOAA_FTP)  # connection
noaa_ftp_conn.connect()  # if timedout
noaa_ftp_conn.login()  # as anonymous
noaa_ftp_conn.cwd(SURFRAD_PATH)  # navigate to surfrad folder

if not os.path.exists(SAVEDATAPATH):
    os.mkdir(SAVEDATAPATH)

# loop over sites
data = {}
stations = {}
for surfrad_site, station_id in SURFRAD_SITES.iteritems():
    noaa_ftp_conn.cwd(surfrad_site)  # open site folder
    years = []  # get list of available years
    noaa_ftp_conn.retrlines('NLST', lambda _: years.append(_))
    # loop over yearly site data
    for y in years:
        # skip any non-year items in the folder
        try:
            year = int(y)
        except ValueError:
            continue
        h5f_name = '%s-%d.h5' % (station_id, year)
        if os.path.exists(os.path.join(SAVEDATAPATH, h5f_name)):
            continue
        # limit to data overlapping ECMWF atmospheric data (for now)
        if ECMWF_MACC_START <= year <= ECMWF_MACC_STOP:
            noaa_ftp_conn.cwd(y)  # open site folder
            files = []  # get list of available daily datafiles
            noaa_ftp_conn.retrlines('NLST', lambda _: files.append(_))
            for f in files:
                buf = StringIO()  # buffer is a Python builtin, **SURPRISE!**
                noaa_ftp_conn.retrbinary('RETR %s' % f, buf.write)
                buf.seek(0)
                station_name = buf.readline().strip()
                latitude, longitude, elevation, _ = buf.readline().split(None, 3)
                if station_id not in stations:
                    stations[station_id] = [(year, f, station_name, float(latitude), float(longitude), float(elevation))]
                else:
                    stations[station_id].append((year, f, station_name, float(latitude), float(longitude), float(elevation)))
                df = pd.read_table(
                    buf, names=NAMES, dtype=zip(NAMES, DTYPES), usecols=USE_COLS, delim_whitespace=True, na_values='-9999.9',
                    index_col='year_month_day_hour_min', parse_dates=[['year', 'month', 'day', 'hour', 'min']],
                    date_parser=lambda y, mo, d, h, m: datetime(y, mo, d, h, m),
                )
                # filter out bad or missing data
                df = df.dropna()
                for qc in ['qc_dwsolar', 'qc_direct_n', 'qc_diffuse', 'qc_temp', 'qc_rh', 'qc_pressure']:
                    df = df[df[qc]==0]
                    df = df.drop(qc, 1)
                # append to data table
                if station_id not in data:
                    data[station_id] = df
                else:
                    data[station_id] = data[station_id].append(df)
                if TESTING: LOGGER.debug('file: %s loaded', f)
                buf.close()
            noaa_ftp_conn.cwd('..')  # navigate back to years
            start, stop = ('%d-01-01T00:00:00' % year), ('%d-12-31T23:59:59' % year)
            data_array = data[station_id][start:stop].to_records()
            data_array = data_array.astype(np.dtype(
                [('year_month_day_hour_min', str, 30)]
                + [(str(n), d) for n, d in data_array.dtype.descr if n != 'year_month_day_hour_min']
            ))
            try:
                with h5py.File(os.path.join(SAVEDATAPATH, h5f_name)) as h5f: h5f['data'] = data_array
            except RuntimeError:
                pass
            if TESTING: break
        if TESTING: break
    if TESTING: break
    noaa_ftp_conn.cwd('..')  # navigate back to sites


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
        tz_info = pytz.timezone(tz_name)  # get tzinfo
    else:
        # coordinates over international waters only depend on longitude
        return lon//15.0
    tz_date = JAN1  # standard time in northern hemisphere
    # get the daylight savings time timedelta
    if tz_info.dst(tz_date):
        # if DST timedelta is not zero, then it must be southern hemisphere
        tz_date = JUN1  # a standard time in southern hemisphere
    return tz.utcoffset(tz_date).total_seconds / 3600.0
    # alternate method using ISO8601 string repr of timezone
    #tz_str = tz_info.localize(tz_date).strftime('%z')  # output timezone from ISO
    # convert ISO timezone string to float, including partial timezones
    #return float(tz_str[:3]) + float(tz_str[3:]) / 60.0
