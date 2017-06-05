"""
Tools for PVSC-44 Clearysky AOD Analysis
"""

# imports
from datetime import datetime
from ftplib import FTP
import logging
import os
from StringIO import StringIO
import time
import threading
from Queue import Queue

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
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
# create console handler and set level to debug
CH = logging.StreamHandler()
CH.setLevel(logging.DEBUG)
# create formatter
LOGFMT = ('[%(levelname)s:%(name)s:%(lineno)d] <%(thread)d> (%(asctime)s)\n'
          '%(message)s')
FORMATTER = logging.Formatter(LOGFMT, '%m/%d/%Y %I:%M:%S %p')
# add formatter to ch
CH.setFormatter(FORMATTER)
# add ch to logger
LOGGER.addHandler(CH)

# timezone lookup, force nearest tz for coords outside of polygons
WHERETZ = tzwhere.tzwhere(shapely=True, forceTZ=True)
# daylight savings time (DST) in northern hemisphere starts in March and ends
# in November and the opposite in southern hemisphere
JAN1 = datetime(2015, 1, 1)  # date with standard time in northern hemisphere
JUN1 = datetime(2015, 6, 1)  # date with standard time in southern hemisphere

# Get SURFRAD data from NOAA
PKGDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.dirname(PKGDIR)
SAVEDATAPATH = os.path.join(BASEDIR, 'surfrad')
ECMWF_MACC_START = 2003
ECMWF_MACC_STOP = 2012
NOAA_FTP = 'aftp.cmdl.noaa.gov'
SURFRAD_PATH = '/'.join(['data', 'radiation', 'surfrad'])
SURFRAD_SITES = {
    'Bondville_IL': 'bon', 'Boulder_CO': 'tbl', 'Desert_Rock_NV': 'dra',
    'Fort_Peck_MT': 'fpk', 'Goodwin_Creek_MS': 'gwn', 'Penn_State_PA': 'psu',
    'Sioux_Falls_SD': 'sxf'
}
# these sites have data outside of the time range of the atmospheric measurements
TOO_SOON = 'Alamosa_CO', 'Red_Lake_AZ', 'Rutland_VT', 'Wasco_OR'
DTYPES = [
    int, int, int, int, int, int, float, float, float, int, float, int, float,
    int, float, int, float, int, float, int, float, int, float, int, float, int,
    float, int, float, int, float, int, float, int, float, int, float, int,
    float, int, float, int, float, int, float, int, float, int
]
NAMES = [
    'year', 'jday', 'month', 'day', 'hour', 'min', 'dt', 'zen', 'dw_solar',
    'qc_dwsolar', 'uw_solar', 'qc_uwsolar', 'direct_n', 'qc_direct_n',
    'diffuse', 'qc_diffuse', 'dw_ir', 'qc_dwir', 'dw_casetemp', 'qc_dwcasetemp',
    'dw_dometemp', 'qc_dwdometemp', 'uw_ir', 'qc_uwir', 'uw_casetemp',
    'qc_uwcasetemp', 'uw_dometemp', 'qc_uwdometemp', 'uvb', 'qc_uvb', 'par',
    'qc_par', 'netsolar', 'qc_netsolar', 'netir', 'qc_netir', 'totalnet',
    'qc_totalnet', 'temp', 'qc_temp', 'rh', 'qc_rh', 'windspd', 'qc_windspd',
    'winddir', 'qc_winddir', 'pressure', 'qc_pressure'
]
USE_COLS = [
    'year', 'month', 'day', 'hour', 'min', 'zen', 'dw_solar', 'qc_dwsolar',
    'direct_n', 'qc_direct_n', 'diffuse', 'qc_diffuse', 'temp', 'qc_temp', 'rh',
    'qc_rh', 'pressure', 'qc_pressure'
]
NA = '-9999.9'
IDX_COL = 'year_month_day_hour_min'
QUEUE = Queue()
RETRIES = 60
SLEEP = 15
MAX_CONN = 3
if not os.path.exists(SAVEDATAPATH):
    os.mkdir(SAVEDATAPATH)


def get_noaa_ftp_conn(noaa_ftp=NOAA_FTP, surfrad_path=SURFRAD_PATH, retry=0):
    """
    Get a NOAA FTP connection to the main SURFRAD folder.
    """
    try:
        noaa_ftp_conn = FTP(noaa_ftp)  # connection
    except Exception as ftp_err:
        LOGGER.exception(ftp_err)
        retry += 1
        if retry < RETRIES:
            time.sleep(SLEEP)
            LOGGER.debug('... retry connection #%d', retry)
            noaa_ftp_conn = get_noaa_ftp_conn(retry=retry)
        else:
            raise ftp_err
    noaa_ftp_conn.connect()  # if timedout
    noaa_ftp_conn.login()  # as anonymous
    noaa_ftp_conn.cwd(surfrad_path)  # navigate to surfrad folder
    return noaa_ftp_conn


def get_surfrad_site_year(surfrad_site, year, h5f_path, queue=QUEUE):
    """
    This is the target function of a thread. It creates an FTP connection to
    NOAA, navigates to the surfrad data and retrieves all of the years in the
    desired range in a buffer.
    """
    meta = []
    try:
        noaa_ftp_conn = get_noaa_ftp_conn()
    except Exception as ftp_err:
        LOGGER.exception(ftp_err)
        meta.append({'year': year, 'site': surfrad_site, 'ftp_error': ftp_err})
        queue.put(['%s/%s' % (surfrad_site, year), meta])
        return
    noaa_ftp_conn.cwd(surfrad_site)  # open site folder
    noaa_ftp_conn.cwd(str(year))  # open year folder
    files = []  # get list of available daily datafiles
    noaa_ftp_conn.retrlines('NLST', lambda _: files.append(_))
    data = None
    for f in files:
        buf = StringIO()  # buffer is a Python builtin, **SURPRISE!**
        try:
            noaa_ftp_conn.retrbinary('RETR %s' % f, buf.write)
        except Exception as ftp_err:
            LOGGER.exception(ftp_err)
            meta.append({'year': year, 'file': f, 'ftp_error': ftp_err})
            buf.close()
            continue
        buf.seek(0)
        station_name = buf.readline().strip()
        latitude, longitude, elevation, _ = buf.readline().split(None, 3)
        meta.append({
            'year': year, 'file': f, 'station name': station_name,
            'latitude': float(latitude), 'longitude': float(longitude),
            'elevation': float(elevation)
        })
        df = pd.read_table(
            buf, names=NAMES, dtype=zip(NAMES, DTYPES), usecols=USE_COLS,
            delim_whitespace=True, na_values=NA, index_col=IDX_COL,
            parse_dates=[['year', 'month', 'day', 'hour', 'min']],
            date_parser=lambda y, mo, d, h, m: datetime(y, mo, d, h, m),
        )
        # filter out bad or missing data
        df = df.dropna()
        for col in USE_COLS:
            if col.startswith('qc'):
                df = df[df[col]==0]
                df = df.drop(col, 1)
        # append to data table
        if data is None:
            data = df
        else:
            data = data.append(df)
        LOGGER.debug('... file: %s loaded', f)
        buf.close()
    # save the data to an hdf5 file
    data_array = data.to_records()
    data_array = data_array.astype(np.dtype(
        [(IDX_COL, str, 30)]
        + [(str(n), d) for n, d in data_array.dtype.descr if n != IDX_COL]
    ))
    try:
        with h5py.File(h5f_path) as h5f:
            h5f['data'] = data_array
    except RuntimeError as h5_err:
        LOGGER.exception(h5_err)
        meta.append({'year': year, 'site': surfrad_site, 'h5_error': h5_err})
    # put the meta into the queue
    queue.put(['%s/%s' % (surfrad_site, year), meta])
    noaa_ftp_conn.cwd('..')  # navigate back to years
    noaa_ftp_conn.close()


def get_surfrad_data(surfrad_sites=SURFRAD_SITES, savedatapath=SAVEDATAPATH,
                     ecmwf_macc_range=(ECMWF_MACC_START, ECMWF_MACC_STOP)):
    # loop over sites
    threads = []
    noaa_ftp_conn = get_noaa_ftp_conn()
    for surfrad_site, station_id in surfrad_sites.iteritems():
        noaa_ftp_conn.cwd(surfrad_site)  # open site folder
        years = []  # get list of available years
        noaa_ftp_conn.retrlines('NLST', lambda _: years.append(_))
        # loop over yearly site data
        connections = []
        for y in years:
            # skip any non-year items in the folder
            try:
                year = int(y)
            except ValueError:
                continue
            h5f_name = '%s-%d.h5' % (station_id, year)
            h5f_path = os.path.join(savedatapath, h5f_name)
            if os.path.exists(h5f_path):
                continue
            # limit to data overlapping ECMWF atmospheric data (for now)
            if ecmwf_macc_range[0] <= year <= ecmwf_macc_range[1]:
                t = threading.Thread(target=get_surfrad_site_year,
                           args=(surfrad_site, year, h5f_path))
                t.start()
                connections.insert(0, t)
            if len(connections) > MAX_CONN:
                LOGGER.debug('number of active threads exceeds max connections.')
                conn = connections.pop()
                threads.append(conn)
                conn.join()
        noaa_ftp_conn.cwd('..')  # navigate back to sites
    noaa_ftp_conn.close()
    return threads


def get_surfrad_station_meta(queue=QUEUE):
    stations = {}
    while queue.empty():
        meta = QUEUE.get()
        stations[meta[0]] = meta[1]
    return stations


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
    return tz.utcoffset(tz_date).total_seconds / 3600.0
    # alternate method using ISO8601 string repr of timezone
    #tz_str = tz.localize(tz_date).strftime('%z')  # output timezone from ISO
    # convert ISO timezone string to float, including partial timezones
    #return float(tz_str[:3]) + float(tz_str[3:]) / 60.0
