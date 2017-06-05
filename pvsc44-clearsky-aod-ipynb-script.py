"""
Jupyter notebook script used to download and review SURFRAD data.
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
import pytz
import seaborn as sns

# set figure size
try:
    %matplotlib inline
except Exception:
    plt.ion()
sns.set_context('notebook', rc={'figure.figsize': (16, 8)})

# logging
logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# Get SURFRAD data from NOAA
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