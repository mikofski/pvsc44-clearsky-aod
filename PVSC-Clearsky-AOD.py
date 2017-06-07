
# coding: utf-8

# # Use of Measured Aerosol Optical Depth and Precipitable Water to Model Clear Sky Irradiance
# *presented at: 2017 IEEE PVSC-44*
# *by: Mark A. Mikofski, Clifford W. Hansen, William F. Holmgren and Gregory M. Kimball*
# 
# ## Introduction
# Predicting irradiance is crucial for developing solar power systems. Clear sky models are used to predict irradiance based on solar position and atmospheric data. This paper compares clear sky predictions using atmospheric data from ECMWF with irradiance measurments from SURFRAD. This notebook presents the analysis for the paper and presentation given by the authors at the 2017 IEEE PVSC-44 in Washingtion DC June 25-30th.
# 
# ## How to use this Jupyter notebook
# This document is a [Jupyter notebook](http://jupyter.org/). It can be used several different ways. You can [view the notebook as a static HTML document](http://nbviewer.jupyter.org/github/mikofski/pvsc44-clearsky-aod/blob/master/PVSC-Clearsky-AOD.ipynb). You can also [clone the repository from GitHub using Git](https://github.com/mikofski/pvsc44-clearsky-aod), install [the requirements](https://github.com/mikofski/pvsc44-clearsky-aod/blob/master/requirements.txt) in a [Python virtual environment](https://virtualenv.pypa.io/en/stable/) and run the notebook interactively using [Python](https://www.python.org/).
# 
# ## PVLIB-Python
# This analysis uses Sandia National Laboratory's PVLIB-Python software extensively. PVLIB-Python is a library of functions for modeling photovoltaic devices, irradiance and atmospheric conditions. The [documentation for PVLIB-Python](http://pvlib-python.readthedocs.io/en/latest/) is online and there is more information at the [Sandia PV Performance Model Collaborative](https://pvpmc.sandia.gov/).

# In[1]:

# imports and settings
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import seaborn as sns
import statsmodels.api as sm

from pvsc44_clearsky_aod import ecmwf_macc_tools

get_ipython().magic(u'matplotlib notebook')

sns.set_context('notebook', rc={'figure.figsize': (16, 8)})


# ## Data
# Two sources of data are used in this analysis, SURFRAD and ECMWF, each described in the following sections.
# 
# ### SURFRAD
# [SURFRAD data from NOAA](https://www.esrl.noaa.gov/gmd/grad/surfrad/) contains irradiance measurements at sub-hour increments from seven stations accross the United States. The data can be viewed online or downloaded from a NOAA FTP site <ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/>. Data is organized into folders for each station and yearly subfolders containing individual daily files. The `README` in each station's folder provides additional information such as the format of the files, types of data in each column, definitions, and abbreviations. The first two rows of each daily file contains header information about the file. The first line is the station name and the second line contains the latitude, longitude and elevation. Data is timestamped at the beginning of each interval. Starting around 2008, most sites increased sampling to 1-minute intervals, other years sampled at 3-minute intervals. The irradiance data includes direct (DNI), diffuse (DHI), and downwelling solar irradiance which is equivalent global horizontal irradiance (GHI). Units of irradiance are W/m<sup>2</sup>. Ambient temperature is given in Celsius, pressure in millibars and relative humidity in percent.
# 
# There are seven sites with data that overlaps the time span of the atmospheric data (see the next section). The following metadata was collected from the daily data files, which all contain the same header for each site, and the `README`.
# 
# |station id |UTC offset |timezone   |station name  |City             |latitude |longitude |elevation |
# |-----------|-----------|-----------|--------------|-----------------|---------|----------|----------|
# |bon        |-6.0       |US/Central |     Bondville|Bondville, IL    |    40.05|    -88.37|       213|
# |tbl        |-7.0       |US/Mountain|Table Mountain|Boulder, CO      |    40.13|   -105.24|      1689|
# |dra        |-8.0       |US/Pacific |   Desert Rock|Desert Rock, NV  |    36.62|   -116.02|      1007|
# |fpk        |-7.0       |US/Mountain|     Fort Peck|Fort Peck, MT    |    48.31|   -105.10|       634|
# |gwn        |-6.0       |US/Central | Goodwin Creek|Goodwin Creek, MS|    34.25|    -89.87|        98|
# |psu        |-5.0       |US/Eastern |    Penn State|Penn State, PA   |    40.72|    -77.93|       376|
# |sxf        |-6.0       |US/Central |   Sioux Falls|Sioux Falls, SD  |    43.73|    -96.62|       473|
# 
# There are three additional sites, Alamosa-CO, Red Lake-AZ, Rutland-VT, and Wasco-OR, that don't have data that cover the same 10 years span of the atmospheric data. Since SURFRAD also contains some atmospheric data, perhaps we can return to examine these sites later.

# In[2]:

# get the "metadata" that contains the timezone, latitude, longitude, elevation and station id for SURFRAD
METADATA = pd.read_csv('metadata.csv', index_col=0)


# In[3]:

# get SURFRAD data
station_id = 'tbl'  # choose a station ID from the metadata list

# CONSTANTS
DATADIR = 'surfrad'  # folder where SURFRAD data is stored relative to this file
NA_VAL = '-9999.9'  # value SURFRAD uses for missing data
USECOLS = [0, 6, 7, 8, 9, 10, 11, 12]  # omit 1:year, 2:month, 3:day, 4:hour, 5:minute and 13:jday columns

# read 2003 data for the station, use first column as index, parse index as dates, and replace missing values with NaN
DATA = pd.read_csv(
    os.path.join(DATADIR, '%s03.csv' % station_id),
    index_col=0, usecols=USECOLS, na_values=NA_VAL, parse_dates=True
)

# append the data from 2003 to 2012 which coincides with the ECMWF data available
for y in xrange(2004, 2013):
    DATA = DATA.append(pd.read_csv(
        os.path.join(DATADIR, '%s%s.csv' % (station_id, str(y)[-2:])),
        index_col=0, usecols=USECOLS, na_values=NA_VAL, parse_dates=True
    ))

DATA['press'] = DATA['press'] * 100.0  # convert mbars to Pascals
DATA.index.rename('timestamps', inplace=True)  # rename the index to "timestamps"
DATA = DATA.tz_localize('UTC')  # set timezone to UTC


# In[4]:

# plot some temperatures
DATA['ta']['2006-07-16':'2006-07-17'].tz_convert(METADATA['tz'][station_id]).plot()
plt.ylabel('ambient temperature $[\degree C]$')


# In[5]:

# plot some irradiance
DATA[['dni', 'dhi', 'ghi']]['2006-07-16':'2006-07-17'].tz_convert(METADATA['tz'][station_id]).plot()
plt.ylabel('irradiance $[W/m^2]$')


# ### ECMWF
# The [European Center for Medium Weather Forecast (ECMWF)](https://www.ecmwf.int/en/forecasts/datasets) hosts atmospheric data from [Monitoring Amospheric Composition & Climate (MACC)](http://www.gmes-atmosphere.eu/) and the [EU Copernicus Atmospheric Monitoring Service (CAMS)](http://atmosphere.copernicus.eu/). This data set contains aerosol optical depth (AOD) measured at several wavelengths and total column water vapor (_AKA_: precipitable water) data in centimeters (cm) derived from ground measurments and satelite data for the entire globe from 2003 to 2012. The [data can be downloaded online](http://apps.ecmwf.int/datasets/data/macc-reanalysis/levtype=sfc/) or from the [ECMWF API using the Python API client after registering for an API key](https://software.ecmwf.int/wiki/display/WEBAPI/Access+ECMWF+Public+Datasets). The downloaded data used in this analysis has spatial resolution of 0.75 &deg; x 0.75 &deg;. The timestamps are every 3-hours marked at the end of the measurement window - _EG_: each timestamp is the average over the preceeding interval.

# In[6]:

# get AOD and water vapor for the SURFRAD station
ATMOS = ecmwf_macc_tools.Atmosphere()  # get the downloaded ECMWF-MACC data for the world

# create a datetime series for the ECMWF-MACC data, it's available from 2003 to 2012 in 3 hour increment
# pandas creates timestamps at the *beginning* of the each interval,shifting the timestamps from the end
# to match the SURFRAD data
TIMES = pd.DatetimeIndex(start='2003-01-01T00:00:00', freq='3H', end='2012-12-31T23:59:59').tz_localize('UTC')

# get the atmospheric data for the SURFRAD station as a pandas dataframe, and append some addition useful calculations
# like AOD at some other wavelengths (380-nm, 500-nm, and 700nm) and convert precipitable water to cm.
STATION_ATM = pd.DataFrame(
    ATMOS.get_all_data(METADATA['latitude'][station_id], METADATA['longitude'][station_id]), index=TIMES
)


# In[7]:

# plot some of the atmospheric data
atm_data = STATION_ATM[['tau380', 'tau500', 'aod550', 'tau700', 'aod1240']]['2006-07-16':'2006-07-17']
atm_data.tz_convert(METADATA['tz'][station_id]).plot()
plt.title('Atmospheric Data')
plt.ylabel('aerosol optical depth')


# ### Solar Position, Airmass and Precipitable Water
# NREL's Solar Position Algorithm (SPA) is used to find solar position including the effects of atmospheric refraction. The path length through the atmosphere or airmass is also calculated. Finally if water vapor measurements are not available, they can be estimated from relative humidity.

# In[8]:

# solarposition, airmass, pressure-corrected & water vapor
SOLPOS = pvlib.solarposition.get_solarposition(
    time=DATA.index,
    latitude=METADATA['latitude'][station_id],
    longitude=METADATA['longitude'][station_id],
    altitude=METADATA['elevation'][station_id],
    pressure=DATA['press'],
    temperature=DATA['ta']
)
AIRMASS = pvlib.atmosphere.relativeairmass(SOLPOS.apparent_zenith)  # relative airmass
AM_PRESS = pvlib.atmosphere.absoluteairmass(AIRMASS, pressure=DATA['press'])  # pressure corrected airmass
PWAT_CALC = pvlib.atmosphere.gueymard94_pw(DATA['ta'], DATA['rh'])  # estimate of atmospheric water vapor in cm
ETR = pvlib.irradiance.extraradiation(DATA.index)  # extra-terrestrial total solar radiation

# append these to the SOLPOS dataframe to keep them together more easily
SOLPOS.insert(6, 'am', AIRMASS)
SOLPOS.insert(7, 'amp', AM_PRESS)
SOLPOS.insert(8, 'pwat_calc', PWAT_CALC)
SOLPOS.insert(9, 'etr', ETR)


# In[9]:

# plot solar position for a day
SOLPOS[['zenith', 'apparent_zenith', 'azimuth']]['2006-07-16':'2006-07-17'].tz_convert(METADATA['tz'][station_id]).plot()
plt.ylabel('solar position $[\deg]$')


# In[10]:

# compare calculated solar position to SURFRAD
(DATA['solzen'] / SOLPOS['apparent_zenith'] - 1)['2006-07-16':'2006-07-17'].tz_convert(METADATA['tz'][station_id]).plot()


# In[11]:

# concatenate atmospheric parameters so they're the same size
# pd.concat fills missing timestamps from low frequency datasets with NaN
atm_params = pd.concat([DATA, STATION_ATM, SOLPOS], axis=1)

# then fill in NaN in atmospheric data by padding with the previous value
atm_params['alpha'].fillna(method='pad', inplace=True)
atm_params['aod1240'].fillna(method='pad', inplace=True)
atm_params['aod550'].fillna(method='pad', inplace=True)
atm_params['pwat'].fillna(method='pad', inplace=True)
atm_params['tau380'].fillna(method='pad', inplace=True)
atm_params['tau500'].fillna(method='pad', inplace=True)
atm_params['tau700'].fillna(method='pad', inplace=True)
atm_params['tcwv'].fillna(method='pad', inplace=True)


# In[12]:

# compare measured and calculated precipitable water
# Pad the ECMWF-MACC data since it's at 3-hour intervals, but SURFRAD is at 1-minute or 3-minute intervals.
# Then rollup daily averages to make long term trends easier to see.
pwat = pd.concat([atm_params['pwat'], atm_params['pwat_calc']], axis=1).resample('D').mean()
pwat['2006-01-01':'2006-12-31'].tz_convert(METADATA['tz'][station_id]).plot()
plt.title('Comparison of measured and calculated daily atmospheric water vapor for 2006.')
plt.ylabel('Precipitable Water [cm]')
plt.legend(['Measured', 'Calculated'])


# ### Linke Turbidity
# Linke turbidity is an atmospheric parameter that combines AOD and P<sub>wat</sub>. A historical data set (_c_. 2003) is parameter in the Ineichen clearsky model.

# In[13]:

# lookup Linke turbidity
# use 3-hour intervals to speed up lookup
LT = pvlib.clearsky.lookup_linke_turbidity(
    time=TIMES,
    latitude=METADATA['latitude'][station_id],
    longitude=METADATA['longitude'][station_id],
)

# calculate Linke turbidity using Kasten pyrheliometric formula
LT_CALC = pvlib.atmosphere.kasten96_lt(
    atm_params['amp'], atm_params['pwat'], atm_params['tau700']
)

# calculate broadband AOD using Bird & Hulstrom approximation
AOD_CALC = pvlib.atmosphere.bird_hulstrom80_aod_bb(atm_params['tau380'], atm_params['tau500'])

# recalculate Linke turbidity using Bird & Hulstrom broadband AOD
LT_AOD_CALC = pvlib.atmosphere.kasten96_lt(
    atm_params['amp'], atm_params['pwat'], AOD_CALC
)


# In[14]:

# insert Linke turbidity to atmospheric parameters table
atm_params.insert(25, 'lt', LT)
atm_params.insert(26, 'lt_calc', LT_CALC)
atm_params.insert(27, 'lt_aod_calc', LT_AOD_CALC)
atm_params.insert(28, 'aod_calc', AOD_CALC)

# Linke turbidity should be continuous, it only depends on time
# fill in Linke turbidity by padding previous values
atm_params['lt'].fillna(method='pad', inplace=True)


# In[15]:

# compare Bird & Hulstrom approximated broadband AOD to AOD at 700-nm
# downsample to monthly intervals to show long term trend for entire range of ECMWF-MACC data
aod_bb = atm_params[['tau700', 'aod_calc']].resample('M').mean()
aod_bb.tz_convert(METADATA['tz'][station_id]).plot()
plt.ylabel('broadband AOD')
plt.title('Comparison of calculated and measured broadband aerosol optical depth')


# In[16]:

# compare historic Linke turbidity to calculated
# downsample to monthly averages to show long term trends
atm_params[['lt', 'lt_calc', 'lt_aod_calc']].resample('M').mean().tz_convert(METADATA['tz'][station_id]).plot()
plt.ylabel('Linke turbidity')
plt.title('Comparison of measured and historical Linke turbidity')


# In[17]:

# how is atmosphere changing over time?
# Calculate the linear regression of the relative difference between historical and measured Linke turbidity
lt_diff = (atm_params['lt_calc'] / atm_params['lt']).resample('A').mean() - 1.0  # yearly differences
x = np.arange(lt_diff.size)  # years
x = sm.add_constant(x)  # add y-intercept
y = lt_diff.values  # numpy array of yearly relative differences
results = sm.OLS(y, x).fit()  # fit linear regression
results.summary()  # output summary


# In[18]:

# plot yearly relative difference versus trendline
plt.plot(lt_diff)
plt.plot(lt_diff.index, results.fittedvalues)
plt.title('Yearly Atmospheric Trend')
plt.legend(['yearly relative differnce', 'trendline'])
plt.ylabel('relative difference between calculated and historic Linke turbidity')
plt.xlabel('years')


# ### Clear sky models
# There are several clear sky models, and they take different arguments.

# In[19]:

# calculate clear sky
INEICHEN_LT = pvlib.clearsky.ineichen(
    atm_params['apparent_zenith'], atm_params['amp'], atm_params['lt'], altitude=METADATA['elevation'][station_id],
    dni_extra=atm_params['etr']
)
INEICHEN_CALC = pvlib.clearsky.ineichen(
    atm_params['apparent_zenith'], atm_params['amp'], atm_params['lt_calc'], altitude=METADATA['elevation'][station_id],
    dni_extra=atm_params['etr']
)
INEICHEN_AOD_CALC = pvlib.clearsky.ineichen(
    atm_params['apparent_zenith'], atm_params['amp'], atm_params['lt_aod_calc'], altitude=METADATA['elevation'][station_id],
    dni_extra=atm_params['etr']
)
SOLIS = pvlib.clearsky.simplified_solis(
    atm_params['apparent_elevation'],
    atm_params['tau700'],
    atm_params['pwat'],
    pressure=atm_params['press'],
    dni_extra=atm_params['etr']
)
BIRD = pvlib.clearsky.bird(
    atm_params['apparent_zenith'],
    atm_params['am'],
    atm_params['tau380'],
    atm_params['tau500'],
    atm_params['pwat'],
    pressure=atm_params['press'],
    dni_extra=atm_params['etr']
)


# In[20]:

# plot direct normal irradiance
LEGEND = ['Ineichen-Linke', 'Ineichen-ECMWF-MACC', 'Ineichen-Bird-Hulstrom', 'SOLIS', 'BIRD', 'SURFRAD']
pd.concat([
    INEICHEN_LT['dni'], INEICHEN_CALC['dni'], INEICHEN_AOD_CALC['dni'],
    SOLIS['dni'], BIRD['dni'], atm_params['dni']
], axis=1)['2006-07-13':'2006-07-17'].resample('H').mean().tz_convert(METADATA['tz'][station_id]).plot()
plt.legend(LEGEND, loc=2)
plt.title('Comparison of clear sky irradiance at %s 2003-2012' % METADATA['station name'][station_id])
plt.ylabel('$DNI [W/m^2]$')


# In[21]:

# plot diffuse irradiance
pd.concat([
    INEICHEN_LT['dhi'], INEICHEN_CALC['dhi'], INEICHEN_AOD_CALC['dhi'],
    SOLIS['dhi'], BIRD['dhi'], atm_params['dhi']
], axis=1)['2006-07-13':'2006-07-17'].resample('H').mean().tz_convert(METADATA['tz'][station_id]).plot()
plt.legend(LEGEND, loc=2)
plt.title('Comparison of clear sky irradiance at %s 2003-2012' % METADATA['station name'][station_id])
plt.ylabel('$DHI [W/m^2]$')


# In[22]:

# plot global horizontal irradiance
pd.concat([
    INEICHEN_LT['ghi'], INEICHEN_CALC['ghi'], INEICHEN_AOD_CALC['ghi'],
    SOLIS['ghi'], BIRD['ghi'], atm_params['ghi']
], axis=1)['2006-07-13':'2006-07-17'].resample('H').mean().tz_convert(METADATA['tz'][station_id]).plot()
plt.legend(LEGEND, loc=2)
plt.title('Comparison of clear sky irradiance at %s 2003-2012' % METADATA['station name'][station_id])
plt.ylabel('$GHI [W/m^2]$')


# In[ ]:




# ![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)
# Use of Measured Aerosol Optical Depth and Precipitable Water to Model Clear Sky Irradiance by [Mark A. Mikofski, Clifford W. Hansen, William F. Holmgren and Gregory M. Kimball](https://github.com/mikofski/pvsc44-clearsky-aod) is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
# 
