# -*- coding: utf-8 -*-
"""
Atmosphere
----------
The atmosphere class loads ECMWF aerosol optical depth and water vapor data
from netCDF4 files. The path to the folder containing the files is the only
argument, but the files must be named as follows:

    * aod550_2011_macc.nc - AOD at 550[nm] for the globe in 2011
    * aod1240_2011_macc.nc - AOD at 1240[nm] for the globe in 2011
    * tcwv_2011_macc.nc - water vapor for the globe in 2011
    * aod550_2012_macc.nc - AOD at 550[nm] for the globe in 2012
    * aod1240_2012_macc.nc - AOD at 1240[nm] for the globe in 2012
    * tcwv_2012_macc.nc - water vapor for the globe in 2012

The atmosphere class, once initialized, can be used to supply AOD and water
vapor (pwat) at any latitude, longitude coordinates at any time of the year by
calling it's ``get_data(lat, lon, utc_time)` method.

SunPower (c) 2017
"""

import os
import logging

from matplotlib import pyplot as plt
import netCDF4
from pvlib.atmosphere import angstrom_aod_at_lambda, angstrom_alpha

plt.ion()
logging.basicConfig()

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ECMWF_PATH = os.path.join(DIRNAME, 'ecmwf-macc')
ECMWF_RNG = (2003, 2012)


class Atmosphere(object):
    """
    Atmospheric data from ECMWF MACC Reanalysis.
    """
    dlat, dlon, dtime = -0.75, 0.75, 3

    def __init__(self, ecmwf_path=ECMWF_PATH):
        self.aod550 = {}
        self.aod1240 = {}
        self.tcwv = {}
        for year in xrange(ECMWF_RNG[0], ECMWF_RNG[1] + 1):
            self.aod550[year] = netCDF4.Dataset(
                os.path.join(ecmwf_path, 'aod550_%d_macc.nc' % year)
            )
            self.aod1240[year] = netCDF4.Dataset(
                os.path.join(ecmwf_path, 'aod1240_%d_macc.nc' % year)
            )
            self.tcwv[year] = netCDF4.Dataset(
                os.path.join(ecmwf_path, 'tcwv_%d_macc.nc' % year)
            )

    @classmethod
    def interp_data(cls, lat, lon, utc_time, data, key):
        """
        Interpolate data using nearest neighbor.
        """
        nctime = data['time']  # time
        ilat = int(round((lat - 90.0) / cls.dlat))  # index of nearest latitude
        # avoid out of bounds latitudes
        if ilat < 0:
            ilat = 0  # if lat == 90, north pole
        elif ilat > 241:
            ilat = 241  # if lat == -90, south pole
        lon = lon % 360.0  # adjust longitude from -180/180 to 0/360
        ilon = int(round(lon / cls.dlon)) % 480  # index of nearest longitude
        # time index before
        before = netCDF4.date2index(utc_time, nctime, select='before')
        fbefore = data[key][before, ilat, ilon]
        fafter = data[key][before + 1, ilat, ilon]
        dt_num = netCDF4.date2num(utc_time, nctime.units)
        time_ratio = (dt_num - nctime[before]) / cls.dtime
        return fbefore + (fafter - fbefore) * time_ratio

    @staticmethod
    def to_wvl(wvl, aod550, alpha=1.14):
        """
        Convert AOD from tau550 to specified wavelength.
        """
        # aod = aod550 * ((wvl / 550.0) ** (-alpha))
        return angstrom_aod_at_lambda(aod550, 550.0, alpha, wvl)

    def get_data(self, lat, lon, utc_time):
        keys = ['aod550', 'aod1240', 'tcwv']
        data = dict.fromkeys(keys)
        for k in keys:
            data[k] = self.interp_data(
                lat, lon, utc_time, data=getattr(self, k)[utc_time.year], key=k
            )
        data['pwat'] = data['tcwv'] / 10.0  # convert  kg / m^2 to cm
        data['alpha'] = angstrom_alpha(data['aod1240'], 1240.0,
                                       data['aod550'], 550.0)
        #     -np.log(data['aod1240'] / data['aod550']) / np.log(1240.0 / 550.0)
        data['tau380'] = self.to_wvl(380.0, data['aod550'], data['alpha'])
        data['tau500'] = self.to_wvl(500.0, data['aod550'], data['alpha'])
        data['tau700'] = self.to_wvl(700.0, data['aod550'], data['alpha'])
        return data

