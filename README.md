# pvsc44-clearsky-aod
The [Jupyter](http://jupyter.org/) notebooks and [Python](http://python.org/) scripts in this repository contain the analysis for
**Use of Measured Aerosol Optical Depth and Precipitable Water to Model Clear Sky Irradiance** presented at the IEEE 44th Photovoltaic
Specialists Conference (PVSC) in Washington, D.C. in June, 2017. The calculations and figures from the paper and presentation were
generated here.

# Quick start
Probably the fastest way to use these notebooks interactively is to open them in
[mybinder](https://mybinder.org/v2/gh/mikofski/pvsc44-clearsky-aod/master). You can also view them statically online at GitHub, or if
you have [Git](https://git-scm.com/) and [the requirements](#requirements) then you can [clone](https://git-scm.com/docs/git-clone) this
repository to your computer, and open it there.

## Requirements
To use these notebooks and scripts you will need Python-2. If you would like to see them in Python-3, start an [issue](../../issues),
and if enough users request it, I can look into making them compatible for Python-3. If you don't have Python, probably the easiest
way to install Python is to get [Anaconda](https://www.anaconda.com/download/).

### Dependencies
These notebooks depend on several Python packages that are either already installed in or available from Anaconda:
* [pvlib](http://pvlib-python.readthedocs.io/en/latest/)
* [numpy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [statsmodels](http://www.statsmodels.org/stable/index.html)
* [h5py](http://www.h5py.org/)
* [scipy](https://www.scipy.org/)

#### PVLIB
PVLIB is available from the [Anaconda PVLIB channel](https://anaconda.org/pvlib/pvlib).

# [PVSC44 Clearsky AOD Analysis](./PVSC44%20Clearsky%20AOD%20Analysis.ipynb)
This is the first notebook. It contains an introduction and a breif of how to use it. Then it begins by parsing the SURFRAD data,
and making some plots of surface temperature and irradiance data as a sanity check. Then it parses data for aerosol optical depth
and water vapor from ECMWF, and provides some links describing how users can obtain that data. The rest of the notebook uses PVLIB
to get the solarposition and clearsky irradiance predictions using various models. Then the notebook generates graphs using seaborn.

# [PVSC44 Clearsky SURFRAD Analysis](./PVSC44%20Clearsky%20SURFRAD%20Analysis.ipynb)
This is the second notebook. It continues from where the first notebook,
[PVSC44 Clearsky AOD Analysis](./PVSC44%20Clearsky%20AOD%20Analysis.ipynb), leaves off. It starts by stating explicitly the
methods for calculating MBE and RMSE and then analyzes and plots the MBE and RMSE of the various clearsky models versus the
measured SURFRAD data.

# [PVSC44 TL sensitivity](https://github.com/mikofski/pvsc44-clearsky-aod/blob/master/PVSC44%20TL%20sensitivity.ipynb)
This is the third notebook. In it Gregory Kimball looks at the sensitivity to Linke Turbidity.

# [PVSC44 ECMWF AOD Sensitivity](https://github.com/mikofski/pvsc44-clearsky-aod/blob/master/PVSC44%20ECMWF%20AOD%20Sensitivity.ipynb)
This is the fourth and final notebook. In it we use Greg's analysis of TL sensitivity, and make the unlikely discovery that the TL and ECMWF MACC AOD data agree when we impose a low light filter:

>So I did that, but in the process I made a funny discovery - the MACC AOD only results in higher $T_L$ if you include low irradiance. But, if you filter out low light conditions, then the MACC AOD calculated $T_L$ actually matches the historical values well.

# License
All content on this site is covered by a [3-clause BSD license](./LICENSE) and a
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
