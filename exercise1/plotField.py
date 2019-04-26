#!/usr/bin/env python
from numpy import *
import netCDF4
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
# from mpl_toolkits.basemap import Basemap


#Read reanalysis data
filename='eraint_2019010100.nc'
ncf = netCDF4.Dataset(filename, 'r')
#prints the content
print(ncf)
lons=ncf.variables['longitude'][:]
lats=ncf.variables['latitude'][:]
u=ncf.variables['u'][0,:]
ncf.close()

