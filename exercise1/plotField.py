#!/usr/bin/env python
import numpy as np
import netCDF4
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
# from mpl_toolkits.basemap import Basemap

def makeCountour(x,y,val, ax):
    ax.contourf(data)

def makeCurve(data,ax):
    ax.plot(data)

#Read reanalysis data
filename='eraint_2019010100.nc'
ncf = netCDF4.Dataset(filename, 'r')
#prints the content
print(ncf)
lons=ncf.variables['longitude'][:]
lats=ncf.variables['latitude'][:]
u=ncf.variables['u'][0,:]
# print(ncf.groups)
# print(ncf.dimensions)
print(ncf.variables.keys())
print(ncf.dimensions.keys())
u=ncf.variables['u'][0,:,:]
v=ncf.variables['v'][0,:,:]
vo=ncf.variables['vo'][0,:,:]
lat=ncf.variables['latitude'][:]
lon=ncf.variables['longitude'][:]
print(lat[:])
# print(u)
# print(u.dimensions)#time, lat, lon
# print(u.units)
# print(u.shape)

zonalU=np.mean(u, axis=1)
zonalV=np.mean(v, axis=1)
zonalVo=np.mean(vo, axis=1)
fig, ax=plt.subplots(nrows=3, ncols=3)
xgrid, ygrid=np.meshgrid(lon,lat)
makeContour(u[:,:],ax[0,0])
# makeCountour(v[:,:],ax[1,0])
# makeCountour(vo[:,:],ax[2,0])
# makeCurve(zonalU, ax[0,1])
# makeCurve(zonalV, ax[1,1])
# makeCurve(zonalVo, ax[2,1])

#Rearrange data
u2=u.T#long, lat
u2=u2[:,90:]
print(u2.shape)
plt.show()




ncf.close()

