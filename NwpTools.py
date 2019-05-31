import numpy as np
import netCDF4


def get_constants(lat=45/180*np.pi):
    """calculate beta, dx and dy at the given latitude.
    
    Keyword Arguments:
        lat {float} -- latitude in rad (default: {45/180*np.pi})
    
    Returns:
        list(float) -- beta, dx, dy in standard units
    """
    beta=2*np.pi*2/(3600*24)*np.cos(lat)/6370000
    dx=2*np.pi*6370*np.cos(lat)/360*1000#longitude distance
    dy=2*np.pi*6370/360*1000#latitude distance
    return beta, dx, dy


def correlation(x,y):
    """Calculate the correlation of two matrices
    
    Arguments:
        x {2darr} -- Matrix 1
        y {2darr} -- Matrix2
    
    Returns:
        float -- Correlation coefficient between -1 and 1
    """
    x_m=np.mean(x)
    y_m=np.mean(y)
    E=np.sum((x-x_m)*(y-y_m))
    N=np.sqrt(np.sum(np.power((x-x_m),2))*np.sum(np.power((y-y_m),2)))
    return E/N

def derx_central(f,dx):
    """Derivative of the columns with periodic boundary conditions.
    
    Arguments:
        f {2darr} -- the field to derive
        dx {scalar} -- the corresponding step size
    
    Returns:
        2darr -- the derivate of f
    """
    l1,l2=f.shape
    f_p1=f[(np.arange(l1)+1)%l1,:]
    f_m1=f[(np.arange(l1)-1)%l1,:]
    return (f_p1-f_m1)/(2*dx)

def dery_central(f,dy):
    """Derivative of the rows with one sided differences at the boundaries.

    Arguments:
        f {2d array} -- the field to derive
        dy {scalar} -- the corresponding step size

    Returns:
        2d array -- derivative
    """
    l1,l2=f.shape
    f_p1=f[:,(np.arange(l2)+1)%l2]
    f_m1=f[:,(np.arange(l2)-1)%l2]
    f_m1[:,0]=f[:,0]
    f_p1[:,-1]=f[:,-1]
    derx= (f_p1-f_m1)/(2*dy)
    derx[:,0]=(f_p1-f_m1)[:,0]/(dy)
    derx[:,-1]=(f_p1-f_m1)[:,-1]/(dy)
    return derx

def vorticity_central(u,v,dx,dy):
    """Calculate the vorticity with central differences
    
    Arguments:
        u {2darr} -- u component of the wind
        v {2darr} -- v component of the wind
        dx {float} -- the step size in longitude
        dy {float} -- the step size in latitude
    
    Returns:
        2darr -- the vorticity field
    """
    return derx_central(v,dx)-dery_central(u,dy)

def forecast_richardson(u0,v0,dx, dy, dt ,beta=1.61e-11):
    vort0=vorticity_central(u0,v0,dx,dy)
    return -u0*dt*derx_central(vort0,dx)-v0*dt*dery_central(vort0,dy)-v0*dt*beta+vort0

def transformVariable(data, sort_lat, select_lat):
    data=data[0,:,:]#choose first timestep
    data=data.T
    data=data[:,sort_lat]
    data=data[:,select_lat]
    return data

    
def readData(filename):
    """Read netcdf file, extract u,v,vort, choose first timestep, sort by latitude and select northern hemisphere.
    
    Arguments:
        filename {string} -- the netcdf filename
    
    Returns:
        2darr -- lons,lat,u,v,vort in format[lons,lat]
    """
    ncf = netCDF4.Dataset(filename, 'r')
    lons=ncf.variables['longitude'][:]
    lats=ncf.variables['latitude'][:]
    vort=ncf.variables['vo'][:]
    u=ncf.variables['u'][:]
    v=ncf.variables['v'][:]
    sort_lat=np.argsort(lats)
    lats=lats[sort_lat]
    select=lats>=0.
    lats=lats[select]
    u=transformVariable(u,sort_lat,select)
    v=transformVariable(v,sort_lat,select)
    vort=transformVariable(vort,sort_lat,select)
    return lons, lats, u, v, vort
