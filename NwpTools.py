import numpy as np
import netCDF4



###Basic numerics
def correlation(x,y):
    """Calculate the correlation of two matrices.
    Every matrix is interpreted as a collection of samples from one stochastic variable.
    
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
def laplace_central(f,dx,dy):
    """Calculate dx^2f+dy^2f.
    
    Arguments:
        f {2darr} -- A scalar field.
        dx {float} -- The stepsize in longitudinal direction
        dy {float} -- The stepsize in latitudinal direction
    
    Returns:
        2darr -- laplace(f)
    """
    return derx_central(derx_central(f,dx),dx)+dery_central(dery_central(f,dy),dy)
def invert_laplace(xi, dx, dy, extend=True):
    """Invert the equation laplace(psi)=xi

    Arguments:
        xi {2darr} -- the laplacian of the field in need
        dx {float} -- stepsize in longitude
        dy {float} -- stepsize in latitude

    Keyword Arguments:
        extend {bool} -- extend xi to xi' like described on sheet 5 (default: {True})

    Returns:
        2darr -- psi, the fourier transform of the (extended) field
    """
    xi_ext = xi
    if extend:
        xi_ext = extend_vorticity(xi)
    im_lapl_f = np.fft.fft2(xi_ext)
    freq_x = np.fft.fftfreq(xi_ext.shape[0], dx / (2 * np.pi))
    freq_y = np.fft.fftfreq(xi_ext.shape[1], dy / (2 * np.pi))
    kx, ky = np.meshgrid(freq_x, freq_y, indexing='ij')
    psi0_f = -1 * im_lapl_f / (kx * kx + ky * ky)
    # the 0. coefficient is not defined by the equation! (Because we are
    # loosing the constant offset when deriving.)
    psi0_f[0, 0] = 10.0
    psi0 = np.real(np.fft.ifft2(psi0_f))
    if extend:
        psi0 = psi0[:xi.shape[0], :xi.shape[1]]
    return psi0
def extend_vorticity(vort):
    """Extend the vorticity field like described on Sheet 5

    Arguments:
        vort {2darr} -- a scalar field

    Returns:
        2darr -- the field where every row is extended like described on Sheet 5
    """
    vort_int = vort.copy()
    vort_ext = -1 * vort[:, -2:0:-1]
    vort_int[:, [0, -1]] = 0
    return np.concatenate((vort_int, vort_ext), axis=1)

###numerical weather prediction
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
def get_wind(psi,dx,dy):
    """Derive wind vector from psi field
    
    Arguments:
        psi {2darr} -- The scalar field to derive the wind vector from [lons,lat]
        dx {float} -- the stepsize in zonal direction
        dy {float} -- the stepsize in meridional direction
    
    Returns:
        list(2darr) -- components of the wind vector u, v
    """
    u = -1 * dery_central(psi, dy)
    v = derx_central(psi, dx)
    return u,v


def forecast_richardson(u0,v0,dx, dy, dt ,beta=1.61e-11):
    vort0=vorticity_central(u0,v0,dx,dy)
    return -u0*dt*derx_central(vort0,dx)-v0*dt*dery_central(vort0,dy)-v0*dt*beta+vort0

###Model setup
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
