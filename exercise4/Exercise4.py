import numpy as np
import netCDF4
# import matplotlib
# matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats

def correlation(x,y):
    x_m=np.mean(x)
    y_m=np.mean(y)
    E=np.sum((x-x_m)*(y-y_m))
    N=np.sqrt(np.sum(np.power((x-x_m),2))*np.sum(np.power((y-y_m),2)))
    return E/N

def derx_central(f,dx):
    l1,l2=f.shape
    f_p1=f[(np.arange(l1)+1)%l1,:]
    f_m1=f[(np.arange(l1)-1)%l1,:]
    return (f_p1-f_m1)/(2*dx)

def dery_central(f,dy):
    """Derivative of the rows with one sided differences.

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

def plot_contour(ax,val,levels=10, **kwargs):
    lvl=np.linspace(np.min(val), np.max(val),levels)
    return ax.contour(val, levels=lvl,**kwargs)

def plot_colormesh(ax,val,**kwargs):
    con= ax.pcolormesh(val,**kwargs)
    ax.set_ylim(0,val.shape[0])
    ax.set_xlim(0,val.shape[1])
    return con

def PlotExercise1():
    filename='eraint_2019020100.nc'
    lons, lats, u, v, vort_real=readData(filename)
    vort_calc=vorticity_central(u,v,dx,dy)*1e6


    fig, ax=plt.subplots(2,2)
    plot_contour(ax[0,0],u,20)
    plot_colormesh(ax[0,1],v)
    plot_colormesh(ax[1,0],vort_real)
    plot_colormesh(ax[1,1],vort_calc)

def PlotExercise2():
    filename0='eraint_2019020100.nc'
    filename1='eraint_2019020106.nc'
    dt=6*3600
    lons, lats, u0, v0, vort_real0=readData(filename0)
    lons, lats, u1, v1, vort_real1=readData(filename1)
    vort_calc0=vorticity_central(u0,v0,dx,dy)
    vort_calc1=vorticity_central(u1,v1,dx,dy)
    vort_fore=forecast_richardson(u0,v0,dx,dy,dt)
    


    fig, ax=plt.subplots(2,2)
    fields=np.array([vort_real0, vort_calc0, vort_real1, vort_fore])
    vmin=np.amin(fields)
    vmax=np.amax(fields)
    for i, axe in enumerate(ax.flatten()):
        contour=plot_colormesh(axe,fields[i], vmin=vmin, vmax=vmax)
        print(fields.shape)
        divider = make_axes_locatable(axe)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(contour, cax=cax)

    #choose area between 30 and 60N
    inds=np.logical_and(lats>30,lats<60)
    print("Correlation of forecast with analysis:")
    print(correlation(vort_fore[:,inds],vort_calc1[:,inds]))
    print("Correlation of constant forecast:")
    print(correlation(vort_calc0[:,inds], vort_calc1[:,inds]))


beta=2*np.pi*2/(3600*24)*np.cos(45/180.*np.pi)/6370000
dx=2*np.pi*6370*np.cos(45/180*np.pi)/360*1000#longitude distance at 45lat
dy=2*np.pi*6370/360*1000#latitude distance

print(beta)
filename='eraint_2019020100.nc'
lons, lats, u, v, vort_real=readData(filename)
vort_calc=vorticity_central(u,v,dx,dy)*1e6

u=u[0:5,0:5]
v=v[0:5,0:5]
vort_real=vort_real[0:5,0:5]
vort_calc=vorticity_central(u,v,dx,dy)*1e6
vort_calc1=forecast_richardson(u,v,dx,dy,3600*6)*1e6
print(u.round(2))
print(v.round(2))
print(vort_calc.round(2))
print((vort_real*1e6).round(2))
print(vort_calc1.round(2))

#Test##
u_test=np.array([np.arange(10)**2 for i in range(10)]).T
der_test=derx_central(u_test,1)
# print(u_test)
# print(der_test)

# PlotExercise1()
PlotExercise2()
plt.show()
