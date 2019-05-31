import numpy as np
import matplotlib.pyplot as plt
from  MyPython import Input as Inp
import sys as sys
from scipy import ndimage


def load_src(name, fpath):
    import os, imp
    p = fpath if os.path.isabs(fpath) \
        else os.path.join(os.path.dirname(__file__), fpath)
    return imp.load_source(name, p)
 
load_src("NwpTools", "/project/meteo/work/Paul.Ockenfuss/NWP/NwpTools.py")
import NwpTools as nwp
load_src("NwpPlots", "/project/meteo/work/Paul.Ockenfuss/NWP/NwpPlots.py")
import NwpPlots as plots


lons,lat, u0,v0,vort_real0=nwp.readData("/project/meteo/work/Paul.Ockenfuss/NWP/Data/eraint_2019020100.nc")
beta, dx, dy=nwp.get_constants()
vort_calc0=nwp.vorticity_central(u0,v0,dx,dy)

vort_calc0_f=np.fft.fft2(vort_calc0)
vort_calc0_1=np.abs(np.fft.ifft2(vort_calc0_f))
freq_x=np.fft.fftfreq(vort_calc0.shape[0],dx/(2*np.pi))
freq_y=np.fft.fftfreq(vort_calc0.shape[1],dy/(2*np.pi))
print(freq_x)
kx,ky=np.meshgrid(freq_x, freq_y,indexing='ij')
psi0_f=-1*vort_calc0_f/(kx*kx+ky*ky)
psi0_f[0,0]=0.0 #the 0. coefficient is not defined by the equation!
psi0_1=np.abs(np.fft.ifft(psi0_f))
vort_calc0_2=ndimage.laplace(psi0_1)

def plot_ft(ax,ft,**kwargs):
    ft_intern=np.fft.fftshift(ft)
    ft_intern=np.abs(ft_intern)
    plots.plot_colormesh(ax,ft_intern)

fig, ax=plt.subplots(3,2)
plots.plot_colormesh(ax[0,0],vort_calc0)
plots.plot_colormesh(ax[0,1],vort_calc0_2)
plot_ft(ax[1,0],vort_calc0_f)
plot_ft(ax[1,1],psi0_f)
plots.plot_colormesh(ax[2,0],vort_calc0_1)
plots.plot_colormesh(ax[2,1],psi0_1)

plt.show()
