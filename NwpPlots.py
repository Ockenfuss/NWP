import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def append_colorbar(fig,ax,con):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(con, cax=cax)


def plot_colormesh(ax,val,fig=None, **kwargs):
    con= ax.pcolormesh(val,**kwargs)
    ax.set_ylim(0,val.shape[0])
    ax.set_xlim(0,val.shape[1])
    if fig is not None:
        append_colorbar(fig,ax,con)
    return con

def plot_ft(ax,ft,fig=None, **kwargs):
    ft_intern=np.fft.fftshift(ft)
    ft_intern=np.abs(ft_intern)
    con=plot_colormesh(ax,ft_intern, fig=fig)
    return con
