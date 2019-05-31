import matplotlib.pyplot as plt


def plot_colormesh(ax,val,**kwargs):
    con= ax.pcolormesh(val,**kwargs)
    ax.set_ylim(0,val.shape[0])
    ax.set_xlim(0,val.shape[1])
    return con
