from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from MyPython import Input as Inp
import sys as sys
from scipy.ndimage import filters as filt


def load_src(name, fpath):
    import os
    import imp
    p = fpath if os.path.isabs(fpath) \
        else os.path.join(os.path.dirname(__file__), fpath)
    return imp.load_source(name, p)


load_src("NwpTools", "../NwpTools.py")
import NwpTools as nwp
load_src("NwpPlots", "../NwpPlots.py")
import NwpPlots as plots



lons, lat, u0, v0, vort_real0 = nwp.readData("../Data/eraint_2019020100.nc")
beta, dx, dy = nwp.get_constants()


def Exercise_b():
    vort_calc0 = nwp.vorticity_central(u0, v0, dx, dy)

    #test#
    # sin=np.sin(np.linspace(0,2*np.pi,300))
    # vort_calc0=np.meshgrid(np.linspace(0,20,3),sin, indexing='ij')[1]

    vort_calc0_f = np.fft.fft2(vort_calc0)
    vort_calc0_1 = np.real(np.fft.ifft2(vort_calc0_f))
    psi0_1, psi0_f = nwp.invert_laplace(vort_calc0, dx, dy)
    vort_calc0_2 = nwp.laplace_central(psi0_1, dx, dy)

    fig, ax = plt.subplots(3, 2, figsize=(14,10))
    plots.plot_colormesh(ax[0, 0], vort_calc0, fig=fig)
    ax[0, 0].set_title("Original field")
    plots.plot_colormesh(ax[0, 1], vort_calc0_2[1:-1, 1:-1], fig=fig)
    ax[0, 1].set_title("Laplace of calculated psi")
    plots.plot_ft(ax[1, 0], vort_calc0_f, fig=fig)
    ax[1, 0].set_title("Fourier transform of original field")
    plots.plot_ft(ax[1, 1], psi0_f, fig=fig)
    ax[1, 1].set_title("Fourier transform of Psi")
    plots.plot_colormesh(ax[2, 0], vort_calc0_1, fig=fig)
    ax[2, 0].set_title("Original field transformed and back")
    con = plots.plot_colormesh(ax[2, 1], psi0_1, fig=fig)
    ax[2, 1].set_title("Calculated Psi")
    fig.savefig("b_invert_laplacian.jpg")


def Exercise_c_test():
    vort_calc0 = np.meshgrid(
        np.linspace(
            0, 10, 3), np.linspace(
            0, 1, 100), indexing='ij')[1]

    vort_calc0_ext = nwp.extend_vorticity(vort_calc0)
    psi0_ext, psi0_f_ext = nwp.invert_laplace(vort_calc0_ext, dx, dy, extend=False)
    psi0 = nwp.invert_laplace(vort_calc0, dx, dy)[0]
    vort_calc1 = nwp.laplace_central(psi0, dx, dy)

    fig, ax = plt.subplots(3, 2, figsize=(14,10))
    ax[0, 0].plot(vort_calc0[0, :])
    ax[0, 0].set_title("Original field")
    ax[1, 0].plot(vort_calc0_ext[0, :])
    ax[1, 0].set_title("Extended field")
    plots.plot_ft(ax[2, 0], psi0_f_ext, fig=fig)
    ax[2, 0].set_title("Fourier Transform of the psi field")
    ax[0, 1].plot(psi0_ext[0, :])
    ax[0, 1].set_title("Extended Psi field")
    ax[1, 1].plot(psi0[0, :])
    ax[1, 1].set_title("Normal Psi field")
    ax[2, 1].plot(vort_calc1[0, :])
    ax[2, 1].set_title("Laplace of Psi for consistency")
    fig.savefig("c_extension1d.jpg")

def Exercise_c_real():
    vort_calc0 = nwp.vorticity_central(u0, v0, dx, dy)
    vort_calc0_ext = nwp.extend_vorticity(vort_calc0)
    psi0_ext, psi0_f_ext = nwp.invert_laplace(vort_calc0_ext, dx, dy, extend=False)
    psi0 = nwp.invert_laplace(vort_calc0, dx, dy)[0]
    vort_calc1 = nwp.laplace_central(psi0, dx, dy)

    fig, ax = plt.subplots(3, 2, figsize=(14,10))
    plots.plot_colormesh(ax[0, 0], vort_calc0, fig=fig)
    ax[0, 0].set_title("Original field")
    plots.plot_colormesh(ax[1, 0], vort_calc0_ext, fig=fig)
    ax[1, 0].set_title("Extended field")
    plots.plot_ft(ax[2, 0], psi0_f_ext, fig=fig)
    ax[2, 0].set_title("Fourier Transform of the psi field")
    plots.plot_colormesh(ax[0, 1], psi0_ext, fig=fig)
    ax[0, 1].set_title("Extended Psi field")
    plots.plot_colormesh(ax[1, 1], psi0, fig=fig)
    ax[1, 1].set_title("Normal Psi field")
    plots.plot_colormesh(ax[2, 1], vort_calc1, fig=fig)
    ax[2, 1].set_title("Laplace of Psi for consistency")
    fig.savefig("c2_extension2d.jpg")

def Exercise_d():
    vort_calc0 = nwp.vorticity_central(u0, v0, dx, dy)
    psi0 = nwp.invert_laplace(vort_calc0, dx, dy)[0]

    u1 = -1 * nwp.dery_central(psi0, dy)
    u1 = u1 - np.mean(u1) + np.mean(u0)
    v1 = nwp.derx_central(psi0, dx)

    fig, ax = plt.subplots(3, 2, figsize=(14,10))
    plots.plot_colormesh(ax[0, 0], u0, fig=fig)
    ax[0,0].set_title("Original u")
    plots.plot_colormesh(ax[1, 0], v0, fig=fig)
    ax[1,0].set_title("Original v")
    plots.plot_colormesh(ax[0, 1], u1, fig=fig)
    ax[0,1].set_title("Rederived u")
    plots.plot_colormesh(ax[1, 1], v1, fig=fig)
    ax[1,1].set_title("Rederived v")
    ax[2, 0].plot(np.mean(psi0, axis=0))
    ax[2, 1].plot(np.mean(u1, axis=0))
    print(np.mean(u0))
    print(np.mean(u1))
    print(np.mean(v0))
    print(np.mean(v1))
    select = np.logical_and(lat > 30, lat < 60)
    print("Correlations:")
    print(nwp.correlation(u0[:,select], u1[:,select]))#remember: correlation independent of means of fields!
    print(nwp.correlation(v0[:,select], v1[:,select]))
    fig.savefig("d_wind_comparison.jpg")

Exercise_b()
Exercise_c_test()
Exercise_c_real()
Exercise_d()
# plt.show()
