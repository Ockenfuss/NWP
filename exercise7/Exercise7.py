import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os as os
import numpy as np
import matplotlib.pyplot as plt
from MyPython import Input as Inp
import sys as sys
from scipy.ndimage import filters as filt
import argparse
from matplotlib.colors import LogNorm#falls mit LogNorm
import matplotlib.colors as colors

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

VERSION="1.0"
par=argparse.ArgumentParser()
par.add_argument('infile')
par.add_argument('-s',action='store_true')
args=par.parse_args()
inp=Inp.Input(args.infile,version=VERSION)
inp.convert_type(float, "dt")
inp.convert_type(float,"total_time_days")
inp.convert_type(float,"total_time_hours")
inp.convert_type(int,"delay")
inp.convert_type(int,"diffusion")
inp.convert_type(int,"number_pictures")
inp.convert_type(bool,"simulation")
inp.show_data()

def produce_vort_movie(vort0,u0,v0,dx,dy,dt,beta,D,total_time, number_pictures, print_info=True):
    """Do a forecast of the vorticity field based on the barotropic forecast equation and return snapshots in equidistant time intervals.
    
    Arguments:
        vort0 {2darr} -- vorticity field at time 0
        u0 {2darr} -- zonal wind field at time 0
        v0 {2darr} -- meridional wind field at time 0
        dx {scalar} -- the gridspacing in longitudinal direction
        dy {scalar} -- the gridspacing in latitudinal direction
        dt {scalar} -- the timestep in s
        beta {scalar} -- the beta parameter (linear term of the expansion of the coriolis force)
        D {scalar} -- diffusion coefficient
        total_time {float} -- the total simulation time in seconds
        number_pictures {int} -- the number of snapshots to be returned.
    
    Keyword Arguments:
        print_info {bool} -- Print information on the actual simulated time (could differ from total_time due to matching with the number of pictures.) (default: {True})
    
    Returns:
        3darr -- snapshots of the vorticity field in format [pic, lon, lat]
    """
    #Initialize: Make a single forward in time step
    mean_u0=np.mean(u0)
    vort1=nwp.barotropic_forecast_equation(vort0,vort0,u0,v0,dx,dy,dt/2,beta,D)#for dt=6h and D=0, this should be the result from sheet 4!
    psi1=nwp.invert_laplace(vort1,dx,dy)
    u1,v1=nwp.get_wind(psi1,dx,dy)
    u1=u1+mean_u0

    ###Produce movie
    total_steps=int(np.floor(total_time/dt))
    steps_per_picture=int(np.floor(total_steps/number_pictures))
    pictures=np.zeros((number_pictures,)+vort0.shape)#format[pic,lon,lat]

    if print_info:
        print("timestep [s]: "+str(dt))
        print("total steps: "+str(total_steps))
        print("number pictures: "+str(number_pictures))
        print("steps per picture: "+str(steps_per_picture))
        print("total simulated time [s]: "+str(dt*number_pictures*steps_per_picture))
        print("Originally desired time [s]:" +str(total_time) + "(this is "+str(total_time-dt*number_pictures*steps_per_picture)+" s differnce)")

    vort_n=vort0*1.0
    vort_np1=vort1*1.0
    u_np1=u1*1.0
    v_np1=v1*1.0
    for i in range(number_pictures):
        vort_np1, vort_n=nwp.run_barotropic_model(vort_np1,vort_n,u_np1,v_np1,dx,dy,dt,beta,D,steps=steps_per_picture, return_before=True)
        psi_np1=nwp.invert_laplace(vort_np1,dx,dy)
        u_np1,v_np1=nwp.get_wind(psi_np1,dx,dy)
        u_np1=u_np1+mean_u0
        pictures[i]=vort_n
        if print_info:
            print("calculated picture "+str(i))
    return pictures

def create_animation(fig, ax, pictures, interval):
    ax.set_xlabel("longitutde")
    ax.set_ylabel("latitude")
    ims=[]
    v_max=np.max(pictures)
    v_min=np.min(pictures)
    print(v_min)
    for i in range(pictures.shape[0]):
        title = plt.text(0.5,1.01,"Picture "+str(i), ha="center",va="bottom", transform=ax.transAxes, fontsize="large")
        im=plots.plot_colormesh(ax,pictures[i].T, norm=colors.SymLogNorm(linthresh=0.00003, linscale=1.0, vmin=v_min, vmax=v_max))
        ims.append([im,title])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=400)
    return ani

def create_1d_animation(fig, ax, x, values, interval):
    #format [frame, position]
    line, = ax.plot([], [], lw=1)
    def animate(i):
        a = x[i]
        b = values[i]
        line.set_data(a, b)
        return line,
    print(x.shape[0])
    anim = FuncAnimation(fig, animate, frames=x.shape[0], interval=interval, blit=True)
    return anim


def energy_spectrum(vort, mean_u, dx, dy):
    psi=nwp.invert_laplace(vort, dx, dy)
    u,v=nwp.get_wind(psi, dx, dy)
    u=u+mean_u
    energy=u*u+v*v#format [lon, lat]
    energy_f=np.fft.fft(energy, axis=0)
    return np.mean(np.abs(energy_f), axis=1), np.fft.fftfreq(vort.shape[0], d=dx/(2*np.pi))

def main():
    lons, lat, u0, v0, vort_real=nwp.readData("../Data/eraint_2019020100.nc")
    lons, lat, u_24h, v_24h, vort_real_24h=nwp.readData("../Data/eraint_2019020200.nc")

    beta, dx,dy=nwp.get_constants()
    vort0=nwp.vorticity_central(u0,v0,dx,dy)
    vort_24h=nwp.vorticity_central(u_24h,v_24h,dx,dy)
    #Create perturbed initial fields
    u0_pert=u0+np.mean(u0)*0.1
    vort0_pert=vort0+0.1*vort_24h

    dt=inp.get("dt")
    D=inp.get("diffusion")
    number_pictures=inp.get("number_pictures")
    total_time=inp.get("total_time_days")*24*3600+inp.get("total_time_hours")*3600

    pictures_truth=np.zeros((number_pictures,)+vort0.shape)
    pictures_pert=np.zeros((number_pictures,)+vort0.shape)
    truth_path="Data/"+inp.get("savename")+"_truth.npy"
    pert_path="Data/"+inp.get("savename")+"_pert.npy"
    if inp.get("simulation"):
        pictures_truth=produce_vort_movie(vort0,u0,v0,dx,dy,dt, beta, D, total_time, number_pictures)
        pictures_pert=produce_vort_movie(vort0_pert,u0_pert,v0,dx,dy,dt, beta, D, total_time, number_pictures)
        np.save(truth_path,pictures_truth)
        np.save(pert_path,pictures_pert)
        inp.write_log([truth_path, pert_path],file_ext=".log")
    else:
        pictures_truth=np.load(truth_path)
        pictures_pert=np.load(pert_path)



    ###RMSE plot
    if inp.get("plot")=="a":
        fig, ax=plt.subplots()
        select=np.logical_and(lat>30,lat<60)
        rmse=[nwp.rmse(pictures_truth[i,:,select], pictures_pert[i,:,select]) for i in range(number_pictures)]
        ax.plot(rmse)


    ###Animation
    if inp.get("plot")=="b":
        fig_ani, ax_ani=plt.subplots()
        ani_truth=create_animation(fig_ani, ax_ani, pictures_truth-pictures_pert, 1000)

    ###Energy spectrum: in our case, we just transform the vorticity field!
    if inp.get("plot")=="c":
        select=np.logical_and(lat>30,lat<60)
        pictures_truth=pictures_truth[:,:,select]
        pictures_pert=pictures_pert[:,:,select]
        spectrum=np.zeros((number_pictures, vort0.shape[0]))
        frequencies=np.zeros((number_pictures, vort0.shape[0]))
        for i in range(number_pictures):
            # spectrum[i], frequencies[i]=energy_spectrum(pictures_truth[i], np.mean(u0), dx, dy)
            vort_f=np.fft.fft(pictures_truth[i]-pictures_pert[i], axis=0)
            spectrum[i]=np.mean(np.abs(vort_f), axis=1)
            frequencies[i]= np.fft.fftfreq(len(lons), d=dx/(2*np.pi))

        fig_spec, ax_spec=plt.subplots()
        ax_spec.set_xlim(np.min(frequencies), np.max(frequencies))
        ax_spec.set_ylim(np.min(spectrum), np.max(spectrum))
        ani_spec=create_1d_animation(fig_spec, ax_spec, frequencies, spectrum, interval=200)



main()

# u=np.linspace(0,10,10)
# v=np.zeros(10)
# x=np.linspace(-10,10,1000)
# energy=np.exp(-1*np.power(x/0.3,2))
# plt.plot(x,energy)
# energy_f=np.fft.fft(energy)
# # plt.plot(x,np.abs(np.fft.fftshift(energy_f)))
# plt.plot(np.fft.fftfreq(len(x)),(np.abs(energy_f)))
# # plt.plot(x,np.abs(energy_f))


plt.show()