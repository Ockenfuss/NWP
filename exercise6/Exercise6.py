from mpl_toolkits.axes_grid1 import make_axes_locatable
import os as os
import numpy as np
import matplotlib.pyplot as plt
from MyPython import Input as Inp
import sys as sys
from scipy.ndimage import filters as filt
import argparse

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

VERSION="1.3"
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
inp.convert_type(bool,"video")
inp.show_data()

def main():
    lons, lat, u0, v0, vort_real=nwp.readData("../Data/eraint_2019020100.nc")
    lons, lat, u_6h, v_6h, vort_real_6h=nwp.readData("../Data/eraint_2019020106.nc")
    lons, lat, u_24h, v_24h, vort_real_24h=nwp.readData("../Data/eraint_2019020200.nc")
    
    beta, dx, dy=nwp.get_constants()
    vort0=nwp.vorticity_central(u0,v0,dx,dy)
    vort_calc_6h=nwp.vorticity_central(u_6h,v_6h,dx,dy)
    vort_calc_24h=nwp.vorticity_central(u_24h,v_24h,dx,dy)
    mean_u0=np.mean(u0)
    dt=inp.get("dt")
    D=inp.get("diffusion")
    number_pictures=inp.get("number_pictures")
    total_time=inp.get("total_time_days")*24*3600+inp.get("total_time_hours")*3600

    #Initialize: Make a single forward in time step
    vort1=nwp.barotropic_forecast_equation(vort0,vort0,u0,v0,dx,dy,dt/2,beta,D)#for dt=6h and D=0, this should be the result from sheet 4!
    psi1=nwp.invert_laplace(vort1,dx,dy)
    u1,v1=nwp.get_wind(psi1,dx,dy)
    u1=u1+mean_u0

    ###Produce movie
    total_steps=int(np.floor(total_time/dt))
    steps_per_picture=int(np.floor(total_steps/number_pictures))
    pictures=np.zeros((number_pictures,)+vort0.shape)#format[pic,lon,lat]

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
        print("calculated picture "+str(i))


    ###Animation
    import matplotlib.animation as animation
    fig_ani, ax_ani=plt.subplots()
    ax_ani.set_xlabel("longitutde")
    ax_ani.set_ylabel("latitude")
    ims=[]
    for i in range(number_pictures):
        title = plt.text(0.5,1.01,"Picture "+str(i), ha="center",va="bottom", transform=ax_ani.transAxes, fontsize="large")
        im=plots.plot_colormesh(ax_ani,pictures[i].T)
        ims.append([im,title])
    ani = animation.ArtistAnimation(fig_ani, ims, interval=inp.get("delay"), blit=True, repeat_delay=400)

    ###Correlation
    select=np.logical_and(lat>30,lat<60)
    correlation_6h=nwp.correlation(vort_n[:,select],vort_calc_6h[:,select])
    correlation_24h=nwp.correlation(vort_n[:,select],vort_calc_24h[:,select])
    print("Correlation of last field with 6h " +str(correlation_6h))
    print("Correlation with 24h: " +str(correlation_24h))

    ###Save
    if args.s:
        animation_path=os.path.join(inp.get("folder"),inp.get("savename"))+".mp4"
        results_path=os.path.join(inp.get("folder"),inp.get("savename"))+".res"
        if inp.get("video"):
            ani.save(animation_path)
        if inp.get("results"):
            np.savetxt(results_path, np.atleast_1d([correlation_6h, correlation_24h]))
        inp.write_log(animation_path,file_ext=".log")

    else:
        plt.show()







main()



