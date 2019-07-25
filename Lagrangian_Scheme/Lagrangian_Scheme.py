import numpy as np
import matplotlib.pyplot as plt

from MyPython import Input as Inp

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


def phi(x,t,u0=10,x0=0, sigma=1, phi0=1):
    return phi0*np.exp(-np.power((x-x0-u0*t),2)/(2*sigma*sigma))

def get_departure_constant(x, u, dt):
    return x-u*dt

def get_departure(x0, u0, dt,return_error=False):
    steps=100
    l=np.max(x0)
    x_dp=get_departure_constant(x0,u0,dt)
    u_dp=np.interp(x_dp%l,x0,u0)
    error=np.zeros((steps))
    for i in range(steps):
        x_dp_new=get_departure_constant(x0,u_dp,dt)
        if return_error:
            error[i]=nwp.rmse(x_dp, x_dp_new)
        x_dp=x_dp_new
        u_dp=np.interp(x_dp%l,x0,u0)
    if return_error:
        return x_dp, u_dp, error
    else:
        return x_dp, u_dp

def advect(x0,u0,dt, steps):
    x_dp, u_dp=get_departure(x0,u0,dt)
    for i in range(steps):
        x_dp, u_dp=get_departure(x0,u_dp,dt)
    return x0, u_dp
    

def plot_departure(ax, x_dp,u_dp, x0, u0):
    x_ap=x_dp+dt*u_dp
    ax.plot(x0, u0)
    ax.scatter(x0, np.ones(len(x0))*0.5)
    ax.scatter(x_dp, np.ones(len(x_dp))*0.0)

    for i in range(len(x0)):
        ax.plot([x_dp[i],x_ap[i]],[0,0.5])
        # ax.add_artist(line)
plot=2
dx=1
x0=np.arange(0,360,dx)
u0=phi(x0,20, sigma=50)
dt=50
x_dp, u_dp, error=get_departure(x0,u0,dt, True)

###Check

if plot==1:
    fig, ax=plt.subplots()
    plot_departure(ax,x_dp, u_dp, x0,u0)
    fig.savefig("Departure_points.pdf")
if plot==2:
    fig, ax=plt.subplots()
    ax.plot(error)
    fig.savefig("Error_evolution.pdf")




###Lagrangian advection
if plot==3:
    fig, ax=plt.subplots()
    x_adv, u_adv=advect(x0,u0,dt,3)
    ax.plot(x_adv, u_adv)
    ax.plot(x0,u0)
    fig.savefig("Advected_curve.pdf")

plt.show()