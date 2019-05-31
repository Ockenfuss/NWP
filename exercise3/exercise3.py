import numpy as np
import matplotlib.pyplot as plt
import argparse
from MyPython import Input as Inp


VERSION="1.0"
par=argparse.ArgumentParser()
par.add_argument('infile')
par.add_argument('-s',action='store_true')
args=par.parse_args()
inp=Inp.Input(args.infile,version=VERSION)
inp.convert_type(int, "plot")
inp.show_data()


def phi(x,t,u0=10,x0=0, sigma=1, phi0=1):
    return phi0*np.exp(-np.power((x-x0-u0*t),2)/(2*sigma*sigma))

def periodic_boundaries(arr, index):
    """Apply periodic boundary conditions at 1D numpy array
    
    Arguments:
        arr {arr} -- Array
        index {arr} -- index/indexarray
    
    Returns:
        arr -- the input array evaluated at periodic conditions
    """
    return arr[index%len(arr)]

def simple_plot(x,y,ax, title="", label=""):
    ax.plot(x,y, label=label)
    ax.set_title(title)
    

def upstream_advection(phi0, courant=0.5):
    return phi0-courant*(phi0-periodic_boundaries(phi0,np.arange(len(phi0))-1))#due to numpy internal index handling, this would also work without periodic_boundaries

def leapfrog_advection(phi0,phi1, mu=0.5):
    """Advance one timestep with leapfrog scheme
    
    Arguments:
        mu {float} -- Courant number
        phi0 {arr} -- Function at time n-1
        phi1 {arr} -- Function at time n
    
    Returns:
        arr -- Function at time n+1
    """
    l=len(phi1)
    ind=np.arange(l)
    return phi0-mu*(phi1[(ind+1)%l]-phi1[(ind-1)%l])

def leapfrog_diffusion(phi_nm1, phi_n, mu=0.5, mu0=0.5):
    l=len(phi_n)
    ind=np.arange(l)
    return leapfrog_advection(phi_nm1, phi_n, mu=0.5)+2*mu0*(phi_n[(ind+1)%l]-2*phi_n[ind%l]+phi_n[(ind-1)%l])

def leapfrog_diffusion_lag(phi_nm1, phi_n, mu=0.5, mu0=0.5):
    l=len(phi_n)
    ind=np.arange(l)
    return leapfrog_advection(phi_nm1, phi_n, mu=0.5)+2*mu0*(phi_nm1[(ind+1)%l]-2*phi_nm1[ind%l]+phi_nm1[(ind-1)%l])


def n_leapfrog_steps(func, phi0, phi1, steps, **kwargs):
    phi_nm1=func(phi0,phi1, **kwargs)#2
    phi_n=func(phi1, phi_nm1, **kwargs)#3
    for i in range(steps):
        phi_np1=func(phi_nm1, phi_n, **kwargs)
        phi_nm1=1*phi_n
        phi_n=1*phi_np1
    return phi_n


def exercise3_part1():
    x=np.arange(0,360,1)
    fig, ax=plt.subplots(2,2, figsize=(20,20))
    mu=1.001
    mu0=0.05
    phi0=phi(x,10, sigma=10)#starting conditions
    phi0=phi0+0.1*np.random.rand(*phi0.shape)
    phi1=upstream_advection(phi0, mu)#starting conditions
    phi_n_adv=n_leapfrog_steps(leapfrog_advection, phi0, phi1, 10, mu=mu)
    phi_n_diff=n_leapfrog_steps(leapfrog_diffusion, phi0, phi1, 10, mu=mu, mu0=mu0)
    phi_n_diff_lag=n_leapfrog_steps(leapfrog_diffusion_lag, phi0, phi1, 100, mu=mu, mu0=mu0)

    simple_plot(x,phi0,ax[0,0], title="Starting Curve")
    simple_plot(x,phi_n_adv,ax[0,1], title="Leapfrog Advection")
    simple_plot(x,phi_n_diff,ax[1,0], title="Leapfrog Diffusion")
    simple_plot(x,phi_n_diff_lag,ax[1,1], title="Leapfrog Diffusion with lag")
    return fig, ax
    


if inp.get("plot")==1:
    fig, ax=exercise3_part1()
    if args.s:
        fig.savefig("PlotAdv.pdf")
        inp.write_log("PlotAdv.log")


if not args.s:
    plt.show()