import numpy as np
import matplotlib.pyplot as plt

from MyPython import Input as Inp


VERSION="1.0"


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

def simple_plot(x,y,ax, **kwargs):
    ax.plot(x,y, **kwargs)

def upstream(courant,phi0):
    return phi0-courant*(phi0-periodic_boundaries(phi0,np.arange(len(phi0))-1))#due to numpy internal index handling, this would also work without periodic_boundaries

#forward in time, centered in space
def centered_in_space(courant, phi0):
    return phi0-courant/2*(periodic_boundaries(phi0,np.arange(len(phi0))+1)-periodic_boundaries(phi0,np.arange(len(phi0))-1))


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

# def euler(mu,phi0):
def n_leapfrog_steps(func, phi0, phi1, steps, **kwargs):
    phi_nm1=func(phi0,phi1, **kwargs)#2
    phi_n=func(phi1, phi_nm1, **kwargs)#3
    for i in range(steps):
        phi_np1=func(phi_nm1, phi_n, **kwargs)
        phi_nm1=1*phi_n
        phi_n=1*phi_np1
    return phi_n


x=np.arange(0,360,1)

fig, ax=plt.subplots()

phi0=phi(x,10, sigma=10)
phi0=phi0+np.random.normal(0.0,0.01,len(phi0))
simple_plot(x,phi0,ax, label="initial")
mu=0.02

###Upstream
phin_up=upstream(mu,phi0)
for i in range(10000):
    phin_up=upstream(mu,phin_up)
    if np.max(phin_up)>3:
        break
simple_plot(x,phin_up,ax, label="upstream")

###centered
phin_centre=upstream(mu,phi0)
for i in range(10000):
    phin_centre=centered_in_space(mu,phin_centre)
    if np.max(phin_centre)>3:
        break
simple_plot(x,phin_centre,ax, label="centered")

###leapfrog
phi1=upstream(mu,phi0)
phin_leapfrog=n_leapfrog_steps(leapfrog_advection,phi0, phi1, 10000,mu=mu)
simple_plot(x,phin_leapfrog,ax, label="leapfrog")


ax.legend()
plt.show()