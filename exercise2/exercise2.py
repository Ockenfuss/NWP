import numpy as np
import matplotlib.pyplot as plt



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

def simple_plot(x,y,ax):
    ax.plot(x,y)

def upstream(courant,phi0):
    return phi0-courant*(phi0-periodic_boundaries(phi0,np.arange(len(phi0))-1))#due to numpy internal index handling, this would also work without periodic_boundaries


def leapfrog(mu, phi0,phi1):
    """Advance time with leapfrog scheme
    
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


x=np.arange(0,360,1)

fig, ax=plt.subplots()

phi0=phi(x,10, sigma=10)
simple_plot(x,phi0,ax)
mu=1.001
phin=upstream(mu,phi0)
for i in range(10000):
    phin=upstream(mu,phin)
simple_plot(x,phin,ax)
plt.show()