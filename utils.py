import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rich import print, inspect
from rich.console import Console
from rich.progress import Progress

def set_params(ncx, ncy, gamma):

    # Frequencies, in octaves.
    fx, fy = 2**np.arange(ncx), 2**np.arange(ncy)
    
    # Octave amplitudes, decreasing as f^(-1/gamma)    
    Ax, Ay = (2**np.arange(1,ncx+1))**(-1/gamma), (2**np.arange(1,ncy+1))**(-1/gamma)
    
    # Octave amplitudes, decreasing as 1/sqrt(f)
    # Ax, Ay = 1/np.sqrt(2**np.arange(1,ncx+1)), 1/np.sqrt(2**np.arange(1,ncy+1))
    
    # print(f'{fx = }')
    # print(f'{Ax = }')
    # print(f'{fy = }')
    # print(f'{Ay = }')
    
    # Random phases generation
    phx = 2 * np.pi * np.random.random(ncx)
    phy = 2 * np.pi * np.random.random(ncy)
    return np.vstack((fx, Ax, phx)), np.vstack((fy, Ay, phy))

def make_Z(x, y, xprms, yprms):

    # Initalize Z and unpack the parameters
    Nx, Ny = x.shape[1], y.shape[0]
    Z = np.zeros((Ny, Nx))
    fx, Ax, phx = xprms
    fy, Ay, phy = yprms

    for i in range(xprms.shape[1]):
        Z += Ax[i] * np.sin((2*np.pi*fx[i])*x/Nx - phx[i])
    for i in range(yprms.shape[1]):
        Z += Ay[i] * np.sin((2*np.pi*fy[i])*y/Ny - phy[i])

    # Normalize
    minZ, maxZ = np.min(Z), np.max(Z)
    gap = maxZ-minZ
    Z = (Z - minZ)/gap
    return Z

def make_animation(x, y, xprms, yprms, step, update, plot, console, progress, save=False, nframes=100, filename = 'animation', fps = 30):
    frames = progress.add_task(description = 'Frames rendering', total = nframes)
    def animate(i, xprms, yprms, surf):
        ax.clear()
        progress.advance(frames, 1)
        xprms, yprms = update(xprms, yprms, step)
        Z = make_Z(x, y, xprms, yprms)
        surf = plot(ax, x, y, Z)
        if (i-1)%10 == 0: console.log(f'Frame n°{str(i-1).ljust(3)} computed | {str(i-1).ljust(3)}/ {str(nframes)}')
        return surf,
    
    # 3D surface plot, black backgrounds, no padding around the Axes.
    fig, ax = plt.subplots(subplot_kw = dict(projection='3d', facecolor='k'), facecolor='k', figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    Z = make_Z(x, y, xprms, yprms)
    surf = plot(ax, x, y, Z)
    # ax.axis('off')

    ani = animation.FuncAnimation(fig, animate, 
                                  fargs=(xprms, yprms, surf),
                                  interval=1000//fps, 
                                  blit=False, 
                                  frames=nframes)
    
    if save:
        ani.save(filename + '.gif', writer=animation.PillowWriter(fps=fps))
        
    return ani