import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import os
import time
from importlib import reload
import ipywidgets as IP

"""The only really useful function here is CrackInteract"""

# ---------------------------------------------------------------------------- #
#                                  ANIMATIONS                                  #
# ---------------------------------------------------------------------------- #
def ScatterAnim(Nx,Ny,Nz,Nt,  Xt,Yt,Zt,  Vt,Ht,  CXt,CYt,CZt,Expt,  dr,dt):
    H = Ht[0]
    X,Y,Z = Xt[0], Yt[0], Zt[0]
    V = Vt[0]
    X_m, Y_m, Z_m = np.meshgrid(np.arange(Nx,step=dr), np.arange(Ny,step=dr), np.arange(Nz,step=dr), indexing='ij')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    vmin, vmax = 0,np.max(Ht)
    cmap = 'copper_r'

    # Plot contour surfaces
    #A = ax.scatter(X[:,:,-1], Y[:,:,-1], Z[:,:,-1], c= H[:,:,-1], s=20*np.power(V[:,:,-1],1/3), vmin =vmin, vmax=vmax, cmap=cmap) # (Y,Z) plane drawing
    #B = ax.scatter(X[ 0,:,:], Y[ 0,:,:], Z[ 0,:,:], c= H[ 0,:,:], s=20*np.power(V[ 0,:,:],1/3), vmin =vmin, vmax=vmax, cmap=cmap) # (X,Z)
    #C = ax.scatter(X[:,-1,:], Y[:,-1,:], Z[:,-1,:], c= H[:,-1,:], s=20*np.power(V[:,-1,:],1/3), vmin =vmin, vmax=vmax, cmap=cmap) # (Y,Z)
    D = ax.scatter(X, Y, Z, c= H, s=1000*np.power(V,1/3), vmin =vmin, vmax=vmax, cmap=cmap,marker='s')
    E = ax.scatter(X_m[:,:,-1]+.5,Y_m[:,:,-1],Z_m[:,:,-1], c=CXt[0][:,:,-1], marker='|', s=200)
    F = ax.scatter(X_m[:,:,-1],Y_m[:,:,-1]+.5,Z_m[:,:,-1], c=CYt[0][:,:,-1], marker='_', s=200)


    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Plot edges
    edges_kw = dict(color='black', linewidth=2, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], [zmax, zmax], **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    # Set labels and zticks
    ax.set(xlabel='X', ylabel='Y', zlabel='Z')

    # Set zoom and angle view
    ax.view_init(90, 90, 0) 
    ax.set_box_aspect(None, zoom=1)

    # Colorbar
    fig.colorbar(D, ax=ax, fraction=0.02, pad=0.1, label='Water content (% of sat.)')   
    plt.title('State at t = ' +str(0)+'ms')

    # ----------------------------------- Anim ----------------------------------- #
    def update(t):
        global D, vmin,vmax

        # for each frame, update the data stored on the artist.
        print('POC')
        H = Ht[t]
        X,Y,Z = Xt[t], Yt[t], Zt[t]
        V = Vt[t]

        # update the scatter plot:
        D.remove()
        #E.remove()
        #F.remove()
        D = ax.scatter(X, Y, Z, c= H, s=1000*np.power(V,1/3), vmin =vmin, vmax=vmax, cmap=cmap,marker='s') # (Y,Z)
        E = ax.scatter(X_m[:,:,-1]+.5,Y_m[:,:,-1],Z_m[:,:,-1], c=CXt[t][:,:,-1], marker='|', s=200)
        F = ax.scatter(X_m[:,:,-1],Y_m[:,:,-1]+.5,Z_m[:,:,-1], c=CYt[t][:,:,-1], marker='_', s=200)


        plt.title('State at t = ' +str(np.around(t*dt,decimals=1)) +'s')
        return (D)

    ani = anim.FuncAnimation(fig=fig, func=update, frames=Nt+1, interval=200)
    ani.save(filename="/home/flavien/Documents/M2/NLP/test.gif", writer="pillow")

    # Show Figure
    plt.show()







# ---------------------------------------------------------------------------- #
#                                     PLOTS                                    #
# ---------------------------------------------------------------------------- #
def CrackInteract(Nx,Ny,Nz,Nt,  Xsim,Ysim,Zsim,  Vsim,Hsim,  CXsim,CYsim,CZsim,Expsim  ,dr,dt, Hnorm='log',Cnorm=(0,1.5),mode=0, hist=True):
    """
    Plots interactive plot of cracks and particles.
    Hnorm   : Color norm type for watr content, 'log' or None (linear)
    Cnorm   : Color norm bounds for cracks, (0,1.5) for 1 run, (.5,1.1) for 10 averages (recommended)
    mode    : Simulation mode : 'average' for averages over all simulations, index of simulation to plot for a single simulation
    hist    : whther to plot histograms or not
    """
    
    if mode=='average':
        Xt,Yt,Zt,  Vt,Ht,  CXt,CYt,CZt,Expt = dr*np.average(Xsim,axis=0),dr*np.average(Ysim,axis=0),dr*np.average(Zsim,axis=0),  dr**3*np.average(Vsim,axis=0),np.average(Hsim,axis=0),  np.average(CXsim,axis=0),np.average(CYsim,axis=0),np.average(CZsim,axis=0),np.average(Expsim,axis=0)
    elif type(mode)==int:
        index_t=mode
        Xt,Yt,Zt,  Vt,Ht,  CXt,CYt,CZt,Expt = dr*Xsim[index_t],dr*Ysim[index_t],dr*Zsim[index_t],  dr**3*Vsim[index_t],Hsim[index_t],  CXsim[index_t],CYsim[index_t],CZsim[index_t],Expsim[index_t]

    cmap = 'copper_r'
    vmin, vmax = 1e-5,np.max(Ht)
    Cnorm_min,Cnorm_max = Cnorm

    def plot_crack(t=Nt,z=Nz-1):
        X_m = (np.vstack((np.roll(Xt[t,:,:,z],-1,axis=0)[:-1,:], np.array([Xt[t,-1,:,z]]    )+dr)) + Xt[t,:,:,z] )/2
        Y_m = (np.hstack((np.roll(Yt[t,:,:,z],-1,axis=1)[:,:-1], np.transpose([Yt[t,:,-1,z]])+dr)) + Yt[t,:,:,z] )/2

        fig = plt.figure(figsize=(np.around(Nx/Ny*7)+4, 7), layout='constrained')
        
        if not(hist):
            ax = fig.add_subplot()
            ax.set(xlabel = r'$x ~ (\mu m)$', ylabel = r'$y ~ (\mu m)$', xlim=(-.05*dr*Nx, np.max(X_m)+.02*dr*Nx), ylim=(-.05*dr*Ny, np.max(Y_m)+.02*dr*Ny), aspect=1.)

        elif(hist):
            CX_avg = np.average(CXt[t,:,:,z],axis=1)
            CY_avg = np.average(CYt[t,:,:,z],axis=0)
            print(np.shape(CX_avg),np.shape(CY_avg))

            ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
            ax.set(xlabel = r'$x ~ (\mu m)$', ylabel = r'$y ~ (\mu m)$', xlim=(-.05*dr*Nx, np.max(X_m)+.02*dr*Nx), ylim=(-.05*dr*Ny, np.max(Y_m)+.02*dr*Ny), aspect=1.)
            # Create marginal Axes, which have 25% of the size of the main Axes.  Note that the inset Axes are positioned *outside* (on the right and the top) of the main Axes, by specifying axes coordinates greater than 1.  Axes coordinates
            # less than 0 would likewise specify positions on the left and the bottom of the main Axes.
            ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
            ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
            # make some labels invisible
            ax_histx.xaxis.set_tick_params(labelbottom=False)
            ax_histy.yaxis.set_tick_params(labelleft=False)

            ax_histx.bar(x= np.arange(1,Nx+1)*dr, height= CX_avg, width=-.9*dr, align='edge')
            ax_histy.barh(y= np.arange(1,Ny+1)*dr, width= CY_avg, height=-.9*dr, align='edge')

            # the xaxis of ax_histx and yaxis of ax_histy are shared with ax, thus there is no need to manually adjust the xlim and ylim of these axis.
            ax_histx.set(ylim=(.5,1),yticks=[0.5, .75, 1])
            ax_histy.set(xlim=(.5,1),xticks=[0.5, .75, 1])


        clay   = ax.scatter(Xt[t,:,:,z], Yt[t,:,:,z], c = Ht[t,:,:,z], s=10e2/max((Nx,Ny))*np.power(Vt[t,:,:,z],1/3), vmin=vmin, vmax=vmax, cmap=cmap,marker='s',norm=Hnorm)
        crackX = ax.scatter(X_m, Yt[t,:,:,z], c = CXt[t,:,:,z], marker='|', s=1e4/Ny, alpha=1, norm=plt.Normalize(Cnorm_min,Cnorm_max))
        crackY = ax.scatter(Xt[t,:,:,z], Y_m, c = CYt[t,:,:,z], marker='_', s=1e4/Nx, alpha=1, norm=plt.Normalize(Cnorm_min,Cnorm_max))
        fig.colorbar(clay,ax=ax)
        fig.colorbar(crackX,ax=ax)
        plt.show()

    # Create interactive plot with sliders for degree and number of points
    IP.interact(plot_crack, t=(0, Nt),z=(0,Nz-1))

