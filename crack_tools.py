import numpy as np
import scipy as sp
import pandas as pd
import os
import wave
import time
from importlib import reload



# ---------------------------------------------------------------------------- #
#                                Base Functions                                #
# ---------------------------------------------------------------------------- #
def Shift(T,dis):
    """
    Shifts a tensor by a displacement vector dis. (T)ijk  ->  (T)i+dis[0],j+dis[1],k+dis[2]  

    Parameters :
    T : tensor to roll
    dis : 3-list of ints (displacement along X,Y,Z)

    Notes : 
    /!\ rolling (T)ijk by +1 along axis 0 returns (T)i-1,j,k
    """
    Rolled = np.copy(T)
    if dis[0]!=0:
        Rolled = np.roll(Rolled, -dis[0], axis=0)
    if dis[1]!=0:
        Rolled = np.roll(Rolled, -dis[1], axis=1)
    if dis[2]!=0:
        Rolled = np.roll(Rolled, -dis[2], axis=2)
    return(Rolled)
    

def Dist(T, axT,dis, Nx,Ny,Nz):
    """
    Computes (i+dis[0],j+dis[1],k+dis[2]) - (i,j,k) distance tensor along an axis axT (determined by T (X,Y or Z)). 
    Periodic boundary condition correction included.
    """
    bound_corr = np.zeros((Nx,Ny,Nz))
    if axT==0:
        if dis[0]==1:
            bound_corr += Nx*np.vstack(( np.zeros((Nx-1,Ny,Nz)), np.ones((1,Ny,Nz)) ))
        elif dis[0]==-1:
            bound_corr -= Nx*np.vstack(( np.ones((1,Ny,Nz)), np.zeros((Nx-1,Ny,Nz)) ))

    if axT==1:
        if dis[1]==1:
            bound_corr += Ny*np.hstack(( np.zeros((Nx,Ny-1,Nz)), np.ones((Nx,1,Nz)) )) 
        elif dis[1]==-1:
            bound_corr -= Ny*np.hstack(( np.ones((Nx,1,Nz)), np.zeros((Nx,Ny-1,Nz)) ))

    if axT==2:
        if dis[2]==1:
            bound_corr += Nz*np.dstack(( np.zeros((Nx,Ny,Nz-1)), np.ones((Nx,Ny,1)) )) 
        elif dis[2]==-1:
            bound_corr -= Nz*np.dstack(( np.ones((Nx,Ny,1)), np.zeros((Nx,Ny,Nz-1)) )) 

    return Shift(T,dis) - T + bound_corr





# ---------------------------------------------------------------------------- #
#                              Energy Minimization                             #
# ---------------------------------------------------------------------------- #
def Energy(X,Y,Z, L0X,L0Y,L0Z, Nx,Ny,Nz, kt,ks, CX,CY,CZ):
    """
    Calculates total tension and shearing energy of a cubic particle system.

    Parameters :
    - pos       : flattened array of position tensors [X,Y,Z]
    - L0X,L0Y,L0Z : tensors, rest lengths of tension spring forces between i and i+1 along axes X,Y,Z
    - Nx,Ny,Nz  : scalars, sizes of the system along x, y and z
    - kt, ks    : scalars, tension and shearing spring constants

    Returns :
    - Etot      : scalar, total enetgy of the system

    Notes : 
    - np.roll(A, -1, axis=k) corresponds to looking at *+1 th* element of A along axis k !!!
    """
    # Tension energies calculation :  /!\ Multiplication by C in order to remove contributions from cracked links
    EtX = CX*kt*np.square(np.roll(X,-1, axis=0) - X - L0X + Nx*np.vstack((np.zeros((Nx-1,Ny,Nz)), np.ones((1,Ny,Nz)))) )/2 #Xi+1 - Xi - L0(i+1,i) - a term to fix X0_XNx distance
    EtY = CY*kt*np.square(np.roll(Y,-1, axis=1) - Y - L0Y + Ny*np.hstack((np.zeros((Nx,Ny-1,Nz)), np.ones((Nx,1,Nz)))) )/2
    EtZ = CZ*kt*np.square(np.dstack(( (np.roll(Z,-1, axis=2) - Z - L0Z)[:,:,:-1], np.zeros((Nx,Ny,1)) )) )/2 # Et(z=0 - z=-1) = 0 BC

    # Shearing energies calculation :
    EsXY = CY*ks*np.square(np.roll(X,-1, axis=1) - X )/2                                                # Due to X distance, exerted by Y neighbours
    EsXZ = CZ*ks*np.square(np.dstack(( (np.roll(X,-1, axis=2) - X )[:,:,:-1], np.zeros((Nx,Ny,1)) )))/2 # Due to X distance, exerted by Z neighbours
    EsYX = CX*ks*np.square(np.roll(Y,-1, axis=0) - Y )/2                                                # Due to Y distance, exerted by X neighbours
    EsYZ = CZ*ks*np.square(np.dstack(( (np.roll(Y,-1, axis=2) - Y )[:,:,:-1], np.zeros((Nx,Ny,1)) )))/2 # Due to Y distance, exerted by Z neighbours
    EsZX = CX*ks*np.square(np.dstack(( (np.roll(Z,-1, axis=0) - Z )[:,:,:-1], np.zeros((Nx,Ny,1)) )))/2 # Due to Z distance, exerted by X neighbours
    EsZY = CY*ks*np.square(np.dstack(( (np.roll(Z,-1, axis=1) - Z )[:,:,:-1], np.zeros((Nx,Ny,1)) )))/2 # Due to Z distance, exerted by Y neighbours
    
    Etot = np.sum((EtX,EtY,EtZ, EsXY,EsXZ,EsYX,EsYZ,EsZX,EsZY)) 
    return(Etot)


def JacobianE(X,Y,Z, L0X,L0Y,L0Z, Nx,Ny,Nz, kt,ks, CX,CY,CZ):
    """
    Jacobian of Energy with respect to all 3*Nx*Ny*Nz positions.

    Notes :
    Each node contibutes two times to tension energy, once for the link before, once for the link after. Cracking has to be rolled for link i-1_i. 
    """
    JtX = - CX*kt*(np.roll(X,-1, axis=0) - X - L0X + Nx*np.vstack((np.zeros((Nx-1,Ny,Nz)), np.ones((1,Ny,Nz)))) ) + np.roll(CX,1,axis=0)*kt*(X - np.roll(X,1, axis=0) - np.roll(L0X,1,axis=0) + Nx*np.vstack((np.ones((1,Ny,Nz)), np.zeros((Nx-1,Ny,Nz)) )))
    JtY = - CY*kt*(np.roll(Y,-1, axis=1) - Y - L0Y + Ny*np.hstack((np.zeros((Nx,Ny-1,Nz)), np.ones((Nx,1,Nz)))) ) + np.roll(CY,1,axis=1)*kt*(Y - np.roll(Y,1, axis=1) - np.roll(L0Y,1,axis=1) + Ny*np.hstack((np.ones((Nx,1,Nz)), np.zeros((Nx,Ny-1,Nz)) )))
    JtZ = - CZ*kt*(np.dstack(( (np.roll(Z,-1, axis=2) - Z - L0Z)[:,:,:-1], np.zeros((Nx,Ny,1)) ))               ) + np.roll(CZ,1,axis=2)*kt*(np.dstack(( np.zeros((Nx,Ny,1)), (Z - np.roll(Z,1, axis=2) - np.roll(L0Z,1,axis=2))[:,:,1:] ))                )

    JsXy = - CY*ks*(np.roll(X,-1, axis=1) - X )                                                + np.roll(CY,1,axis=1)*ks*(X - np.roll(X,1, axis=1))
    JsXz = - CZ*ks*(np.dstack(( (np.roll(X,-1, axis=2) - X )[:,:,:-1], np.zeros((Nx,Ny,1)) ))) + np.roll(CZ,1,axis=2)*ks*(np.dstack(( np.zeros((Nx,Ny,1)), (X - np.roll(X,1, axis=2))[:,:,1:] )))
    JsYx = - CX*ks*(np.roll(Y,-1, axis=0) - Y )                                                + np.roll(CX,1,axis=0)*ks*(Y - np.roll(Y,1, axis=0))
    JsYz = - CZ*ks*(np.dstack(( (np.roll(Y,-1, axis=2) - Y )[:,:,:-1], np.zeros((Nx,Ny,1)) ))) + np.roll(CZ,1,axis=2)*ks*(np.dstack(( np.zeros((Nx,Ny,1)), (Y - np.roll(Y,1, axis=2))[:,:,1:] )))
    JsZx = - CX*ks*(np.dstack(( (np.roll(Z,-1, axis=0) - Z )[:,:,:-1], np.zeros((Nx,Ny,1)) ))) + np.roll(CX,1,axis=0)*ks*(np.dstack(( np.zeros((Nx,Ny,1)), (Z - np.roll(Z,1, axis=0))[:,:,1:] )))
    JsZy = - CY*ks*(np.dstack(( (np.roll(Z,-1, axis=1) - Z )[:,:,:-1], np.zeros((Nx,Ny,1)) ))) + np.roll(CY,1,axis=1)*ks*(np.dstack(( np.zeros((Nx,Ny,1)), (Z - np.roll(Z,1, axis=1))[:,:,1:] )))
    
    return(JtX+JsXy+JsXz , JtY+JsYx+JsYz , JtZ+JsZx+JsZy) 


def HessianE(pos, arb,Nx,Ny,Nz, kt,ks, CX, CY,CZ): #DOES NOT WORK PROPERLY
    X = np.reshape(arb[0         :  Nx*Ny*Nz], (Nx,Ny,Nz))
    Y = np.reshape(arb[  Nx*Ny*Nz:2*Nx*Ny*Nz], (Nx,Ny,Nz))
    Z = np.reshape(arb[2*Nx*Ny*Nz:3*Nx*Ny*Nz], (Nx,Ny,Nz)) # Hess is independant of X, arb is an arbitrary vector

    cx, cy, cz = np.pad(CX, ((0,0),(0,0),(1,1)), mode='constant', constant_values=0), np.pad(CY, ((0,0),(0,0),(1,1)), 'constant', constant_values=0), np.pad(CZ, ((0,0),(0,0),(1,1)), 'constant', constant_values=0) #pad upper and lower borders w/ 0 (no links between bottom and top)
    
    # Tension energy curvatures
    diiHtX = (cx*2*kt + np.roll(cx,1,axis=0)*2*kt)[:,:,1:-1] * X                      # derivative wrt x_i**, x_i**
    dijHtX = - 2*kt*cx[:,:,1:-1]                             * np.roll(X, -1, axis=0) # derivative wrt x_i**, x_i+1**
    djiHtX = - 2*kt*np.roll(cx,1,axis=0)[:,:,1:-1]           * np.roll(X,  1, axis=0) # derivative wrt 

    diiHtY = (cy*2*kt + np.roll(cy,1,axis=1)*2*kt)[:,:,1:-1] * Y                      # derivative wrt y_*i*, y_*i*
    dijHtY = - 2*kt*cy[:,:,1:-1]                             * np.roll(Y, -1, axis=1) # derivative wrt y_*i*, y_*i+1*
    djiHtY = - 2*kt*np.roll(cy,1,axis=1)[:,:,1:-1]           * np.roll(X,  1, axis=1)

    diiHtZ = (cz*2*kt + np.roll(cz,1,axis=2)*2*kt)[:,:,1:-1] * Z
    dijHtZ = - 2*kt*cz[:,:,1:-1]                             * np.roll(Z, -1, axis=2)
    djiHtZ = - 2*kt*np.roll(cz,1,axis=1)[:,:,1:-1]           * np.roll(Z,  1, axis=2)

    # Shearing energy curvatures
    diiHsXy = (cy*2*ks + np.roll(cy,1,axis=1)*2*ks)[:,:,1:-1] * X                      # 2nd der of energy due to X displacement wrt x_*i*, x_*i*
    dijHsXy = - 2*ks*cy[:,:,1:-1]                             * np.roll(X, -1, axis=1) # derivative wrt x_*i*, x_*i+1*
    djiHsXy = - 2*ks*np.roll(cy,1,axis=1)[:,:,1:-1]           * np.roll(X,  1, axis=1)

    diiHsXz = (cz*2*ks + np.roll(cz,1,axis=2)*2*ks)[:,:,1:-1] * X
    dijHsXz = - 2*ks*cz[:,:,1:-1]                             * np.roll(X, -1, axis=2)
    djiHsXz = - 2*ks*np.roll(cz,1,axis=2)[:,:,1:-1]           * np.roll(X,  1, axis=2)

    diiHsYx = (cx*2*ks + np.roll(cx,1,axis=0)*2*ks)[:,:,1:-1] * Y
    dijHsYx = - 2*ks*cx[:,:,1:-1]                             * np.roll(Y, -1, axis=0)
    djiHsYx = - 2*ks*np.roll(cx,1,axis=0)[:,:,1:-1]           * np.roll(Y,  1, axis=0)

    diiHsYz = (cz*2*ks + np.roll(cz,1,axis=2)*2*ks)[:,:,1:-1] * Y
    dijHsYz = - 2*ks*cz[:,:,1:-1]                             * np.roll(Y, -1, axis=2)
    djiHsYz = - 2*ks*np.roll(cz,1,axis=2)[:,:,1:-1]           * np.roll(Y,  1, axis=2)

    diiHsZx = (cx*2*ks + np.roll(cx,1,axis=0)*2*ks)[:,:,1:-1] * Z
    dijHsZx = - 2*ks*cx[:,:,1:-1]                             * np.roll(Z, -1, axis=0)
    djiHsZx = - 2*ks*np.roll(cx,1,axis=0)[:,:,1:-1]           * np.roll(Z,  1, axis=0)

    diiHsZy = (cy*2*ks + np.roll(cy,1,axis=1)*2*ks)[:,:,1:-1]
    dijHsZy = - 2*ks*cy[:,:,1:-1]                             * np.roll(Z, -1, axis=1)
    djiHsZy = - 2*ks*np.roll(cy,1,axis=1)[:,:,1:-1]           * np.roll(Z,  1, axis=1)

    # Sum all 
    resX = diiHtX + dijHtX + djiHtX   +   diiHsXy + dijHsXy + djiHsXy   +   diiHsXz + dijHsXz + djiHsXz
    resY = diiHtY + dijHtY + djiHtY   +   diiHsYx + dijHsYx + djiHsYx   +   diiHsYz + dijHsYz + djiHsYz
    resZ = diiHtZ + dijHtZ + djiHtZ   +   diiHsZx + dijHsZx + djiHsZx   +   diiHsZy + dijHsZy + djiHsZy
    return((resX, resY, resZ))

def UpdatePos(X,Y,Z,L0X,L0Y,L0Z, Nx,Ny,Nz, kt,ks, CX,CY,CZ, bottom_fixed):
    """
    Updates the positions of a cubic particle system whose particle size have changed, thus changing rest lengths for tension and shearing forces. 
    """
    if bottom_fixed :
        def fitfunc(pos):
            # De-flatten params :
            Xfit = np.dstack(( X[:,:,0], np.reshape(pos[0             :  Nx*Ny*(Nz-1)], (Nx,Ny,Nz-1)) ))
            Yfit = np.dstack(( Y[:,:,0], np.reshape(pos[  Nx*Ny*(Nz-1):2*Nx*Ny*(Nz-1)], (Nx,Ny,Nz-1)) ))
            Zfit = np.dstack(( Z[:,:,0], np.reshape(pos[2*Nx*Ny*(Nz-1):3*Nx*Ny*(Nz-1)], (Nx,Ny,Nz-1)) ))
            return Energy(Xfit,Yfit,Zfit, L0X,L0Y,L0Z, Nx,Ny,Nz, kt,ks, CX,CY,CZ)

        def fitjac(pos):
            # De-flatten params :
            Xfit = np.dstack(( X[:,:,0], np.reshape(pos[0             :  Nx*Ny*(Nz-1)], (Nx,Ny,Nz-1)) ))
            Yfit = np.dstack(( Y[:,:,0], np.reshape(pos[  Nx*Ny*(Nz-1):2*Nx*Ny*(Nz-1)], (Nx,Ny,Nz-1)) ))
            Zfit = np.dstack(( Z[:,:,0], np.reshape(pos[2*Nx*Ny*(Nz-1):3*Nx*Ny*(Nz-1)], (Nx,Ny,Nz-1)) ))
            
            Jx,Jy,Jz = JacobianE(Xfit,Yfit,Zfit, L0X,L0Y,L0Z, Nx,Ny,Nz, kt,ks, CX,CY,CZ)
            return np.ravel((Jx[:,:,1:],Jy[:,:,1:],Jz[:,:,1:]))        
        
        fit = sp.optimize.minimize(fitfunc, x0 = np.ravel((X[:,:,1:],Y[:,:,1:],Z[:,:,1:])), jac=fitjac) #??? , hessp=fithessp, method='trust-ncg'
        
        X = np.dstack(( X[:,:,0], np.reshape(fit.x[0             :  Nx*Ny*(Nz-1)], (Nx,Ny,Nz-1)) ))
        Y = np.dstack(( Y[:,:,0], np.reshape(fit.x[  Nx*Ny*(Nz-1):2*Nx*Ny*(Nz-1)], (Nx,Ny,Nz-1)) ))
        Z = np.dstack(( Z[:,:,0], np.reshape(fit.x[2*Nx*Ny*(Nz-1):3*Nx*Ny*(Nz-1)], (Nx,Ny,Nz-1)) ))
    
    else :
        def fitfunc(pos):
            # De-flatten params :
            X = np.reshape(pos[0         :  Nx*Ny*Nz], (Nx,Ny,Nz))
            Y = np.reshape(pos[  Nx*Ny*Nz:2*Nx*Ny*Nz], (Nx,Ny,Nz))
            Z = np.reshape(pos[2*Nx*Ny*Nz:3*Nx*Ny*Nz], (Nx,Ny,Nz))
            return Energy(X,Y,Z, L0X,L0Y,L0Z, Nx,Ny,Nz, kt,ks, CX,CY,CZ)
    
        def fitjac(pos):
            # De-flatten params :
            X = np.reshape(pos[0         :  Nx*Ny*Nz], (Nx,Ny,Nz))
            Y = np.reshape(pos[  Nx*Ny*Nz:2*Nx*Ny*Nz], (Nx,Ny,Nz))
            Z = np.reshape(pos[2*Nx*Ny*Nz:3*Nx*Ny*Nz], (Nx,Ny,Nz))
            return np.ravel(JacobianE(X,Y,Z, L0X,L0Y,L0Z, Nx,Ny,Nz, kt,ks, CX,CY,CZ))
    
        def fithessp(pos, arb):
            return np.ravel(HessianE(pos, arb,Nx,Ny,Nz, kt,ks, CX,CY,CZ)) # HessianMatrix*arb vector (NxNyNz length)  

        fit = sp.optimize.minimize(fitfunc, x0 = np.ravel((X,Y,Z)), jac=fitjac) # hessp=fithessp, method='trust-ncg'  
    
        X = np.reshape(fit.x[0         :  Nx*Ny*Nz], (Nx,Ny,Nz))
        Y = np.reshape(fit.x[  Nx*Ny*Nz:2*Nx*Ny*Nz], (Nx,Ny,Nz))
        Z = np.reshape(fit.x[2*Nx*Ny*Nz:3*Nx*Ny*Nz], (Nx,Ny,Nz))

    return(X,Y,Z)




# ---------------------------------------------------------------------------- #
#                          Drying, Shrinking, Cracking                         #
# ---------------------------------------------------------------------------- #
def Drying(H,V, CX,CY,CZ, Nx,Ny,Nz):
    cx, cy, cz = np.pad(CX, ((0,0),(0,0),(1,1)), mode='constant', constant_values=1), np.pad(CY, ((0,0),(0,0),(1,1)), 'constant', constant_values=1), np.pad(CZ, ((0,0),(0,0),(1,1)), 'constant', constant_values=1) #pad upper and lower borders w/ ones to avoid seeing cracks at other end
    Surface = np.dstack(( np.zeros((Nx,Ny,Nz-1)), np.ones((Nx,Ny,1)) )) #should be unified by using adjcrackZ
    
    AdjCrackX = np.where(cx==0,1,0) + np.where(Shift(cx,(-1,0,0))==0,1,0) 
    AdjCrackY = np.where(cy==0,1,0) + np.where(Shift(cy,(0,-1,0))==0,1,0) # nb of adjacent cracks
    AdjCrack = AdjCrackX + AdjCrackY
    
    NearCrackXY = np.where(Shift(cx,(0,1,0))==0,1,0)+ np.where(Shift(cx,(0,-1,0))==0,1,0) + np.where(Shift(cx,(-1,1,0))==0,1,0)+ np.where(Shift(cx,(-1,-1,0))==0,1,0) # Cracks along X in front and behind
    NearCrackYX = np.where(Shift(cy,(1,0,0))==0,1,0)+ np.where(Shift(cy,(-1,0,0))==0,1,0) + np.where(Shift(cy,(1,-1,0))==0,1,0)+ np.where(Shift(cy,(-1,-1,0))==0,1,0) # Cracks along Y left and right
    NearCrackXZ = np.where(Shift(cx,(0,0,1))==0,1,0)+ np.where(Shift(cx,(0,0,-1))==0,1,0) + np.where(Shift(cx,(-1,0,1))==0,1,0)+ np.where(Shift(cx,(-1,0,-1))==0,1,0) # Cracks along X above and below
    NearCrackYZ = np.where(Shift(cy,(0,0,1))==0,1,0)+ np.where(Shift(cy,(0,0,-1))==0,1,0) + np.where(Shift(cy,(0,-1,1))==0,1,0)+ np.where(Shift(cy,(0,-1,-1))==0,1,0) # Cracks along Y, nearby along Z (above or below)
    NearCrack = NearCrackXZ + NearCrackYZ #+ NearCrackXY + NearCrackYX
    Exposition = Surface + AdjCrack[:,:,1:-1] +.25*NearCrack[:,:,1:-1] # Exposed surface ; .25 can be varied in [0,.5] 
    
    k = .1*1/6 #k kinetics constant related to drying of a single face, at fixed normalized air humidity
    H = H*(1 - k*Exposition)  
    return(H, Exposition)





# ---------------------------------------------------------------------------- #
#                                    Wrapper                                   #
# ---------------------------------------------------------------------------- #
def Simulate(Nx,Ny,Nz,Nt,  Xt,Yt,Zt,  V0,Vt,Ht,  CXt,CYt,CZt,Expt,thresh,  kt=1,ks=1,  bottom_fixed=True):
    """
  	Simulates drying one time with input parameters (assuming input params are already initialized correctly)
  	"""
    X,Y,Z, V,H, CX,CY,CZ,Exp = Xt[0],Yt[0],Zt[0], Vt[0],Ht[0], CXt[0],CYt[0],CZt[0],Expt[0]

    for t in range(Nt):
        TIME_REF = time.time()
    	# Drying at surface and nearby cracks :
        H, Exp = Drying(H,V, CX,CY,CZ, Nx,Ny,Nz)

        # Some room if we want to take water diffusion into account

        # Shrinking :
        V = V0*(2+H)/3 # Expect 30% volume reduction (1 + 5H/6 before)
        size = np.power(V,1/3)/2 # HALF size of the cube

        # Rest lengths calculation
        L0X = np.roll(size,-1, axis=0) + size # i+1_i spring rest length is equal to sum of sizes of bloc i and i+1. 
        L0Y = np.roll(size,-1, axis=1) + size 
        L0Z = np.roll(size,-1, axis=2) + size # last elements along z have no meaning, since BC will be E(z=0 <-> z=-1) = 0

        # Position update : 
        # Minimization of energy, initguess X,Y,Z, ok if nearby local minimum bc anyway system does not explore whole space in fact
        # RQ : if we get to energy min each time, it means quasistatic, which might not actually be compatible w/ fast cracking
        # Z CRACKING HAS NOT YET BEEN TAKEN INTO ACCOUNT (link breaking)
        X,Y,Z = UpdatePos(X,Y,Z,L0X,L0Y,L0Z, Nx,Ny,Nz, kt,ks, CX,CY,CZ, bottom_fixed)

        # Cracking decision : compare L-L0 or L/L0 with thresh?
        Xdist = Dist(X,0,(1,0,0), Nx,Ny,Nz)
        Ydist = Dist(Y,1,(0,1,0), Nx,Ny,Nz)
        Zdist = Dist(Z,2,(0,0,1), Nx,Ny,Nz)    

        CX = np.min((CX,np.where(Xdist-L0X>thresh, 0, CX)), axis=0) # break links whose distance is bigger than threshold, among those not broken yet
        CY = np.min((CY,np.where(Ydist-L0Y>thresh, 0, CY)), axis=0) # Where True, yield 0, otherwise yield 1
        #CZ = np.min((CZ,np.where(Zdist-L0Z-thresh, 0, 1)), axis=0) # Useless since we don't take into account horizontal cracks

        # Append parameters to time evolution lists
        Xt.append(np.copy(X))
        Yt.append(np.copy(Y))
        Zt.append(np.copy(Z))
        Vt.append(np.copy(V))
        Ht.append(np.copy(H))
        CXt.append(np.copy(CX))
        CYt.append(np.copy(CY))
        CZt.append(np.copy(CZ))
        Expt.append(np.copy(Exp))

        if t%10 == 0:
            print('Step n°' +str(t) + ' took ' + str(time.time()- TIME_REF) + ' s')

    return(Xt,Yt,Zt,  Vt,Ht,  CXt,CYt,CZt,Expt)


def RepeatSim(Nx,Ny,Nz,Nt,  sigV=.001,sigC=.05,thresh=.13,  Nsim=1,save=True,  kt=1,ks=1,  dr=1,dt=1):
    """
    Repeats {initialization + simulation} Nsim times using given parameters, and saves it in folder
    """
    Xsim,Ysim,Zsim,  Vsim,Hsim,  CXsim,CYsim,CZsim,Expsim = [],[],[],  [],[],  [],[],[],[]
    
    for iter in range (Nsim):
        # ------------------------------ INITIALIZATION ------------------------------ #
        TIME_REF = time.time()

        H = np.ones((Nx, Ny, Nz))
        V0 =  np.random.normal(loc=1, scale =sigV, size=(Nx,Ny,Nz)) 
        """
        V0 = np.random.uniform(low=1-sigV, high=1+sigV, size=(Nx,Ny,Nz))"""
        V = V0*(2+H)/3 # original volume of elements, multiplied by a water related swelling factor

        X, Y, Z = np.meshgrid(np.arange(Nx,step=dr), np.arange(Ny,step=dr), np.arange(Nz,step=dr), indexing='ij') # X,Y,Z coordinates of each "particle"
        CX = np.vstack(( np.random.normal(loc=1, scale =sigC, size=(Nx-1,Ny,Nz)),np.zeros((1,Ny,Nz)) ))
        CY = np.hstack(( np.random.normal(loc=1, scale =sigC, size=(Nx,Ny-1,Nz)),np.zeros((Nx,1,Nz)) ))# sigC Noise on link strength
        CZ = np.dstack(( np.random.normal(loc=1, scale =sigC, size=(Nx,Ny,Nz-1)),np.zeros((Nx,Ny,1)) ))
        
        """
        CX = np.vstack(( np.random.uniform(low=1-sigC, high=1+sigC, size=(Nx-1,Ny,Nz)),np.zeros((1,Ny,Nz)) ))
        CY = np.hstack(( np.random.uniform(low=1-sigC, high=1+sigC, size=(Nx,Ny-1,Nz)),np.zeros((Nx,1,Nz)) ))# 1% Noise on link strength
        CZ = np.dstack(( np.random.uniform(low=1-sigC, high=1+sigC, size=(Nx,Ny,Nz-1)),np.zeros((Nx,Ny,1)) )) """ #If uniform noise rather than gauss

        size = np.power(V,1/3)/2 #HALF size of the cube
        L0X = np.roll(size,-1, axis=0) + size # i+1_i spring rest length is equal to sum of sizes of bloc i and i+1. 
        L0Y = np.roll(size,-1, axis=1) + size 
        L0Z = np.roll(size,-1, axis=2) + size # last elements along z have no meaning, since BC will be E(z=0 <-> z=-1) = 0
        X,Y,Z = UpdatePos(X,Y,Z,L0X,L0Y,L0Z, Nx,Ny,Nz, kt,ks, CX,CY,CZ, bottom_fixed=False)

        Ht, Xt, Yt, Zt, Vt, CXt, CYt, CZt = [np.copy(H)], [np.copy(X)], [np.copy(Y)], [np.copy(Z)], [np.copy(V)], [np.copy(CX)], [np.copy(CY)], [np.copy(CZ)]
        Expt = [np.dstack(( np.zeros((Nx,Ny,Nz-1)), np.ones((Nx,Ny,1)) ))]

        print('Initialization took ' + str(time.time()- TIME_REF) + ' s, for simulation n°' + str(iter))
        # -------------------------------- INTERATION -------------------------------- #
        Xt,Yt,Zt,  Vt,Ht,  CXt,CYt,CZt,Expt = Simulate(Nx,Ny,Nz,Nt,  Xt,Yt,Zt,  V0,Vt,Ht,  CXt,CYt,CZt,Expt,thresh)


        # ----------------------------------- SAVE ----------------------------------- #
        Xsim.append(np.copy(Xt))
        Ysim.append(np.copy(Yt))
        Zsim.append(np.copy(Zt))
        Vsim.append(np.copy(Vt))
        Hsim.append(np.copy(Ht))
        CXsim.append(np.copy(CXt))
        CYsim.append(np.copy(CYt))
        CZsim.append(np.copy(CZt))
        Expsim.append(np.copy(Expt))

        if(save):
            np.save('INPUT FOLDER PATH HERE'+str((Nx,Ny,Nz,Nt)) + '_' + str(iter+1) + '_runs' , [Xsim,Ysim,Zsim,  Vsim,Hsim,  CXsim,CYsim,CZsim,Expsim])
    
    return(Xsim,Ysim,Zsim,  Vsim,Hsim,  CXsim,CYsim,CZsim,Expsim)