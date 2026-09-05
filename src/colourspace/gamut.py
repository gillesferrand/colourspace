# -*- coding: utf-8 -*-
"""
Limits of the full human gamut, or the sRGB gamut, in CIE LCH space: Cmax(L,H)
"""

from __future__ import print_function
import sys
import numpy as np
import pylab as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from . import convert
from . import limits

import os
this_dir, this_filename = os.path.split(__file__)
this_dir += "/gamut"

# the gamut is cached
Cmax = {}

# ranges for L and H
# (several functions need to know this)
L_min = 0 ; L_max = 100
H_min = 0 ; H_max = 360

#---------------------------
# find the gamut boundary
# method 1: sample the LH plane, for each point convert to native space and check if within limits
#---------------------------

def valid_LCH_full(L,C,H):
    """ Checks if a LCH colour is in the human gamut """
    X,Y,Z = convert.Lab2XYZ(convert.LCH2Lab((L,C,H)))
    XYZ = np.stack((X,Y,Z),axis=-1)
    return limits.within_limits(XYZ,'XYZ',kind='cmp') # assumes limits have been set

def valid_LCH_sRGB(L,C,H):
    """ Checks if a LCH colour is in the sRGB gamut """
    R,G,B = convert.LCH2RGB(L,C,H)
    valid = lambda x: np.logical_and(0 <= x, x <= 1)
    return valid(R)*valid(G)*valid(B)
    #return np.logical_and(np.logical_and(valid(R),valid(G)),valid(B))

valid_LCH = {"full": valid_LCH_full, \
             "sRGB": valid_LCH_sRGB}

def find_Cmax_forward(res, gmt, save=False, plot=True, version='mapping'):
    """ Finds the maximum Cmax for each (L,H) pair with a precision of res points per unit of L,C,H """
    if res not in Cmax.keys(): Cmax[res] = {}
    if gmt=='full' and not 'XYZ' in limits.triangulation['cmp'].keys():
        limits.set_limits(l_step=10, l_min=360, l_max=780)
        limits.triangulate('XYZ')
    L = np.linspace(L_min,L_max,int((L_max-L_min)*res+1))
    H = np.linspace(H_min,H_max,int((H_max-H_min)*res+1))
    Cmax[res][gmt] = np.zeros((len(L),len(H)),dtype=np.float32)
    # version with loops
    if version == 'looping':
        for i in range(len(L)):
            for k in range(len(H)):
                Cmax[res][gmt][i,k] = find_Cmax_for_LH(L=L[i], H=H[k], Cres=1./res, gmt=gmt)
                sys.stdout.write("L = %.2f, H = %.2f, Cmax = %.2f"%(L[i],H[k],Cmax[res][gmt][i,k]))
                sys.stdout.write("\r")
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    # version with mapping
    if version == 'mapping':
        LL, HH = np.meshgrid(L,H,indexing='ij')
        Cmax[res][gmt][:,:] = find_Cmax_for_LH(LL, HH, Cres=1./res, gmt=gmt)
    if version == 'mapping3D':
        C = np.linspace(0, 200, int(200.*res)+1)
        LLL, CCC, HHH = np.meshgrid(L,C,H,indexing='ij')
        valid = valid_LCH[gmt](LLL,CCC,HHH)
        for i in range(len(L)):
            for k in range(len(H)):
                j_valid = np.where(valid[i,:,k])[0]
                Cmax[res][gmt][i,k] = C[j_valid[-1]] if len(j_valid)>0 else 0
    if save: save_Cmax_npy(res=res, gmt=gmt)
    if plot: plot_Cmax(res=res, gmt=gmt)

def find_Cmax_for_LH(L, H, Cres, gmt):
    """Finds the maximum C for a given (L,H) at a given resolution in a given gamut"""
    edge_detector = lambda L,H: find_edge_by_dichotomy(lambda c: valid_LCH[gmt](L,c,H), xmin=0, xmax=200, dx=Cres)
    Cmax = np.vectorize(edge_detector)(L,H)
    Cmax = np.where(L<=  0,     0,Cmax)
    Cmax = np.where(L>=100,     0,Cmax)
    Cmax = np.where(L<   0,np.nan,Cmax)
    Cmax = np.where(L> 100,np.nan,Cmax)
    #Cmax = np.where(np.logical_or(L<=0,L>=100),     0,Cmax)
    #Cmax = np.where(np.logical_or(L< 0,L> 100),np.nan,Cmax)
    return Cmax

def find_edge_by_dichotomy(func, xmin, xmax, dx=1., iter_max=100):
    """Returns the point `x` (within resolution `dx`) where boolean function `func` changes value
       `func` is assumed to switch from True to False between `xmin` and `xmax`
    """
    xleft  = xmin
    xright = xmax
    xmid = 0.5*(xright-xleft)
    i = 0
    delta = dx
    while delta >= dx and i<iter_max:
        i += 1
        #print("i = %4i: x = [%6.2f, %6.2f]: func(%6.2f) = %i"%(i,xleft,xright,xmid,func(xmid)),end='')
        if func(xmid): xleft  = xmid
        else:          xright = xmid
        xmid_old = xmid
        xmid = 0.5*(xleft+xright)
        delta = abs(xmid_old-xmid)
        #print("-> x = [%6.2f, %6.2f], delta=%f"%(xleft,xright,delta))
    if i >= iter_max: print("edge not found at precision ",dx,"in ",iter_max," iterations")
    return np.around(xmid,int(np.ceil(np.log10(1./dx))))

#---------------------------
# find the gamut boundary
# method 2: discretize the gamut boundary in the native space, project it back to the LH plane
#---------------------------

def get_RGB_faces(num=10):
    """ Samples the faces of the RGB cube with `num` points per axis """
    array = {}
    for coord in ['R','G','B']: array[coord] = np.array([])
    block = {}
    block['0'] = np.zeros(num**2)
    block['1'] = np.ones( num**2)
    x = np.linspace(0,1,int(num))
    X,Y = np.meshgrid(x,x)
    block['x'] = X.flatten()
    block['y'] = Y.flatten()
    for x,y,z in [('R','G','B'), ('G','B','R'), ('B','R','G')]:
        for side in ['0','1']:
            array[x] = np.concatenate((array[x], block['x']))
            array[y] = np.concatenate((array[y], block['y']))
            array[z] = np.concatenate((array[z], block[side]))
    return array['R'],array['G'],array['B']

def get_edges_LCH_sRGB(res):
    R,G,B = get_RGB_faces(num=res)
    L,C,H = convert.RGB2LCH(R,G,B)
    return np.stack((L,C,H),axis=-1)

def get_edges_LCH_full(res):
    limits.set_limits(l_step=res, l_min=360, l_max=780)
    return limits.limits['cmp']['LCH']

get_edges_LCH = {"full": get_edges_LCH_full, \
                 "sRGB": get_edges_LCH_sRGB}

def find_Cmax_backward(res_native, res_LH, gmt, save=False, plot=True):
    """ Finds the maximum Cmax(L,H) by discretizing the gamut boundary in its native space
        for sRGB gamut: res_native = number of points along R, G, B
        for full gamut: res_native = delta_Lambda in nm
        the LH plane will be re-sampled regularly at resolution res_LH
    """
    if res_LH not in Cmax.keys(): Cmax[res_LH] = {}
    # get edges in LCH space
    LCH_max = get_edges_LCH[gmt](res_native)
    # interpolate the implicit function C(L,H)
    L = LCH_max[:,0]
    C = LCH_max[:,1]
    H = LCH_max[:,2]
    L_grid, H_grid = np.mgrid[L_min:L_max:1j*(L_max-L_min+1)*res_LH, H_min:H_max:1j*(H_max-H_min+1)*res_LH]
    C_grid = interpolate.griddata((L,H),C,(L_grid,H_grid),method='linear')
    C_grid[np.where(np.isnan(C_grid))] = 0
    # fix the edges
    C_grid[:, 0] = 0.5*(C_grid[:,1]+C_grid[:,-2])
    C_grid[:,-1] = 0.5*(C_grid[:,1]+C_grid[:,-2])

    Cmax[res_LH][gmt] = np.zeros(C_grid.shape,dtype=np.float32)
    Cmax[res_LH][gmt][:,:] = C_grid[:,:]
    if save: save_Cmax_npy(res=res_LH, gmt=gmt)
    if plot: plot_Cmax(res=res_LH, gmt=gmt)

#-------------------
# display the gamut
#-------------------

def get_extremum(res, gmt):
    """ Prints the LCH value of the colour of highest C """
    C = Cmax[res][gmt].max()
    iL,iH = np.unravel_index(Cmax[res][gmt].argmax(),Cmax[res][gmt].shape)
    nL = len(Cmax[res][gmt][:,0])
    L = L_min + iL/(nL-1.) * (L_max-L_min)
    nH = len(Cmax[res][gmt][0,:])
    H = H_min + iH/(nH-1.) * (H_max-H_min)
    return np.array((L,C,H))

def plot_Cmax(res, gmt, vmax=200, fig=1, figsize=None, dpi=None, dir=this_dir, fname="Cmax", axes=['on','off']):
    """ Plots Cmax(H,L) """
    plot2D(Cmax[res][gmt], name=gmt, vmax=vmax, fname='%s_res%i_%s'%(fname,res,gmt), fig=fig, figsize=figsize, dpi=dpi, dir=dir, axes=axes)

def plot2D(array, marker='', colour='', vmin=0, vmax=200, cbar=3, fig=1, figsize=None, dpi=None, aspect="equal", name="", fname="Cmax", dir=this_dir, axes=['on','off']):
    """ Plots a surface represented explicitly by a 2D array XY or implicitly by a set of 3D points XYZ """
    cmap = "Greys_r"
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    if dir != "":
        fname='%s/%s'%(dir,fname)
        ext = ".png"
    if fig != 0:
        plt.figure(fig,figsize=figsize)
        plt.title("%s gamut"%name)
        plt.xlabel("H")
        plt.ylabel("L", rotation='horizontal')
        plt.xlim([0,360])
        plt.ylim([0,100])
        if array.shape[1]==3:
            # array is the 3D surface of a 2D function
            L = array[:,0]
            C = array[:,1]
            H = array[:,2]
            plt.tricontourf(H,L,C, cmap=cmap, norm=norm)
            if marker != '':
                ax = plt.gca()
                if len(colour)>0:
                    ax.plot(H,L,marker,c=colour)
                else:
                    #ax.plot(H,L,marker,color=convert.clip3(convert.LCH2RGB(L,C,H)).tolist())
                    for h, l, c in zip(H, L, array): ax.plot(h,l,marker,color=convert.clip3(convert.LCH2RGB(c[0],c[1],c[2])))
            plt.gca().set_aspect(aspect)
        else:
            # array is a 2D map
            plt.imshow(array, origin='lower', extent=[H_min, H_max, L_min, L_max], aspect=aspect, interpolation='nearest', cmap=cmap, norm=norm)
        locator = ticker.MultipleLocator(60)
        plt.gca().xaxis.set_major_locator(locator)
        if cbar>0:
            cax = make_axes_locatable(plt.gca()).append_axes("right", size="%.f%%"%cbar, pad=0.10)
            cb = plt.colorbar(plt.gci(), cax=cax)
            cb.locator = ticker.MultipleLocator(50)
            cb.update_ticks()
            cb.set_label("Cmax")
        if dir != "" and 'on' in axes:
            print("writing %s"%(fname+"_axon"+ext))
            plt.savefig(fname+"_axon"+ext, dpi=dpi, bbox_inches='tight')
    if dir != "" and 'off' in axes and array.shape[1] > 3:
        print("writing %s"%(fname+"_axoff"+ext))
        plt.imsave(arr=array, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, fname=fname+"_axoff"+ext, dpi=dpi if dpi!=None else 200)
        #plt.imsave(fname+"_axoff"+ext, plt.get_cmap(cmap)(norm(np.flipud(array))), dpi=dpi if dpi!=None else 200)

def plot3D(RGB, angle=(0,0), fig=0, figsize=None, dpi=None, dir="", fname="RGB"):
    """ Plots a set of (R,G,B) points in 3D
        (beware: mplot3d does not composite colours correctly, and cannot handle large sets)
    """
    # figure
    fg = plt.figure("RGB",figsize=figsize)
    ax = fg.add_subplot(111, projection='3d')
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(angle[0],angle[1])
    #ax.grid(False)
    # plot
    R,G,B = RGB
    RGB_list = np.stack((R,G,B),axis=-1)
    print(len(RGB_list)," points")
    ax.scatter(R,G,B,color=RGB_list,marker='o',depthshade=False)
    # save
    if dir != "":
        fname = "%s/%s_%s.png"%(dir,fname,space)
        print("writing %s"%(fname))
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    if fig<0: plt.close(fg)

#-------------------------
# save and load the gamut
#-------------------------

def save_Cmax_npy(res, gmt, dir=this_dir):
    """ Saves a gamut as a numpy binary file """
    global Cmax
    fname = '%s/Cmax_res%.0f_%s.npy'%(dir,res,gmt)
    print("saving gamut to %s"%fname)
    np.save(fname, Cmax[res][gmt])

def load_Cmax_npy(res, gmt, dir=this_dir):
    """ Loads a gamut from a numpy binary file """
    global Cmax
    fname = '%s/Cmax_res%.0f_%s.npy'%(dir,res,gmt)
    print("loading gamut from %s"%fname)
    if res not in Cmax.keys(): Cmax[res] = {}
    Cmax[res][gmt] = np.load(fname)

def save_Cmax_txt(res, gmt, dir=this_dir):
    """ Saves a gamut as a text file """
    global Cmax
    fname = '%s/Cmax_res%.0f_%s.txt'%(dir,res,gmt)
    file = open(fname, 'w')
    print("saving gamut to %s"%fname)
    L = np.linspace(L_min,L_max,int((L_max-L_min)*res+1))
    H = np.linspace(H_min,H_max,int((H_max-H_min)*res+1))
    digits = np.ceil(np.log10(res))
    format = "%%%i.%if"%(4+digits,digits)
    formats = format+"\t"+format+"\t"+format+"\n"
    for i in range(len(L)):
        for k in range(len(H)):
            file.write(formats%(L[i],H[k],Cmax[res][gmt][i,k]))
    file.close()

def load_Cmax_txt(res, gmt, dir=this_dir):
    """ Loads a gamut from a text file (as written by save_Cmax_txt()) """
    global Cmax
    Cmax[res] = {}
    fname = '%s/Cmax_res%.0f_%s.txt'%(dir,res,gmt)
    file = open(fname, 'r')
    print("loading gamut from %s"%fname)
    L = np.linspace(L_min,L_max,int((L_max-L_min)*res+1))
    H = np.linspace(H_min,H_max,int((H_max-H_min)*res+1))
    Cmax[res][gmt] = np.zeros((len(L),len(H)),dtype=np.float32)
    for i in range(len(L)):
        for k in range(len(H)):
            Cmax[res][gmt][i,k] = float(file.readline().strip("\n").split("\t")[-1])
    file.close()

#---------------
# use the gamut
#---------------

def set_Cmax(res,gmt):
    """ Loads or computes a gamut as needed (only needed once) """
    global Cmax
    if res in Cmax.keys() and gmt in Cmax[res].keys(): return
    try:
        load_Cmax_npy(res,gmt)
    except:
        print("couldn't load gamut '%s' at res=%f, computing it"%(gmt,res))
        find_Cmax_forward(res, gmt)

def Cmax_for_LH(L,H,res=1,gmt='full'):
    """ Returns the maximum C for a given pair (L,H)
        at a given resolution in a given gamut """
    global Cmax
    set_Cmax(res,gmt) # the gamut array is cached
    H = H%360
    L_valid = np.logical_and(L>=0, L<=100)
    C = np.where(L_valid,interpolate_Cmax_for_LH(L,H,Cmax[res][gmt]),np.nan)
    return C

def interpolate_Cmax_for_LH(L,H,Cmax):
    """ Bi-linearly interpolates tabulated Cmax(L,H) at given L,H
        (expects L in [L_min,L_max] = [0,100] and H in [H_min,Hmax] = [0,360])
    """
    # L
    nL = Cmax.shape[0]
    i = (L-L_min)/float(L_max-L_min) * (nL-1)
    i0 = (np.floor(i)).astype(int)
    i0 = np.maximum(np.minimum(i0,nL-1),0)
    i1 = np.where(i0 < nL-1, i0 + 1, i0)
    x = i - i0
    # H
    nH = Cmax.shape[1]
    j = (H-H_min)/float(H_max-H_min) * (nH-1)
    j = np.where(np.isnan(j),0,j) # to avoid casting error
    j0 = (np.floor(j)).astype(int)
    j0 = np.maximum(np.minimum(j0,nH-1),0)
    j1 = np.where(j0 < nH-1, j0 + 1, j0)
    y = j - j0
    # C (bilinear interpolation)
    C = Cmax[i0,j0] * (1-x)*(1-y) \
      + Cmax[i0,j1] * (1-x)*   y  \
      + Cmax[i1,j0] *    x *(1-y) \
      + Cmax[i1,j1] *    x *   y
    return C
