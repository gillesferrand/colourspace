# -*- coding: utf-8 -*-
"""
Limits of the full human gamut, or the sRGB gamut, in CIE LCH space: Cmax(L,H)
"""

import sys
import numpy as np
import pylab as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
#import colour # http://colour-science.org
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

def valid_LCH_full(LCH):
    """ Checks if a LCH colour is in the human gamut """
    # using colour-science triangulation: not precise
    #illuminant='D65'
    #XYZ = colour.Lab_to_XYZ(colour.LCHab_to_Lab(LCH), illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant])
    #xyY = colour.XYZ_to_xyY(XYZ)
    #return colour.volume.macadam_limits.is_within_macadam_limits(xyY,illuminant)
    XYZ = convert.Lab2XYZ(convert.LCH2Lab((LCH[0],LCH[1],LCH[2])))
    return limits.within_limits(XYZ,'XYZ')

def valid_LCH_sRGB(LCH):
    """ Checks if a LCH colour is in the sRGB gamut """
    valid = lambda x: 0 <= x and x <= 1
    #illuminant='D65'
    #XYZ = colour.Lab_to_XYZ(colour.LCHab_to_Lab(LCH), illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant])
    #RGB = colour.XYZ_to_sRGB(colour.Lab_to_XYZ(colour.LCHab_to_Lab(LCH), illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]))
    #return valid(RGB[0]) and valid(RGB[1]) and valid(RGB[2])
    R,G,B = convert.LCH2RGB(LCH[0],LCH[1],LCH[2])
    return valid(R) and valid(G) and valid(B)

valid_LCH = {"full": valid_LCH_full, \
             "sRGB": valid_LCH_sRGB}

def find_Cmax_forward(res, gamut, verbose=False, save=False, plot=True):
    """ Finds the maximum Cmax for each (L,H) pair with a precision of res points per unit of L,C,H """
    if res not in Cmax.keys(): Cmax[res] = {}
    L = np.linspace(L_min,L_max,(L_max-L_min)*res+1)
    H = np.linspace(H_min,H_max,(H_max-H_min)*res+1)
    # version with loops
    Cmax[res][gamut] = np.zeros((len(L),len(H)),dtype=np.float32)
    for i in range(len(L)):
        for k in range(len(H)):
            Cmax[res][gamut][i,k] = find_Cmax_for_LH(L=L[i], H=H[k], Cres=1./res, gamut=gamut)
            if verbose:
                sys.stdout.write("L = %.2f, H = %.2f, Cmax = %.2f"%(L[i],H[k],Cmax[res][gamut][i,k]))
                sys.stdout.write("\r")
                sys.stdout.flush()
    if verbose:
        sys.stdout.write("\n")
        sys.stdout.flush()
    # version with mapping
    #LH = [(L[i], H[k]) for i in range(len(L)) for k in range(len(H))]
    #function = lambda (x,y) : find_Cmax_for_LH(L=x, H=y, res=res, validator=validator, verbose=False)
    #Cmax[res][gamut] = np.reshape(map(function,LH), (len(L),len(H)))
    if save: save_Cmax_npy(res=res, gamut=gamut)
    if plot: plot_Cmax(res=res, gamut=gamut)

def find_Cmax_for_LH(L, H, Cres, gamut):
    """Finds the maximum C for a given (L,H) at a given resolution in a given gamut"""
    if L>=0 and L<=100 and H>=0 and H<=360:
        if L>0 and L<100:
            validator = lambda c: valid_LCH[gamut]([L,c,H])
            Cmax = find_edge_by_dichotomy(validator, xmin=0, xmax=200, dx=Cres)
        else:
            Cmax = 0
    else:
        Cmax = np.nan
    return Cmax

def find_edge_by_dichotomy(func, xmin, xmax, dx=1., iter_max=100):
    """returns the point `x` (within resolution `dx`) where boolean function `func` changes value
       `func` is assumed to switch from True to False between `xmin` and `xmax`
    """
    xleft  = xmin
    xright = xmax
    xmid = 0.5*(xright-xleft)
    i = 0
    delta = dx
    while delta >= dx and i<iter_max:
        i += 1
        #print "i = %4i: x = [%6.2f, %6.2f]: func(%6.2f) = %i"%(i,xleft,xright,xmid,func(xmid)),
        if func(xmid): xleft  = xmid
        else:          xright = xmid
        xmid_old = xmid
        xmid = 0.5*(xleft+xright)
        delta = abs(xmid_old-xmid)
        #print "-> x = [%6.2f, %6.2f], delta=%f"%(xleft,xright,delta)
    if i >= iter_max: print "edge not found at precision ",dx,"in ",iter_max," iterations"
    return np.around(xmid,int(np.ceil(np.log10(1./dx))))

#---------------------------
# find the gamut boundary
# method 2: discretize the gamut boundary in the native space, project it back to the LH plane
#---------------------------

def get_RGB_faces(num=10,verbose=False):
    RGB_list = np.zeros((num**2*6,3))
    value = lambda face: 0 if '-' in face else 1
    faces = ["-R","+R","-G","+G","-B","+B"]
    for i in range(len(faces)):
        R = np.array([value(faces[i])]) if 'R' in faces[i] else np.linspace(0,1,num)
        G = np.array([value(faces[i])]) if 'G' in faces[i] else np.linspace(0,1,num)
        B = np.array([value(faces[i])]) if 'B' in faces[i] else np.linspace(0,1,num)
        for j in range(num):
            for k in range(num):
                if 'R' in faces[i]: iR = 0 ; iG = j ; iB = k
                if 'G' in faces[i]: iR = j ; iG = 0 ; iB = k
                if 'B' in faces[i]: iR = j ; iG = k ; iB = 0
                ijk = (i*num + j)*num + k
                RGB_list[ijk,:] = [R[iR], G[iG], B[iB]]
                if verbose:
                    text = "R = %4.2f, G = %4.2f, B = %4.2f (%3.0f%%)\r"%(R[iR],G[iG],B[iB],100*(ijk+1)/(num**2*6))
                    sys.stdout.write(text)
                    sys.stdout.flush()
    if verbose:
        sys.stdout.write("\n")
        sys.stdout.flush
    return RGB_list

def get_edges_LCH_sRGB(res):
    RGB = get_RGB_faces(num=res)
    LCH = np.zeros((len(RGB),3))
    for i in range(len(RGB)): LCH[i] = convert.RGB2LCH(RGB[i][0],RGB[i][1],RGB[i][2])
    return LCH

def get_edges_LCH_full(res):
    limits.set_limits(l_min=360, l_max=780, l_step=res)
    return limits.limits['cmp']['LCH']

get_edges_LCH = {"full": get_edges_LCH_full, \
                 "sRGB": get_edges_LCH_sRGB}

def find_Cmax_backward(res_native, res_LH, gamut, save=False, plot=True):
    """ Finds the maximum Cmax(L,H) by discretizing the gamut boundary in its native space
        for sRGB gamut: res_native = number of points along R, G, B
        for full gamut: res_native = delta_Lambda in nm
        the LH plane will be re-sampled regularly at resolution res_LH
    """
    if res_LH not in Cmax.keys(): Cmax[res_LH] = {}
    # get edges in LCH space
    LCH_max = get_edges_LCH[gamut](res_native)
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
    
    Cmax[res_LH][gamut] = np.zeros(C_grid.shape,dtype=np.float32)
    Cmax[res_LH][gamut][:,:] = C_grid[:,:]
    if save: save_Cmax_npy(res=res_LH, gamut=gamut)
    if plot: plot_Cmax(res=res_LH, gamut=gamut)

#-------------------
# display the gamut
#-------------------

def get_extremum(res, gamut):
    """ Prints the LCH value of the colour of highest C """
    C = Cmax[res][gamut].max()
    iL,iH = np.unravel_index(Cmax[res][gamut].argmax(),Cmax[res][gamut].shape)
    nL = len(Cmax[res][gamut][:,0])
    L = L_min + iL/(nL-1.) * (L_max-L_min)
    nH = len(Cmax[res][gamut][0,:])
    H = H_min + iH/(nH-1.) * (H_max-H_min)
    return np.array((L,C,H))

def plot_Cmax(res, gamut, vmax=200, fig=1, figsize=None, dir=this_dir, fname="Cmax", axes=['on','off']):
    """ Plots Cmax(H,L) """
    plot2D(Cmax[res][gamut], name=gamut, vmax=vmax, fname='%s_res%i_%s'%(fname,res,gamut), fig=fig, figsize=figsize, dir=dir, axes=axes)

def plot2D(array, marker='', colour='', vmin=0, vmax=200, cbar=3, fig=1, figsize=None, aspect="equal", name="", fname="Cmax", dir=this_dir, axes=['on','off']):
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
                if colour != '':
                    ax.plot(H,L,marker,c=colour)
                else:
                    for h, l, c in zip(H, L, array): ax.plot(h,l,marker,color=convert.clip3(convert.LCH2RGB(c[0],c[1],c[2])))
            plt.gca().set_aspect(aspect)
        else:
            # array is a 2D map
            plt.imshow(array, origin='lower', extent=[H_min, H_max, L_min, L_max], aspect=aspect, interpolation='nearest', cmap=cmap, norm=norm)
        if cbar>0:
            cax = make_axes_locatable(plt.gca()).append_axes("right", size="%.f%%"%cbar, pad=0.10)
            cb = plt.colorbar(plt.gci(), cax=cax)
            cb.locator = ticker.MultipleLocator(50)
            cb.update_ticks()
            cb.set_label("Cmax")
        if dir != "" and 'on' in axes:
            print "writing %s"%(fname+"_axon"+ext)
            plt.savefig(fname+"_axon"+ext, dpi=None, bbox_inches='tight')
    if dir != "" and 'off' in axes and array.shape[1] > 3:
        print "writing %s"%(fname+"_axoff"+ext)
        plt.imsave(arr=array, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, fname=fname+"_axoff"+ext)
        #plt.imsave(fname+"_axoff"+ext, plt.get_cmap(cmap)(norm(np.flipud(array))))

def plot3D(RGB_list, angle=(0,0), fig=0, figsize=None, dir="", fname="RGB"):
    """ Plots a set of RGB points in 3D
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
    print len(RGB_list)," points"
    x = RGB_list[:,0]
    y = RGB_list[:,1]
    z = RGB_list[:,2]
    ax.scatter(x,y,z,color=RGB_list,marker='o',depthshade=False)
    # save
    if dir != "":
        fname = "%s/%s_%s.png"%(dir,fname,space)
        print "writing %s"%(fname)
        plt.savefig(fname, dpi=None, bbox_inches='tight')
    if fig<0: plt.close(fg)

#-------------------------
# save and load the gamut
#-------------------------

def save_Cmax_npy(res, gamut, dir=this_dir):
    """ Saves a gamut as a numpy binary file """
    global Cmax
    fname = '%s/Cmax_res%.0f_%s.npy'%(dir,res,gamut)
    print "saving gamut to %s"%fname
    np.save(fname, Cmax[res][gamut])

def load_Cmax_npy(res, gamut, dir=this_dir):
    """ Loads a gamut from a numpy binary file """
    global Cmax
    fname = '%s/Cmax_res%.0f_%s.npy'%(dir,res,gamut)
    print "loading gamut from %s"%fname
    if res not in Cmax.keys(): Cmax[res] = {}
    Cmax[res][gamut] = np.load(fname)

def save_Cmax_txt(res, gamut, dir=this_dir):
    """ Saves a gamut as a text file """
    global Cmax
    fname = '%s/Cmax_res%.0f_%s.txt'%(dir,res,gamut)
    file = open(fname, 'w')
    print "saving gamut to %s"%fname
    L = np.linspace(L_min,L_max,(L_max-L_min)*res+1)
    H = np.linspace(H_min,H_max,(H_max-H_min)*res+1)
    digits = np.ceil(np.log10(res))
    format = "%%%i.%if"%(4+digits,digits)
    formats = format+"\t"+format+"\t"+format+"\n"
    for i in range(len(L)):
        for k in range(len(H)):
            file.write(formats%(L[i],H[k],Cmax[res][gamut][i,k]))
    file.close()

def load_Cmax_txt(res, gamut, dir=this_dir):
    """ Loads a gamut from a text file (as written by save_Cmax_txt()) """
    global Cmax
    Cmax[res] = {}
    fname = '%s/Cmax_res%.0f_%s.txt'%(dir,res,gamut)
    file = open(fname, 'r')
    print "loading gamut from %s"%fname
    L = np.linspace(L_min,L_max,(L_max-L_min)*res+1)
    H = np.linspace(H_min,H_max,(H_max-H_min)*res+1)
    Cmax[res][gamut] = np.zeros((len(L),len(H)),dtype=np.float32)
    for i in range(len(L)):
        for k in range(len(H)):
            Cmax[res][gamut][i,k] = float(file.readline().strip("\n").split("\t")[-1])
    file.close()

#---------------
# use the gamut
#---------------

def set_Cmax(res,gamut):
    """ Loads or computes a gamut as needed (only needed once) """
    global Cmax
    if res in Cmax.keys() and gamut in Cmax[res].keys(): return
    try:
        load_Cmax_npy(res,gamut)
    except:
        print "couldn't load gamut '%s' at res=%f, computing it"%(gamut,res)
        find_Cmax_forward(res, gamut)

def Cmax_for_LH(L,H,res=1,gamut='full'):
    """ Returns the maximum C for a given pair (L,H)
        at a given resolution in a given gamut """
    global Cmax
    set_Cmax(res,gamut) # the gamut array is cached
    Cmax_ = Cmax[res][gamut]
    # L
    if not(0<=L<=100):
        print "Invalid L"
        return
    nL = Cmax_.shape[0]
    i = (L-L_min)/float(L_max-L_min) * (nL-1)
    i0 = int(np.floor(i))
    i1 = i0 + 1
    if i1 > nL-1: i1 = i0
    x = i - i0
    # H
    if not(0<=H):
        print "Invalid H"
        return
    H = H%360
    nH = Cmax_.shape[1]
    j = (H-H_min)/float(H_max-H_min) * (nH-1)
    j0 = int(np.floor(j))
    j1 = j0 + 1
    if j1 > nH-1: j1 = j0
    y = j - j0
    # C (bilinear interpolation)
    C = Cmax_[i0,j0] * (1-x)*(1-y) \
      + Cmax_[i0,j1] * (1-x)*   y  \
      + Cmax_[i1,j0] *    x *(1-y) \
      + Cmax_[i1,j1] *    x *   y
    return C
