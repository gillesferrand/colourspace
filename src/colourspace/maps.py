# -*- coding: utf-8 -*-
"""
Generation of custom colour maps:
- equi-luminant (1D) stepping in H, at Cmax
- diverging     (1D) stepping in C or (2D) stepping in C for each L, from one hue to another
- mono-hue      (1D) stepping in L, at Cmax
make_cmap_favs() generates and writes a set of predefined maps.

For all three cmap making functions:
- 'res' is the number of steps per unit of the quantity L,C,H
- if 'mode' is 'crop' then colours out of the sRGB gamut are discarded,
  if 'mode' is 'clip' then invalid R,G,B values are just clipped
  ('clip' cmaps are more vivid, but less uniform, than 'crop' cmaps)
- 'targets' are
  'mpl' to generate a cmap for Matplotlib (stored in the CMAP dictionnary)
  'png' to write the RGB array as an image file
Matplotlib cmaps can be plotted and written to disk (as png files of normalized width) with plot_cmaps(),
and can be tested on dummy data with test_cmaps()
"""

from __future__ import print_function
import os
import numpy as np
import pylab as plt
import matplotlib
from packaging.version import Version
try:
    import pyx
except:
    pass
from . import convert
from . import gamut

CMAP = {}     # cache for Matplotlib cmaps
CMAP_PyX = {} # cache for PyX cmaps

# wrappers for the gamut functions at the chosen resolution
res_gamut = 10
Cmax_for_LH = {}
Cmax_for_LH['crop'] = lambda l,h: gamut.Cmax_for_LH(l,h,res=res_gamut,gmt="sRGB")
Cmax_for_LH['clip'] = lambda l,h: gamut.Cmax_for_LH(l,h,res=res_gamut,gmt="full")

def hue(H):
    return H if 0 <= H <= 360 else H%360

#--------------
# sample cmaps
#--------------

# equilum

def make_cmap_equilum(L=70, H=[0,250], Hres=1, modes=['clip','crop'], sym=True, targets=['mpl','png'], mpl_reg=False, png_dir=".", out=False):
    """ Draws a line at constant L in the LH plane, in the chosen H range, at the Cmax for this L
        (if sym==True then Cmax is set for all hues, otherwise for each hue independently)
    """
    #print("drawing path at L = %3i for H = %3i – %3i with the Cmax for this L"%(L,H[0],H[1]))
    if H[0] <= H[1]:
        H_range = np.linspace(H[0],H[1],int((H[1]-H[0])*Hres+1))
    else:
        H_range1 = np.linspace(H[0],360 ,int((360 -H[0])*Hres+1))
        H_range2 = np.linspace(0   ,H[1],int((H[1]-0   )*Hres+1))
        H_range = np.concatenate((H_range1, H_range2))
    Cmax = {}
    RGB = {}
    for mode in modes:
        Cmax[mode] = Cmax_for_LH[mode](L,H_range) # the Cmax for each hue
        if sym: Cmax[mode] = Cmax[mode].min() # the Cmax for all hues
        RGB [mode] = convert.clip3(convert.LCH2RGB(L,Cmax[mode],H_range))
        name = 'equilum_L%03i_H%03i-%03i_%s'%(L,hue(H[0]),hue(H[1]),mode)
        generate_cmaps(RGB[mode], name, targets, mpl_reg=mpl_reg, png_dir=png_dir)
    if out: return RGB

# diverging

def make_cmap_diverging(H1=30+180, H2=30, L=50, modes=['clip','crop'], sym=True, Cres=1, targets=['mpl','png'], mpl_reg=False, png_dir=".", out=False):
    """ For a given L, draws a path from H1 at max chroma to H2 at max chroma
        (if sym==True then Cmax is set for both hues, otherwise for each hue independently)
    """
    RGB = {}
    for mode in modes:
        Cmax1 = Cmax_for_LH[mode](L,H1) # the Cmax for H1
        Cmax2 = Cmax_for_LH[mode](L,H2) # the Cmax for H2
        Cmax12 = np.minimum(Cmax1, Cmax2) # the Cmax that accomodates both hues
        #print("L = ",L," : Cmax1 = ",Cmax1,", Cmax2 =",Cmax2)
        C_range1  = np.linspace(0, Cmax1 , int(Cmax12*Cres+1))
        C_range2  = np.linspace(0, Cmax2 , int(Cmax12*Cres+1))
        C_range12 = np.linspace(0, Cmax12, int(Cmax12*Cres+1))
        if sym:
            RGB12 = convert.clip3(convert.LCH2RGB(L,C_range12,H1)) # H1 side, restricted to Cmax(H2)
            RGB21 = convert.clip3(convert.LCH2RGB(L,C_range12,H2)) # H2 side, restricted to Cmax(H1)
            RGB[mode] = np.concatenate((RGB12[::-1], RGB21[1:]))
        else:
            RGB1  = convert.clip3(convert.LCH2RGB(L,C_range1 ,H1)) # H1 side, full range
            RGB2  = convert.clip3(convert.LCH2RGB(L,C_range2 ,H2)) # H2 side, full range
            RGB[mode] = np.concatenate((RGB1 [::-1], RGB2 [1:]))
        name = 'diverging_L%03i_H%03i-%03i_%s'%(L,hue(H1),hue(H2),mode)
        generate_cmaps(RGB[mode], name, targets, mpl_reg=mpl_reg, png_dir=png_dir)
    if out: return RGB

def make_cmap_diverging2D(H1=30+180, H2=30, L=[0,100], Lres=1, modes=['clip','crop'], sym=True, Csteps=128, png_dir=".", png_prefix="cmap", out=False):
    """ For a given range of L, stitches the half planes H=H1 and H=H2 along the gray line, each extended to the maximal chroma
        (if sym==True then Cmax is set for both hues, otherwise for each hue independently)
    """
    if len(L)==0: return
    if len(L)==1: L=[L[0],L[0]]
    L_range = np.linspace(L[0],L[1],int(abs(L[1]-L[0])*Lres)+1)
    C_range = np.linspace(0, 1, int(Csteps+1)) # normalized to Cmax for this L
    LL, CC_ = np.meshgrid(L_range,C_range,indexing='ij')
    RGB = {}
    for mode in modes:
        Cmax1 = Cmax_for_LH[mode](L_range,H1) # the Cmax for H1
        Cmax2 = Cmax_for_LH[mode](L_range,H2) # the Cmax for H2
        Cmax12 = np.minimum(Cmax1, Cmax2) # the Cmax that accomodates both hues
        if sym:
            CC = CC_ * np.repeat(Cmax12[:,np.newaxis],Csteps+1,axis=1)
            RGB12 = convert.clip3(convert.LCH2RGB(LL,CC,H1)) # H1 side, restricted to Cmax(H2)
            RGB21 = convert.clip3(convert.LCH2RGB(LL,CC,H2)) # H2 side, restricted to Cmax(H1)
            RGB[mode] = np.concatenate((RGB12[:,::-1,:], RGB21[:,1:,:]), axis=1)
        else:
            CC = CC_ * np.repeat(Cmax1[:,np.newaxis],Csteps+1,axis=1)
            RGB1  = convert.clip3(convert.LCH2RGB(LL,CC,H1)) # H1 side, full range
            CC = CC_ * np.repeat(Cmax2[:,np.newaxis],Csteps+1,axis=1)
            RGB2  = convert.clip3(convert.LCH2RGB(LL,CC,H2)) # H2 side, full range
            RGB[mode] = np.concatenate((RGB1 [:,::-1,:], RGB2 [:,1:,:]), axis=1)
        name = 'diverging2D_L%03i-%03i_H%03i-%03i_%s'%(L_range[0],L_range[-1],hue(H1),hue(H2),mode)
        fullname = png_dir+"/"+png_prefix+"_"+name+".png"
        if not os.path.exists(png_dir): os.makedirs(png_dir)
        write_RGB_as_PNG(RGB[mode], fname=fullname)
    if out: return RGB

# monohue

def make_cmap_monohue(H=0, L=[0,50], Lres=1, modes=['clip','crop'], sym=False, targets=['mpl','png'], mpl_reg=False, png_dir=".", out=False):
    """ For a given H, draws a path from L[0] to L[1] at the maximal C
        (if sym==True then Cmax is set for both L and 100-L, otherwise for each L independently)
    """
    if len(L)<2: return
    L_range = np.linspace(L[0],L[1],int(abs(L[1]-L[0])*Lres+1))
    Cmax = {}
    RGB = {}
    for mode in modes:
        if sym: Cmax_func = lambda l: np.minimum(Cmax_for_LH[mode](l,H), Cmax_for_LH[mode](100-l,H)) # the Cmax for (L,H) and (100-L,H)
        else:   Cmax_func = lambda l: Cmax_for_LH[mode](l,H)                                         # the Cmax for (L,H)
        #Cmax[mode] = Cmax_func(L_range)
        #print("drawing %s path from (%i, %i, %i) to (%i, %i, %i)"%(mode,L_range[0],Cmax[mode][0],H,L_range[-1],Cmax[mode][-1],H))
        RGB[mode] = convert.clip3(convert.LCH2RGB(L_range,Cmax_func(L_range),H))
        name = 'monohue_L%03i-%03i_H%03i_%s'%(L[0],L[1],hue(H),mode)
        generate_cmaps(RGB[mode], name, targets, mpl_reg=mpl_reg, png_dir=png_dir)
    if out: return RGB

# selection

def make_cmap_favs(types=['equilum','diverging','monohue'], modes=['clip','crop'], targets=['mpl','png'], mpl_reg=False, dir='.', plot=True):
    """ Generates and plots a selection of colour maps of different types (for mpl and as png) """
    global CMAP
    if 'equilum' in types:
        print("-------")
        print("equilum")
        print("-------")
        CMAP = {}
        for mode in modes:
            if 'png' in targets:
                for L in np.arange(20,90,10):
                    make_cmap_equilum(L=L, H=[0,250], Hres=1, modes=[mode], targets=['png'], png_dir=dir)
            if 'mpl' in targets:
                for L in np.arange(20,90,10):
                    make_cmap_equilum(L=L, H=[0,250], Hres=2, modes=[mode], targets=['mpl'], mpl_reg=mpl_reg)
                if plot: plot_cmaps(title="equilum_%s colour maps"%mode, fig=0, dir=dir)
    if 'diverging' in types:
        print("---------")
        print("diverging")
        print("---------")
        sym = True
        CMAP = {}
        for mode in modes:
            if 'png' in targets:
                for L in np.arange(20,90,10):
                    make_cmap_diverging(H1=30+180, H2=30, L=L, Cres=1, modes=[mode], sym=sym, targets=['png'], png_dir=dir)
                make_cmap_diverging2D(H1=30+180, H2=30, L=[0,100], Lres=1, Csteps=128, modes=[mode], sym=sym, png_dir=dir)
            if 'mpl' in targets:
                for L in np.arange(20,90,10):
                    make_cmap_diverging(H1=30+180, H2=30, L=L, Cres=4, modes=[mode], sym=sym, targets=['mpl'], mpl_reg=mpl_reg)
                if plot: plot_cmaps(title="diverging_%s colour maps"%mode, fig=0, dir=dir)
    if 'monohue' in types:
        print("-------")
        print("monohue")
        print("-------")
        sym = True
        CMAP = {}
        for mode in modes:
            if 'png' in targets:
                for H in 40 + 60*np.arange(6):
                    make_cmap_monohue(H=H, L=[  0, 50], Lres=1 , sym=sym, modes=[mode], targets=['png'], png_dir=dir)
                    make_cmap_monohue(H=H, L=[100, 50], Lres=1 , sym=sym, modes=[mode], targets=['png'], png_dir=dir)
                    make_cmap_monohue(H=H, L=[0  ,100], Lres=1 , sym=sym, modes=[mode], targets=['png'], png_dir=dir)
            if 'mpl' in targets:
                for H in 40 + 60*np.arange(6):
                    make_cmap_monohue(H=H, L=[  0, 50], Lres=10, sym=sym, modes=[mode], targets=['mpl'], mpl_reg=mpl_reg)
                    make_cmap_monohue(H=H, L=[100, 50], Lres=10, sym=sym, modes=[mode], targets=['mpl'], mpl_reg=mpl_reg)
                    make_cmap_monohue(H=H, L=[0  ,100], Lres= 5, sym=sym, modes=[mode], targets=['mpl'], mpl_reg=mpl_reg)
                if plot: plot_cmaps(title="monohue_%s colour maps"%mode, fig=0, dir=dir)

#---------------
# generic cmaps
#---------------

def make_cmap_segmented(LCH_x, LCH_y, name="segmented", modes=['clip','crop'], targets=['mpl','png'], mpl_reg=False, png_dir=".", out=False):
    """ Makes a cmap from paths linear by part in L, C, H coordinates
        LCH_x[X] is an array of input values in [0,1]
        LCH_y[X] is an array of output values in the range of coord X
        for X = L,C,H
        (y values can be discontinuous at a point x, given by a pair [V1,V2])
    """
    # checks
    for coord in ['L','C','H']:
        if len(LCH_x[coord])<2 \
        or LCH_x[coord][0] != 0 or LCH_x[coord][-1] != 1 \
        or (len(LCH_x[coord])>2 and (np.array(LCH_x[coord][1:-1])-np.array(LCH_x[coord][0:-2])).min() < 0):
            print("Invalid range for %s: must be monotonous from 0 to 1"%coord)
            return None
        if len(LCH_x[coord]) != len(LCH_y[coord]):
            print("The x and y arrays must be of the same length")
            return None
    # draw paths
    LCH_range = {}
    get_value = lambda value, side: value[side] if type(value)==list else value
    for coord in ['L','C','H']:
        coord_range = np.array([])
        for i in range(len(LCH_x[coord])-1):
            n_points = int((LCH_x[coord][i+1]-LCH_x[coord][i]) * 1024)
            segment = np.linspace(get_value(LCH_y[coord][i],1), get_value(LCH_y[coord][i+1],0), n_points)
            coord_range = np.concatenate((coord_range, segment))
        LCH_range[coord] = coord_range
    # Note: the position of the boundaries is only controlled +/- 1
    # but such shifts are negligible as long as the total number of points is large enough
    n = min([len(LCH_range['L']),len(LCH_range['C']),len(LCH_range['H'])])
    L_range = LCH_range['L'][0:n]
    C_range = LCH_range['C'][0:n]
    H_range = LCH_range['H'][0:n]
    # make cmap
    Cmax = {}
    RGB = {}
    for mode in modes:
        Cmax[mode] = Cmax_for_LH[mode](L_range,H_range) # the Cmax for each (L,H) pair
        RGB[mode] = convert.clip3(convert.LCH2RGB(L_range,np.minimum(C_range,Cmax[mode]),H_range))
        generate_cmaps(RGB[mode], name+"_"+mode if len(modes)>1 else name, targets, mpl_reg=mpl_reg, png_dir=png_dir)
    if out: return RGB

def make_cmap_path(L,C,H, name="custom", modes=['clip','crop'], targets=['mpl','png'], mpl_reg=False, png_dir=".", out=False):
    """ Makes a cmap from an arbitrary path in LCH space, defined by the sets of L, C, H values along the path """
    Cmax = {}
    RGB = {}
    for mode in modes:
        Cmax[mode] = Cmax_for_LH[mode](L,H) # the Cmax for each (L,H) pair
        RGB[mode] = convert.clip3(convert.LCH2RGB(L,np.minimum(C,Cmax[mode]),H))
        generate_cmaps(RGB[mode], name+"_"+mode if len(modes)>1 else name, targets, mpl_reg=mpl_reg, png_dir=png_dir)
    if out: return RGB

#-----------------
# cmap generation
#-----------------

def generate_cmaps(RGB_list, name, targets=['mpl'], mpl_rev=True, mpl_reg=False, png_height=32, png_prefix="cmap", png_dir="."):
    """ Generates colour maps from a 1D array of RGB triplets, for the specified targets:
        PNG (written to disk), Matplotlib (cached in CMAP), or PyX (cached in CMAP_PyX) """
    for target in targets:
        if target == 'png':
            if not os.path.exists(png_dir): os.makedirs(png_dir)
            RGB_array = np.tile(RGB_list, (png_height,1,1))
            fname = png_dir+"/"+png_prefix+"_"+name+".png"
            write_RGB_as_PNG(RGB_array, fname)
        if target.lower() == 'mpl':
            print("creating cmap '%s' for Matplotlib (%4i steps)"%(name,len(RGB_list)))
            CMAP[name] = matplotlib.colors.ListedColormap(RGB_list, name)
            if mpl_rev: CMAP[name+'_r'] = matplotlib.colors.ListedColormap(RGB_list[::-1], name)
            if mpl_reg: register_to_mpl([name], mpl_rev)
        if target.lower() == 'pyx':
            print("creating cmap '%s' for PyX (%4i steps)"%(name,len(RGB_list)))
            def make_function(RGB_list,i):
                return lambda x: RGB_list[int(np.floor(x*(len(RGB_list)-1)))][i]
            CMAP_PyX[name     ] = pyx.color.functiongradient({'r':make_function(RGB_list,0),
                                                              'g':make_function(RGB_list,1),
                                                              'b':make_function(RGB_list,2)}, pyx.color.rgb)
            CMAP_PyX[name+'_r'] = pyx.color.functiongradient({'r':make_function(RGB_list[::-1],0),
                                                              'g':make_function(RGB_list[::-1],1),
                                                              'b':make_function(RGB_list[::-1],2)}, pyx.color.rgb)

def write_RGB_as_PNG(arr, fname):
    """ writes a RGB array as a PNG file, at its intrinsic size """
    print('writing %s (%ix%i)'%(fname,arr.shape[0],arr.shape[1]))
    plt.imsave(arr=arr, fname=fname, origin='lower')

def register_to_mpl(names, reversed=True):
    """ Adds a cmap to Matplotlib's internal list """
    for name in names:
        print("registering cmap '%s' to Matplotlib"%(name))
        if Version(matplotlib.__version__) < Version('3.8'):
            matplotlib.cm.register_cmap(cmap=CMAP[name], name=name)
            if reversed: matplotlib.cm.register_cmap(cmap=CMAP[name+'_r'], name=name+'_r')
        else:
            matplotlib.colormaps.register(cmap=CMAP[name], name=name)
            if reversed: matplotlib.colormaps.register(cmap=CMAP[name+'_r'], name=name+'_r')

#---------------
# cmap plotting
#---------------

def get_cmap(name, nsteps=None):
    """ Returns a named colour map, looking first in the local cache CMAP, then in Matplotlib's registered colours
        optionally resampled in `nsteps` bins
    """
    if name in CMAP.keys():
        cmap = CMAP[name]
    else:
        try:
            cmap = matplotlib.cm.get_cmap(name)
        except:
            print("Unknown cmap: ",name)
            cmap = None
    if cmap != None and nsteps != None and nsteps>0: 
        if hasattr(cmap, '_resample'): cmap = cmap._resample(nsteps)
        if hasattr(cmap, 'resampled'): cmap = cmap.resampled(nsteps)
    return cmap

def plot_cmaps(names=[], filters=[], reverse=False, nsteps=None, width=256, height=32, fig=1, figsize=None, dpi=None, frame=False, labels="left", labelsize=10, title="", titlesize=14, dir=".", fname_all="cmaps", fname="cmap"):
    """ Plots all colour maps listed by name
        If `fname` is set writes them individually as PNG images of size `width` by `height`
        If `fname_all` is set writes the figure with all of them as a PNG image
    """
    # adapted from http://matplotlib.org/examples/color/colormaps_reference.html
    if len(names)==0: names = list_all(filters=filters, reverse=reverse)
    nrows = len(names)
    if nrows == 0: return
    plt.close(fig)
    fig, axes = plt.subplots(num=fig, nrows=nrows, figsize=figsize)
    if not hasattr(axes, "__len__"): axes = [axes]
    # adjust layout
    fig_w, fig_h = fig.get_size_inches()*72 # size in points
    pad = 0.05
    left   = 0 + pad*min(fig_w,fig_h)/fig_w
    right  = 1 - pad*min(fig_w,fig_h)/fig_w
    bottom = 0 + pad*min(fig_w,fig_h)/fig_h
    top    = 1 - pad*min(fig_w,fig_h)/fig_h
    wspace = 0
    hspace = (labelsize/fig_h)*nrows #pad*nrows
    if labels=="left"  : left   += (labelsize*15/fig_w)
    if labels=="right" : right  -= (labelsize*15/fig_w)
    if labels=="bottom": bottom += (labelsize/fig_h) ; hspace += (labelsize/fig_h)*nrows
    if labels=="top"   : top    -= (labelsize/fig_h) ; hspace += (labelsize/fig_h)*nrows
    if title!="": top -= titlesize/fig_h
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    fig.suptitle(title, fontsize=titlesize)
    gradient = np.linspace(0, 1, int(width))
    for ax, name in zip(axes, names):
        cmap = get_cmap(name, nsteps)
        if cmap!=None:
            ax.imshow(np.tile(gradient,(2,1)), aspect='auto', interpolation='nearest', cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        if not frame:
            for pos in ['top','right','bottom','left']: ax.spines[pos].set_visible(False)
        if labels=="left":
            ax.set_ylabel(name, fontsize=labelsize, family='monospace', rotation='horizontal', ha='right', va='center')
        if labels=="right":
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(name, fontsize=labelsize, family='monospace', rotation='horizontal', ha='left' , va='center')
        if labels=="bottom":
            ax.set_xlabel(name, fontsize=labelsize, family='monospace', rotation='horizontal', ha='center', va='top')
        if labels=="top":
            ax.set_xlabel(name, fontsize=labelsize, family='monospace', rotation='horizontal', ha='center', va='bottom')
            ax.xaxis.set_label_position("top")
        if fname != "" and cmap!=None:
            fullname = "%s/%s%i_%s.png"%(dir,fname,width,name)
            print('writing ',fullname)
            plt.imsave(arr=np.tile(gradient,(height,1)), origin='lower', fname=fullname, cmap=cmap, dpi=dpi if dpi!=None else 200)
    if fname_all != "":
        fullname = "%s/%s%s.png"%(dir,fname_all,"_"+title if title!="" else "")
        print('writing ',fullname)
        plt.savefig(fullname, dpi=dpi, bbox_inches='tight')
    if fig==0: plt.close(fig)

from mpl_toolkits.axes_grid1 import make_axes_locatable

def test_cmaps(data=[], names=[], titles=[], filters=[], reverse=False, nsteps=None, subplots=(1,1), figsize=None, dpi=None, titlesize=12, dir=".", fname="testcmap"):
    """ Displays dummy 2D data with all the colour maps listed by name """
    if len(data)==0: data = mock_data(f_x=1, phi_x=-0.5, f_y=1, phi_y=0, min=0, max=1)
    if len(names)==0: names = list_all(filters=filters, reverse=reverse)
    if len(titles)==0: titles = [""]*len(names)
    if nsteps==None: nsteps = [None]*len(names)
    if len(titles)!=len(names): print("You must define title for each cmap")
    if len(nsteps)!=len(names): print("You must define nstep for each cmap")
    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    for i in range(len(names)):
        cmap = get_cmap(names[i], nsteps[i])
        if cmap==None: continue
        if subplots[0]*subplots[1]>1: plt.axes(axes[i])
        else: plt.axes(axes)
        im = plt.imshow(data, aspect='equal', interpolation='nearest', cmap=cmap)
        title_i = names[i] if titles[i]=='' else titles[i]
        plt.title(title_i, fontsize=titlesize)
        plt.xticks([])
        plt.yticks([])
        #using an axis divider so that the colour bar always be of the same height as the plot
        cax = make_axes_locatable(plt.gca()).append_axes("right",size="5%",pad=0.25)
        cbar = plt.colorbar(im, cax=cax)
        if fname != "" and i==len(names)-1:
            fullname = '%s/%s'%(dir,fname)
            if len(names)==1: fullname += '_%s'%(names[i])
            fullname += '.png'
            print('writing ',fullname)
            plt.savefig(fullname, dpi=dpi, bbox_inches='tight')

def mock_data(f_x=1, phi_x=-0.5, f_y=1, phi_y=0, min=0, max=1, res=300):
    """ Generates a 2D periodic pattern (f=frequency, phi=phase), rescaled to [min,max] """
    x = np.arange(0, 1, 1./res)
    y = np.arange(0, 1, 1./res)
    X,Y = np.meshgrid(x,y)
    Z = np.sin((f_x*X+phi_x)*np.pi)*np.sin((f_y*Y+phi_y)*np.pi)
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    Z = min + Z * (max-min)
    return Z

def list_all(filters=[], reverse=False):
    """ Lists names of all the colour maps present in CMAP """
    names = []
    for name in CMAP.keys():
        if all(x in name for x in filters):
            if name[-2:] != '_r' or reverse == True: names.append(name)
    names = sorted(names, key=rank_cmap)
    print('found cmaps: ',names)
    return names

def rank_cmap(name):
    """ Ranks a cmap by name for custom ordering """
    # reverse the order of L
    pos = name.find("_L")
    if pos>0: name = name.replace(name[pos:pos+2+3],"_L%03i"%(100-int(name[pos+2:pos+2+3])))
    return name

#-------------
# cmap curves
#-------------

def plot_path(cmap, nsteps=None, dir=".", fname="", colours=[], widths=[], styles=[], markers=[], Cmax_ls=':', H_patch='cut', C_tol=0.01, axes=[], figsize=(4,), dpi=None, stack='Z', Z_axes='left', Z_pad=0, Z_margin=0, space='LCH', xlim=[0,1], ylim=None, xticks=0.25, yticks=None, title="", legend_label="", legend_axis=0, legend_loc=None, cmap_size="0%", cmap_pad=0.3):
        """ Plots the path of a Matplotlib `cmap` in the three dimensions of the colour space
            Extracts the curves R,G,B or L,C,H according to `space', then calls plot_3curves(curves,`space`,`stack`)
            Optionally adds an image of the cmap itself as the abscissa
        """
        if isinstance(cmap, str): cmap = get_cmap(cmap, nsteps=nsteps)
        # get the curves
        if hasattr(cmap,'colors'): # ListedColormap
            RGB = np.array(cmap.colors)
        else: # LinearSegmentedColormap
            RGB = cmap(np.linspace(0,1,100))
        R = RGB[:,0]
        G = RGB[:,1]
        B = RGB[:,2]
        if space == 'RGB':
            curves = (R, G, B)
        if space == 'LCH':
            curves = convert.RGB2LCH(R,G,B)
            # patch H where C=0 (really H is undefined)
            where_C_zero = np.where(curves[1]<=C_tol)[0]
            if H_patch == 'cut': 
                curves[2][where_C_zero] = np.nan
            if H_patch[:3] == 'int': 
                if len(where_C_zero)>0 and len(where_C_zero)<len(curves[1]):
                    where_C_zero_chunks = np.split(where_C_zero, np.where(where_C_zero[1:] != where_C_zero[:-1] + 1)[0] + 1)
                    for j in range(len(where_C_zero_chunks)): 
                        if where_C_zero_chunks[j][ 0]>0 and where_C_zero_chunks[j][-1]<len(curves[2])-1:
                            iL = where_C_zero_chunks[j][ 0]-1
                            iR = where_C_zero_chunks[j][-1]+1
                        if where_C_zero_chunks[j][ 0]==0: 
                            iL = len(curves[2])-1 if len(curves[2])-1 not in where_C_zero else where_C_zero_chunks[-1][ 0]-1
                            iR = where_C_zero_chunks[j][-1]+1
                        if where_C_zero_chunks[j][-1]==len(curves[2])-1:
                            iL = where_C_zero_chunks[j][ 0]-1
                            iR =  0 if 0 not in where_C_zero else where_C_zero_chunks[ 0][-1]+1
                        #print(iL,"C = ",curves[2][iL])
                        for i in where_C_zero_chunks[j]: 
                            if where_C_zero_chunks[j][ 0]>0 and where_C_zero_chunks[j][-1]<len(curves[2])-1:
                                frac = (i-iL)/(iR-iL)
                            if where_C_zero_chunks[j][ 0]==0: 
                                frac = (len(curves[2])-iL+i)/(len(curves[2])-iL+iR)
                            if where_C_zero_chunks[j][-1]==len(curves[2])-1: 
                                frac = (i+1-iL)/(len(curves[2])-iL+iR)
                            #print(i,"C = ",curves[2][i]," -> ",curves[2][iL],"+",frac,"*",(curves[2][iR]-curves[2][iL])," = ",curves[2][iL]+frac*(curves[2][iR]-curves[2][iL]))
                            curves[2][i] = curves[2][iL] + frac * (curves[2][iR]-curves[2][iL])
                        #print(iR,"C = ",curves[2][iR])
        # plot the curves
        axes = plot_3curves(curves, colours=colours, widths=widths, styles=styles, markers=markers, Cmax_ls=Cmax_ls, axes=axes, figsize=figsize, stack=stack, Z_axes=Z_axes, Z_pad=Z_pad, Z_margin=Z_margin, space=space, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks, title=title, legend_label=legend_label, legend_axis=legend_axis, legend_loc=legend_loc)
        # add the cmap itself
        if cmap_size != "0%":
            for i in range(3):
                cax = make_axes_locatable(axes[i]).append_axes("bottom",size=cmap_size,pad=cmap_pad)
                gradient = np.linspace(0, 1, 256)
                cax.imshow(np.tile(gradient,(2,1)), aspect='auto', interpolation='nearest', cmap=cmap)
                cax.set_xticks([])
                cax.set_yticks([])
                for axis in ['top','bottom','left','right']: cax.spines[axis].set_linewidth(0.)
        # save to file
        if fname != "":
            fullname = '%s/%s.png'%(dir,fname)
            print('writing ',fullname)
            plt.savefig(fullname, dpi=dpi, bbox_inches='tight')
        return axes

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_3curves(curves, colours=[], widths=[], styles=[], markers=[], Cmax_ls=':', axes=[], figsize=(4,), stack='Z', Z_axes='left', Z_pad=0, Z_margin=0, space='LCH', xlim=[0,1], ylim=None, xticks=0.25, yticks=None, title="", legend_label="", legend_axis=0, legend_loc=None):
    """ Plots the three given 1D arrays `curves`
        The figure is prepared by calling setfig_3curves(`space`,`stack`)
        If `Cmax_ls` is set, the sRGB gamut is overplotted on the C plot with this linestyle
    """
    # checks
    if len(curves)!=3:
        print("You must provide 3 arrays")
        return
    n = []
    for i in range(3): n.append(len(curves[i]))
    if n[0] != n[1] or n[1] != n[2]:
        print("The 3 arrays must be of the same size")
        return
    # styling
    label_colours = ['k']*3 if len(colours)>0 else []
    default_colours = {'RGB': ['r','g','b'], 'LCH': ['darkgoldenrod','darkcyan','darkmagenta']}
    if len(colours)==0: colours = default_colours[space]
    if len(colours)==1: colours *= 3
    if len(widths )==0: widths  = [1.5]*3
    if len(widths )==1: widths  *= 3
    if len(styles )==0: styles  = ['-']*3
    if len(styles )==1: styles  *= 3
    if len(markers)==0: markers = ['']*3
    if len(markers)==1: markers *= 3
    if len(label_colours)==0: label_colours = colours
    # axes
    if len(axes)==0: axes = setfig_3curves(figsize=figsize, stack=stack, Z_axes=Z_axes, Z_pad=Z_pad, Z_margin=Z_margin, space=space, label_colours=label_colours, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks, title=title)
    # plot
    x = np.linspace(0,1,n[0])
    for i in range(3):
        axes[i].plot(x, curves[i], c=colours[i], lw=widths[i], ls=styles[i], marker=markers[i], label=legend_label)
    # legend
    if legend_label != "":
        axes[legend_axis].legend(loc=legend_loc, shadow=False)
    # gamut
    if space == 'LCH' and Cmax_ls != '':
        Cmax = Cmax_for_LH['crop'](curves[0][:],curves[2][:])
        axes[1].plot(x, Cmax, c=colours[1], lw=widths[1], ls=Cmax_ls)
    return axes

import copy

def setfig_3curves(figsize=None, stack='Z', Z_axes='left', Z_pad=0, Z_margin=0, space='LCH', label_colours=[], xlim=[0,1], ylim=None, xticks=0.25, yticks=None, title=""):
    """ Sets up the figure with three plots
        `stack` = 'V' vertical | 'H' horizontal | 'Z' depth (for the latter, axes are on the `Z_axes` side)
        `space` = 'RGB' | 'LCH'
    """
    # axes
    if stack == 'V': rows = 3; cols = 1
    if stack == 'H': rows = 1; cols = 3
    if stack == 'Z': rows = 1; cols = 1
    if figsize!=None and len(figsize)==1: figsize = (cols*figsize[0], rows*figsize[0])
    fig, axes = plt.subplots(rows, cols, figsize=figsize)#, constrained_layout=True)
    if stack == 'Z':
        if Z_pad == 0: # auto padding
            Z_pad = 0.2 * (4/figsize[0]) # paddings were set for size=4, they should be kept constant
        delta = {'left': lambda i: 0-(2-i)*Z_pad, 'right': lambda i: 1+i*Z_pad}
        axes_root = axes
        axes = []
        for i in range(3):
            axes.append(axes_root.twinx())
            axes[i].spines[Z_axes].set_position(("axes",delta[Z_axes](i)))
            axes[i].spines[Z_axes].set_visible(True)
            axes[i].yaxis.set_label_position(Z_axes)
            axes[i].yaxis.set_ticks_position(Z_axes)
        axes_root.yaxis.set_visible(False)
        if Z_margin>0 and Z_axes=='left' : plt.subplots_adjust(left =  Z_margin)
        if Z_margin>0 and Z_axes=='right': plt.subplots_adjust(right=1-Z_margin)
    # ticks
    default_yticks = {'RGB': [0.2, 0.2, 0.2], 'LCH': [10., 20., 60.]}
    if yticks == None: yticks = [None]
    if len(yticks)==1: yticks = yticks*3
    for i in range(3):
        locator = matplotlib.ticker.MultipleLocator(base=xticks)
        axes[i].xaxis.set_major_locator(locator)
        if yticks[i] == None: yticks[i] = default_yticks[space][i]
        locator = matplotlib.ticker.MultipleLocator(base=yticks[i])
        axes[i].yaxis.set_major_locator(locator)
    # labels
    if len(label_colours)==0: label_colours = ['k']*3
    if stack == 'V' or stack == 'Z':
        for i in range(3): axes[i].set_ylabel(" "+space[i]+" ", rotation=0, fontsize=12, color=label_colours[i])
        axes[0].set_title(title)
    if stack == 'H':
        for i in range(3): axes[i].set_title(space[i], fontsize=12, color=label_colours[i])
        axes[0].set_ylabel(title)
    # ranges
    for i in range(3):
        axes[i].set_xlim(xlim)
    default_ylim_max = {'RGB': [1, 1, 1], 'LCH': [100, 100, 360]}
    if ylim == None: ylim = [0,None]
    if len(ylim)==2: ylim = [copy.deepcopy(ylim), copy.deepcopy(ylim), copy.deepcopy(ylim)] #*3
    for i in range(3):
        if ylim[i][1] == None: ylim[i][1] = default_ylim_max[space][i]
        axes[i].set_ylim(ylim[i])
    if stack == 'H' or stack == 'V': plt.tight_layout()
    return axes
