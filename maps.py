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

import os
import numpy as np
import pylab as plt
import matplotlib
from . import convert
from . import gamut

CMAP = {}

# wrappers for the gamut functions at the chosen resolution
res_gamut = 10
Cmax_for_LH = {}
Cmax_for_LH['crop'] = lambda l,h: gamut.Cmax_for_LH(l,h,res=res_gamut,gmt="sRGB")
Cmax_for_LH['clip'] = lambda l,h: gamut.Cmax_for_LH(l,h,res=res_gamut,gmt="full")

def hue(H):
    return H if 0 <= H <= 360 else H%360

# equilum

def make_cmap_equilum(L=70, H=[0,250], Hres=1, modes=['clip','crop'], sym=True, targets=['mpl','png'], png_dir=".", out=False):
    """ Draws a line at constant L in the LH plane, in the chosen H range, at the Cmax for this L
        (if sym==True then Cmax is set for all hues, otherwise for each hue independently)
    """
    #print "drawing path at L = %3i for H = %3i – %3i with the Cmax for this L"%(L,H[0],H[1])
    if H[0] <= H[1]:
        H_range = np.linspace(H[0],H[1],(H[1]-H[0])*Hres+1)
    else:
        H_range1 = np.linspace(H[0],360 ,(360 -H[0])*Hres+1)
        H_range2 = np.linspace(0   ,H[1],(H[1]-0   )*Hres+1)
        H_range = np.concatenate((H_range1, H_range2))
    Cmax = {}
    RGB = {}
    for mode in modes:
        Cmax[mode] = Cmax_for_LH[mode](L,H_range) # the Cmax for each hue
        if sym: Cmax[mode] = Cmax[mode].min() # the Cmax for all hues
        RGB [mode] = convert.clip3(convert.LCH2RGB(L,Cmax[mode],H_range))
        name = 'equilum_L%03i_H%03i-%03i_%s'%(L,hue(H[0]),hue(H[1]),mode)
        generate_cmaps(RGB[mode], name, targets, png_dir=png_dir)
    if out: return RGB

# diverging

def make_cmap_diverging(H1=30+180, H2=30, L=50, modes=['clip','crop'], sym=True, Cres=1, targets=['mpl','png'], png_dir=".", out=False):
    """ For a given L, draws a path from H1 at max chroma to H2 at max chroma
        (if sym==True then Cmax is set for both hues, otherwise for each hue independently)
    """
    RGB = {}
    for mode in modes:
        Cmax1 = Cmax_for_LH[mode](L,H1) # the Cmax for H1
        Cmax2 = Cmax_for_LH[mode](L,H2) # the Cmax for H2
        Cmax12 = np.minimum(Cmax1, Cmax2) # the Cmax that accomodates both hues
        #print "L = ",L," : Cmax1 = ",Cmax1,", Cmax2 =",Cmax2
        C_range1  = np.linspace(0, Cmax1 , Cmax1 *Cres+1)
        C_range2  = np.linspace(0, Cmax2 , Cmax2 *Cres+1)
        C_range12 = np.linspace(0, Cmax12, Cmax12*Cres+1)
        if sym:
            RGB12 = convert.clip3(convert.LCH2RGB(L,C_range12,H1)) # H1 side, restricted to Cmax(H2)
            RGB21 = convert.clip3(convert.LCH2RGB(L,C_range12,H2)) # H2 side, restricted to Cmax(H1)
            RGB[mode] = np.concatenate((RGB12[::-1], RGB21[1:]))
        else:
            RGB1  = convert.clip3(convert.LCH2RGB(L,C_range1 ,H1)) # H1 side, full range
            RGB2  = convert.clip3(convert.LCH2RGB(L,C_range2 ,H2)) # H2 side, full range
            RGB[mode] = np.concatenate((RGB1 [::-1], RGB2 [1:]))
        name = 'diverging_L%03i_H%03i-%03i_%s'%(L,hue(H1),hue(H2),mode)
        generate_cmaps(RGB[mode], name, targets, png_dir=png_dir)
    if out: return RGB

def make_cmap_diverging2D(H1=30+180, H2=30, L=[0,100], Lres=1, modes=['clip','crop'], sym=True, Csteps=128, png_dir=".", png_prefix="cmap", out=False):
    """ For a given range of L, stitches the half planes H=H1 and H=H2 along the gray line, each extended to the maximal chroma
        (if sym==True then Cmax is set for both hues, otherwise for each hue independently)
    """
    if len(L)==0: return
    if len(L)==1: L=[L[0],L[0]]
    L_range = np.linspace(L[0],L[1],int(abs(L[1]-L[0])*Lres)+1)
    C_range = np.linspace(0, 1, Csteps+1) # normalized to Cmax for this L
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

def make_cmap_monohue(H=0, L=[0,50], Lres=1, modes=['clip','crop'], sym=False, targets=['mpl','png'], png_dir=".", out=False):
    """ For a given H, draws a path from L[0] to L[1] at the maximal C
        (if sym==True then Cmax is set for both L and 100-L, otherwise for each L independently)
    """
    if len(L)<2: return
    L_range = np.linspace(L[0],L[1],abs(L[1]-L[0])*Lres+1)
    Cmax = {}
    RGB = {}
    for mode in modes:
        if sym: Cmax_func = lambda l: np.minimum(Cmax_for_LH[mode](l,H), Cmax_for_LH[mode](100-l,H)) # the Cmax for (L,H) and (100-L,H)
        else:   Cmax_func = lambda l: Cmax_for_LH[mode](l,H)                                         # the Cmax for (L,H)
        #Cmax[mode] = Cmax_func(L_range)
        #print "drawing %s path from (%i, %i, %i) to (%i, %i, %i)"%(mode,L_range[0],Cmax[mode][0],H,L_range[-1],Cmax[mode][-1],H)
        RGB[mode] = convert.clip3(convert.LCH2RGB(L_range,Cmax_func(L_range),H))
        name = 'monohue_L%03i-%03i_H%03i_%s'%(L[0],L[1],hue(H),mode)
        generate_cmaps(RGB[mode], name, targets, png_dir=png_dir)
    if out: return RGB

# selection

def make_cmap_favs(types=['equilum','diverging','monohue'], modes=['clip','crop'], targets=['mpl','png'], dir='.', plot=True):
    """ Generates and plots a selection of colour maps of different types (for mpl and as png) """
    global CMAP
    if 'equilum' in types:
        print "-------"
        print "equilum"
        print "-------"
        for mode in modes:
            CMAP = {}
            if 'png' in targets:
                for L in np.arange(20,90,10):
                    make_cmap_equilum(L=L, H=[0,250], Hres=1, modes=[mode], targets=['png'], png_dir=dir)
            if 'mpl' in targets:
                for L in np.arange(20,90,10):
                    make_cmap_equilum(L=L, H=[0,250], Hres=2, modes=[mode], targets=['mpl'])
                if plot: plot_cmaps(title="equilum_%s colour maps"%mode, fig=0, dir=dir)
    if 'diverging' in types:
        print "---------"
        print "diverging"
        print "---------"
        for mode in modes:
            CMAP = {}
            if 'png' in targets:
                for L in np.arange(20,90,10):
                    make_cmap_diverging(H1=30+180, H2=30, L=L, Cres=1, modes=[mode], targets=['png'], png_dir=dir)
                make_cmap_diverging2D(H1=30+180, H2=30, L=[0,100], Lres=1, Csteps=128, modes=[mode], png_dir=dir)
            if 'mpl' in targets:
                for L in np.arange(20,90,10):
                    make_cmap_diverging(H1=30+180, H2=30, L=L, Cres=4, modes=[mode], targets=['mpl'])
                if plot: plot_cmaps(title="diverging_%s colour maps"%mode, fig=0, dir=dir)
    if 'monohue' in types:
        print "-------"
        print "monohue"
        print "-------"
        for mode in modes:
            CMAP = {}
            if 'png' in targets:
                for H in [40, 140, 290]:
                    make_cmap_monohue(H=H, L=[  0, 50], Lres=1 , sym=False, modes=[mode], targets=['png'], png_dir=dir)
                    make_cmap_monohue(H=H, L=[100, 50], Lres=1 , sym=False, modes=[mode], targets=['png'], png_dir=dir)
                    make_cmap_monohue(H=H, L=[0  ,100], Lres=1 , sym=False, modes=[mode], targets=['png'], png_dir=dir)
            if 'mpl' in targets:
                for H in [40, 140, 290]:
                    make_cmap_monohue(H=H, L=[  0, 50], Lres=10, sym=False, modes=[mode], targets=['mpl'])
                    make_cmap_monohue(H=H, L=[100, 50], Lres=10, sym=False, modes=[mode], targets=['mpl'])
                    make_cmap_monohue(H=H, L=[0  ,100], Lres= 5, sym=False, modes=[mode], targets=['mpl'])
                if plot: plot_cmaps(title="monohue_%s colour maps"%mode, fig=0, dir=dir)


# generate colour maps for Matplotlib, or as PNG

def generate_cmaps(RGB_list, name, targets, png_height=32, png_prefix="cmap", png_dir="."):
    """ Generates colour maps from a 1D array of RGB triplets, for the specified targets (Matplotlib or PNG) """
    for target in targets:
        if target == 'png':
            if not os.path.exists(png_dir): os.makedirs(png_dir)
            RGB_array = np.zeros((png_height,len(RGB_list),3))
            for n in range(len(RGB_list)): RGB_array[:,n,0:3] = RGB_list[n]
            fname = png_dir+"/"+png_prefix+"_"+name+".png"
            write_RGB_as_PNG(RGB_array, fname)
        if target == 'mpl':
            print "creating cmap '%s' for Matplotlib (%4i steps)"%(name,len(RGB_list))
            CMAP[name     ] = matplotlib.colors.ListedColormap(RGB_list      , name)
            CMAP[name+'_r'] = matplotlib.colors.ListedColormap(RGB_list[::-1], name)

def write_RGB_as_PNG(arr, fname):
    """ writes a RGB array as a PNG file, at its intrinsic size """
    print 'writing %s (%ix%i)'%(fname,arr.shape[0],arr.shape[1])
    plt.imsave(arr=arr, fname=fname, origin='lower')

def register_to_mpl(names, reversed=True):
    """ Adds a cmap to Matplotlib's list """
    for name in names:
        matplotlib.cm.register_cmap(cmap=CMAP[name], name=name)
        if reversed: matplotlib.cm.register_cmap(cmap=CMAP[name+'_r'], name=name+'_r')

# plotting

def get_cmap(name, nsteps=None):
    """ Returns a named colour map, looking first in the local cache CMAP, then in Matplotlib's registered colours
        optionally resampled in `nsteps` bins
    """
    if name in CMAP.keys():
        cmap = CMAP[name]
    elif name in matplotlib.cm.cmap_d.keys():
        cmap = matplotlib.cm.cmap_d[name]
    else:
        print "Unknown cmap: ",name
        cmap = None
    if cmap != None and nsteps>0: cmap = cmap._resample(nsteps)
    return cmap

def plot_cmaps(names=[], reverse=False, nsteps=None, width=256, height=32, fig=1, figsize=None, frame=False, labels="left", labelsize=10, title="", titlesize=14, dir=".", fname_all="cmaps", fname="cmap"):
    """ Plots all colour maps listed by name
        If `fname` is set writes them individually as PNG images of size `width` by `height`
        If `fname_all` is set writes the figure with all of them as a PNG image
    """
    # adapted from http://matplotlib.org/examples/color/colormaps_reference.html
    if len(names)==0: names = list_all(reverse=reverse)
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
    gradient = np.linspace(0, 1, width)
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
            print 'writing ',fullname
            plt.imsave(arr=np.tile(gradient,(height,1)), origin='lower', fname=fullname, cmap=cmap)
    if fname_all != "":
        fullname = "%s/%s%s.png"%(dir,fname_all,"_"+title if title!="" else "")
        print 'writing ',fullname
        plt.savefig(fullname, dpi=None, bbox_inches='tight')
    if fig==0: plt.close(fig)

from mpl_toolkits.axes_grid1 import make_axes_locatable

def test_cmaps(data=[], names=[], reverse=False, nsteps=None, figsize=None, titlesize=12, dir=".", fname="testcmap"):
    """ Displays dummy 2D data with all the colour maps listed by name """
    if len(data)==0: data = mock_data(f_x=-1, phi_x=0.5, f_y=1, phi_y=0)
    if len(names)==0: names = list_all(reverse=reverse)
    for i in range(len(names)):
        cmap = get_cmap(names[i], nsteps)
        if cmap==None: continue
        plt.close(i+1)
        fig = plt.figure(i+1,figsize=figsize)
        im = plt.imshow(data, aspect='equal', interpolation='nearest', cmap=cmap)
        plt.title(names[i], fontsize=titlesize)
        plt.xticks([])
        plt.yticks([])
        #plt.colorbar()
        #using an axis divider so that the colour bar always be of the same height as the plot
        cax = make_axes_locatable(plt.gca()).append_axes("right",size="5%",pad=0.25)
        cbar = plt.colorbar(im, cax=cax)
        #fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        if fname != "":
            fullname = '%s/%s_%s.png'%(dir,fname,names[i])
            print 'writing ',fullname
            plt.savefig(fullname, dpi=None, bbox_inches='tight')

def mock_data(f_x, phi_x, f_y, phi_y, res=100):
    """ Generates a 2D periodic pattern """
    x = np.arange(0, 1, 1./res)
    y = np.arange(0, 1, 1./res)
    X,Y = np.meshgrid(x,y)
    Z = np.sin((f_x*X+phi_x)*np.pi)*np.sin((f_y*Y+phi_y)*np.pi)
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return Z

def list_all(reverse=False):
    """ Lists names of all the colour maps present in CMAP """
    names = []
    for name in CMAP.keys():
        if name[-2:] != '_r' or reverse == True: names.append(name)
    names = sorted(names, key=rank_cmap)
    print 'found cmaps: ',names
    return names

def rank_cmap(name):
    """ Ranks a cmap by name for custom ordering """
    # reverse the order of L
    pos = name.find("_L")
    if pos>0: name = name.replace(name[pos:pos+2+3],"_L%03i"%(100-int(name[pos+2:pos+2+3])))
    return name
