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
Cmax_for_LH['crop'] = lambda l,h: gamut.Cmax_for_LH(l,h,res=res_gamut,gamut="sRGB")
Cmax_for_LH['clip'] = lambda l,h: gamut.Cmax_for_LH(l,h,res=res_gamut,gamut="full")

def hue(H):
    return H if 0 <= H <= 360 else H%360

# equilum

def make_cmap_equilum(L=70, H=[0,250], Hres=1, modes=['clip','crop'], targets=['mpl','png'], png_dir=".", out=False):
    """ Draws a line at constant L in the LH plane, in the chosen H range, at the Cmax for this L
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
        Cmax[mode] = np.array(map(lambda h: Cmax_for_LH[mode](L,h), H_range)).min() # the Cmax for all hues
        RGB [mode] = np.array(map(lambda h: convert.clip3(convert.LCH2RGB(L,Cmax[mode],h)), H_range))
        name = 'equilum_L%03i_H%03i-%03i_%s'%(L,hue(H[0]),hue(H[1]),mode)
        generate_cmaps(RGB[mode], name, targets, png_dir=png_dir)
    if out: return RGB

# diverging

def make_cmap_diverging(H1=30+180, H2=30, L=50, modes=['clip','crop'], sym=True, Cres=1, Csteps=0, Cmax=0, targets=['mpl','png'], png_dir=".", out=False):
    """ For a given L, draws a path from H1 at max chroma to H2 at max chroma
        (if sym==True then Cmax is set for both hues, otherwise for each hue independently)
    """
    RGB = {}
    for mode in modes:
        if Cmax>0:
            Cmax1 = Cmax
            Cmax2 = Cmax
            Cmax12 = Cmax
        else:
            Cmax1 = Cmax_for_LH[mode](L,H1) # the Cmax for H1
            Cmax2 = Cmax_for_LH[mode](L,H2) # the Cmax for H2
            Cmax12 = min(Cmax1, Cmax2) # the Cmax that accomodates both hues
            #print "L = ",L," : Cmax1 = ",Cmax1,", Cmax2 =",Cmax2
        if Csteps>0: # useful for making 2D maps, by stacking rows of equal length
            C_range1  = np.linspace(0, Cmax1 , Csteps+1)
            C_range2  = np.linspace(0, Cmax2 , Csteps+1)
            C_range12 = np.linspace(0, Cmax12, Csteps+1)
        else:
            C_range1  = np.linspace(0, Cmax1 , Cmax1 *Cres+1)
            C_range2  = np.linspace(0, Cmax2 , Cmax2 *Cres+1)
            C_range12 = np.linspace(0, Cmax12, Cmax12*Cres+1)
        RGB[mode] = {}
        if sym:
            RGB12 = np.array(map(lambda c: convert.clip3(convert.LCH2RGB(L,c,H1)), C_range12)) # H1 side, restricted to Cmax(H2)
            RGB21 = np.array(map(lambda c: convert.clip3(convert.LCH2RGB(L,c,H2)), C_range12)) # H2 side, restricted to Cmax(H1)
            RGB[mode] = np.concatenate((RGB12[::-1], RGB21[1:]))
        else:
            RGB1  = np.array(map(lambda c: convert.clip3(convert.LCH2RGB(L,c,H1)), C_range1 )) # H1 side, full range
            RGB2  = np.array(map(lambda c: convert.clip3(convert.LCH2RGB(L,c,H2)), C_range2 )) # H2 side, full range
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
    L_range = np.linspace(L[0],L[1],abs(L[1]-L[0])*Lres+1)
    RGB_array = {}
    for mode in modes:
        RGB_array[mode] = np.zeros((len(L_range),2*Csteps-1,3))
        for i in range(len(L_range)):
            RGB_list = make_cmap_diverging(H1, H2, L_range[i], modes=[mode], sym=sym, Csteps=Csteps, targets=[], out=True)
            for j in range(2*Csteps-1): RGB_array[mode][i,j,0:3] = RGB_list[mode][j]
        name = 'diverging2D_L%03i-%03i_H%03i-%03i_%s'%(L_range[0],L_range[-1],hue(H1),hue(H2),mode)
        fullname = png_dir+"/"+png_prefix+"_"+name+".png"
        if not os.path.exists(png_dir): os.makedirs(png_dir)
        write_RGB_as_PNG(RGB_array[mode], fname=fullname)
    if out: return RGB_array

# monohue

def make_cmap_monohue(H, L=[0,50], Lres=1, modes=['clip','crop'], sym=False, targets=['mpl','png'], png_dir=".", out=False):
    """ For a given H, draws a path from L[0] to L[1] at the maximal C
        (if sym==True then Cmax is set for both L and 100-L, otherwise for each L independently)
    """
    if len(L)<2: return
    L_range = np.linspace(L[0],L[1],abs(L[1]-L[0])*Lres+1)
    Cmax = {}
    RGB = {}
    for mode in modes:
        if sym: Cmax_func = lambda l: min(Cmax_for_LH[mode](l,H), Cmax_for_LH[mode](100-l,H)) # the Cmax for (L,H) and (100-L,H)
        else:   Cmax_func = lambda l: Cmax_for_LH[mode](l,H)                                  # the Cmax for (L,H)
        Cmax[mode] = np.array(map(Cmax_func, L_range))
        #print "drawing %s path from (%i, %i, %i) to (%i, %i, %i)"%(mode,L_range[0],Cmax[mode][0],H,L_range[-1],Cmax[mode][-1],H)
        RGB[mode] = np.array(map(lambda l: convert.clip3(convert.LCH2RGB(l,Cmax_func(l),H)), L_range))
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
                if plot: plot_cmaps(title="equilum_%s"%mode, fig=0, dir=dir)
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
                if plot: plot_cmaps(title="diverging_%s"%mode, fig=0, dir=dir)
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
                if plot: plot_cmaps(title="monohue_%s"%mode, fig=0, dir=dir)


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

def plot_cmaps(names=[], reverse=False, width=256, height=32, fig=1, figsize=None, title="", dir=".", fname_all="cmaps", fname="cmap"):
    """ Plots all colour maps listed by name (in the local cache CMAP) """
    # adapted from http://matplotlib.org/examples/color/colormaps_reference.html
    if len(names)==0: names = list_all(reverse=reverse)
    nrows = len(names)
    if nrows == 0: return
    plt.close(fig)
    figure, axes = plt.subplots(num=fig, nrows=nrows, figsize=figsize)
    figure.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title(title+" colour maps", fontsize=14)
    gradient = np.linspace(0, 1, width)
    for ax, name in zip(axes, names):
        cmap = CMAP[name]
        ax.imshow(np.tile(gradient,(2,1)), aspect='auto', interpolation='nearest', cmap=cmap)
        ax.set_axis_off() # turn off *all* ticks and spines
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        figure.text(x_text, y_text, name, va='center', ha='right', fontsize=10, family='monospace')
        if fname != "":
            fullname = "%s/%s%i_%s.png"%(dir,fname,width,name)
            print 'writing ',fullname
            plt.imsave(arr=np.tile(gradient,(height,1)), origin='lower', fname=fullname, cmap=cmap)
    if fname_all != "":
        fullname = "%s/%s%s.png"%(dir,fname_all,"_"+title if title!="" else "")
        print 'writing ',fullname
        plt.savefig(fullname, dpi=None, bbox_inches='tight')
    if fig==0: plt.close(fig)

def test_cmaps(names=[], reverse=False, figsize=None, dir=".", fname="testcmap"):
    """ Displays dummy 2D data with all the colour maps listed by name (in the local cache CMAP) """
    if len(names)==0: names = list_all(reverse=reverse)
    x = np.arange(0, np.pi, 0.01)
    y = np.arange(0, np.pi, 0.01)
    X,Y = np.meshgrid(x,y)
    #Z = np.sin(X)*np.sin(Y)
    Z = np.sin(-2*X)*np.sin(Y)
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    for i in range(len(names)):
        cmap = CMAP[names[i]]
        plt.close(i+1)
        plt.figure(i+1,figsize=figsize)
        #fig, ax = plt.subplots(num=i+1, nrows=1)
        #fig.subplots_adjust(top=0.95, bottom=0, left=0, right=1)
        plt.imshow(Z, aspect='equal', interpolation='nearest', cmap=cmap)
        plt.title(names[i])
        plt.colorbar()
        #ax = plt.gca()
        #ax.set_axis_off()
        plt.xticks([])
        plt.yticks([])
        fig = plt.gcf()
        fig.subplots_adjust(top=0.95, bottom=0, left=0, right=1)
        if fname != "":
            fullname = '%s/%s_%s.png'%(dir,fname,names[i])
            print 'writing ',fullname
            plt.savefig(fullname, dpi=None, bbox_inches='tight')

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
