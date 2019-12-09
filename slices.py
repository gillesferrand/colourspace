# -*- coding: utf-8 -*-
"""
Generates slices in the LCH colour space:
- L versus H as function of C
- C versus H as function of L
- L versus C as function of H

res is the number of points per unit of L,C,H

When a colour is outside the sRGB gamut:
for mode='crop' the colour is discarded: it is replaced with a gray of the same L
for mode='clip' the colour is faked: the R,G,B values are clipped to [0,1]
When a colour is outside the human gamut: it is replaced with black
"""

import sys
import os
import numpy as np
import pylab as plt
from . import convert
from . import gamut
res_gamut = 1

#-------
# LH(C)
#-------

def LH_planes(C=np.arange(0,201,1),L=[0,100],H=[0,360],res=1,showfig=False,dir="LHplanes",name="LHplane",modes=['crop','clip'],axes=['on','off'],figsize=None):
    """ Generates the LH plane for a range of C """
    for Cj in C: LH = LH_plane(Cj,L=L,H=H,res=res,showfig=showfig,dir=dir,name=name,modes=modes,axes=axes,figsize=figsize)

def LH_plane(C,L=[0,100],H=[0,360],res=1,showfig=True,dir=".",name="LHplane",modes=['crop','clip'],axes=['on','off'],figsize=None):
    """ Generates an RGB array that samples the LH plane for a given C """
    nL = int((L[1]-L[0])*res+1)
    nH = int((H[1]-H[0])*res+1)
    L_range = np.linspace(L[0],L[1],nL)
    H_range = np.linspace(H[0],H[1],nH)
    arr = {}
    for mode in ['crop','clip']:
        arr[mode] = np.ones((nL,nH,3),dtype=np.float32)
    RGB_black = (0, 0, 0)
    for i in range(len(L_range)):
        RGB_gray = convert.clip3(convert.LCH2RGB(L_range[i],0,0))
        for k in range(len(H_range)):
            display_triplet(L_range[i],C,H_range[k])
            if C <= gamut.Cmax_for_LH(L_range[i],H_range[k],res=res_gamut,gamut='full'):
                RGB = convert.LCH2RGB(L_range[i],C,H_range[k])
                RGB_crop = convert.crop3(RGB)
                if np.nan in RGB_crop: RGB_crop = RGB_gray
                RGB_clip = convert.clip3(RGB)
            else:
                RGB_crop = RGB_black
                RGB_clip = RGB_black
            arr['crop'][i,k] = RGB_crop
            arr['clip'][i,k] = RGB_clip
    sys.stdout.write("\n")
    sys.stdout.flush()
    # figure
    ext = "C%03i"%C
    xlabel = "H"
    ylabel = "L"
    extent = [H[0], H[1], L[0], L[1]]
    xticks = H_ticks if H==[0,360] else None
    yticks = L_ticks if L==[0,100] else None
    make_figure(showfig=showfig,dir=dir,name=name,ext=ext,arr=arr,extent=extent,xlabel=xlabel,ylabel=ylabel,xticks=xticks,yticks=yticks,modes=modes,axes=axes,figsize=figsize)
    return arr

def LH_plane_max(L=[0,100],H=[0,360],res=1,showfig=True,dir=".",name="LHplane",kinds=['max','equ'],modes=['crop','clip'],axes=['on','off'],figsize=None):
    """ Generates an RGB array that samples the LH plane at the max possible C
        Max means either the Cmax that accomodates all H for a given L ("equ") or the Cmax of this H for this L ("max")
    """
    nL = int((L[1]-L[0])*res+1)
    nH = int((H[1]-H[0])*res+1)
    L_range = np.linspace(L[0],L[1],nL)
    H_range = np.linspace(H[0],H[1],nH)
    arr = {}
    for kind in ['equ','max']:
        arr[kind] = {}
        for mode in ['crop','clip']:
            arr[kind][mode] = np.ones((nL,nH,3),dtype=np.float32)
    for i in range(len(L_range)):
        RGB_gray = convert.clip3(convert.LCH2RGB(L_range[i],0,0))
        LH = [[L_range[i], H_range[k]] for k in range(len(H_range))]
        Cmax_full = np.array(map(lambda (l,h): gamut.Cmax_for_LH(l,h,res=res_gamut,gamut='full'), LH)).min()
        Cmax_sRGB = np.array(map(lambda (l,h): gamut.Cmax_for_LH(l,h,res=res_gamut,gamut='sRGB'), LH)).min()
        for k in range(len(H_range)):
            # maximal possible chroma for all hues, in the cropped space
            C = Cmax_sRGB
            display_triplet(L_range[i],C,H_range[k])
            arr['equ']['crop'][i,k] = convert.clip3(convert.LCH2RGB(L_range[i],C,H_range[k]))
            # maximal possible chroma for this hue, in the cropped space
            C = gamut.Cmax_for_LH(L_range[i],H_range[k],res=res_gamut,gamut='sRGB')
            display_triplet(L_range[i],C,H_range[k])
            arr['max']['crop'][i,k] = convert.clip3(convert.LCH2RGB(L_range[i],C,H_range[k]))
            # maximal possible chroma for all hues, in the clipped space
            C = Cmax_full
            display_triplet(L_range[i],C,H_range[k])
            arr['equ']['clip'][i,k] = convert.clip3(convert.LCH2RGB(L_range[i],C,H_range[k]))
            # maximal possible chroma for this hue, in the clipped space
            C = gamut.Cmax_for_LH(L_range[i],H_range[k],res=res_gamut,gamut='full')
            display_triplet(L_range[i],C,H_range[k])
            arr['max']['clip'][i,k] = convert.clip3(convert.LCH2RGB(L_range[i],C,H_range[k]))
    sys.stdout.write("\n")
    sys.stdout.flush()
    # figure
    xlabel = "H"
    ylabel = "L"
    extent = [H[0], H[1], L[0], L[1]]
    xticks = H_ticks if H==[0,360] else None
    yticks = L_ticks if L==[0,100] else None
    for kind in kinds:
        ext = "Cmax%s"%kind
        make_figure(showfig=showfig,dir=dir,name=name,ext=ext,arr=arr[kind],extent=extent,xlabel=xlabel,ylabel=ylabel,xticks=xticks,yticks=yticks,modes=modes,axes=axes,figsize=figsize)
    return arr

#-------
# CH(L)
#-------

def CH_planes(L=np.arange(0,101,1),C=[0,200],H=[0,360],res=1,stretch=False,showfig=False,dir="CHplanes",name="CHplane",modes=['crop','clip'],axes=['on','off'],figsize=None):
    """ Generates the CH plane for a range of L """
    for Li in L: CH = CH_plane(Li,C=C,H=H,res=res,stretch=stretch,showfig=showfig,dir=dir,name=name,modes=modes,axes=axes,figsize=figsize)

def CH_plane(L,C=[0,200],H=[0,360],res=1,stretch=False,showfig=True,dir=".",name="CHplane",modes=['crop','clip'],axes=['on','off'],figsize=None):
    """ Generates an RGB array that samples the CH plane for a given L """
    nC = int((C[1]-C[0])*res+1)
    nH = int((H[1]-H[0])*res+1)
    H_range = np.linspace(H[0],H[1],nH)
    arr = {}
    for mode in ['crop','clip']:
        arr[mode] = np.ones((nC,nH,3),dtype=np.float32)
    RGB_black = (0, 0, 0)
    RGB_gray = convert.clip3(convert.LCH2RGB(L,0,0))
    for k in range(len(H_range)):
        Cmax_sRGB = gamut.Cmax_for_LH(L,H_range[k],res=res_gamut,gamut='sRGB')
        Cmax_full = gamut.Cmax_for_LH(L,H_range[k],res=res_gamut,gamut='full')
        if stretch:
            C_range = np.linspace(C[0],Cmax_sRGB,nC) # C up to max representable
            for j in range(len(C_range)):
                display_triplet(L,C_range[j],H_range[k])
                arr['crop'][j,k] = convert.clip3(convert.LCH2RGB(L,C_range[j],H_range[k]))
            C_range = np.linspace(C[0],Cmax_full,nC) # C up to max possible
            for j in range(len(C_range)):
                display_triplet(L,C_range[j],H_range[k])
                arr['clip'][j,k] = convert.clip3(convert.LCH2RGB(L,C_range[j],H_range[k]))
        else:
            C_range = np.linspace(C[0],C[1],nC)
            for j in range(len(C_range)):
                display_triplet(L,C_range[j],H_range[k])
                if C_range[j] <= Cmax_full:
                    RGB = convert.LCH2RGB(L,C_range[j],H_range[k])
                    RGB_crop = convert.crop3(RGB)
                    if np.nan in RGB_crop: RGB_crop = RGB_gray
                    RGB_clip = convert.clip3(RGB)
                else:
                    RGB_crop = RGB_black
                    RGB_clip = RGB_black
                arr['crop'][j,k] = RGB_crop
                arr['clip'][j,k] = RGB_clip
    sys.stdout.write("\n")
    sys.stdout.flush()
    # figure
    ext = "L%03i"%L
    xlabel = "H"
    ylabel = "C"
    extent = [H[0], H[1], C[0], C[1]]
    xticks = H_ticks if H==[0,360] else None
    yticks = C_ticks if C==[0,200] else None
    if stretch: # C normalized to 100%
        name += "_stretch"
        ylabel = "C/Cmax"
        extent = [H[0], H[1], C[0], 100]
        yticks = [0, 50, 100]
    make_figure(showfig=showfig,dir=dir,name=name,ext=ext,arr=arr,extent=extent,xlabel=xlabel,ylabel=ylabel,xticks=xticks,yticks=yticks,modes=modes,axes=axes,figsize=figsize)
    return arr

#-------
# LC(H)
#-------

def LC_planes(H=np.arange(0,360,1),L=[0,100],C=[0,200],res=1,stretch=False,showfig=False,dir="LCplanes",name="LCplane",modes=['crop','clip'],axes=['on','off'],figsize=None):
    """ Generates the LC plane for a range of H """
    for Hk in H: LC = LC_plane(Hk,L=L,C=C,res=res,stretch=stretch,showfig=showfig,dir=dir,name=name,modes=modes,axes=axes,figsize=figsize)

def LC_plane(H,L=[0,100],C=[0,200],res=1,stretch=False,showfig=True,dir=".",name="LCplane",modes=['crop','clip'],axes=['on','off'],figsize=None):
    """ Generates an RGB array that samples the LC plane for a given H """
    nL = int((L[1]-L[0])*res+1)
    nC = int((C[1]-C[0])*res+1)
    L_range = np.linspace(L[0],L[1],nL)
    arr = {}
    for mode in ['crop','clip']:
        arr[mode] = np.ones((nL,nC,3),dtype=np.float32)
    RGB_black = (0, 0, 0)
    for i in range(len(L_range)):
        Cmax_sRGB = gamut.Cmax_for_LH(L_range[i],H,res=res_gamut,gamut='sRGB')
        Cmax_full = gamut.Cmax_for_LH(L_range[i],H,res=res_gamut,gamut='full')
        if stretch:
            C_range = np.linspace(C[0],Cmax_sRGB,nC) # C up to max representable
            for j in range(len(C_range)):
                display_triplet(L_range[i],C_range[j],H)
                arr['crop'][i,j] = convert.clip3(convert.LCH2RGB(L_range[i],C_range[j],H))
            C_range = np.linspace(C[0],Cmax_full,nC) # C up to max possible
            for j in range(len(C_range)):
                display_triplet(L_range[i],C_range[j],H)
                arr['clip'][i,j] = convert.clip3(convert.LCH2RGB(L_range[i],C_range[j],H))
        else:
            RGB_gray = convert.clip3(convert.LCH2RGB(L_range[i],0,0))
            C_range = np.linspace(C[0],C[1],nC) # C as requested
            for j in range(len(C_range)):
                display_triplet(L_range[i],C_range[j],H)
                if C_range[j] <= Cmax_full:
                    RGB = convert.LCH2RGB(L_range[i],C_range[j],H)
                    RGB_crop = convert.crop3(RGB)
                    if np.nan in RGB_crop: RGB_crop = RGB_gray
                    RGB_clip = convert.clip3(RGB)
                else:
                    RGB_crop = RGB_black
                    RGB_clip = RGB_black
                arr['crop'][i,j] = RGB_crop
                arr['clip'][i,j] = RGB_clip
    sys.stdout.write("\n")
    sys.stdout.flush()
    # figure
    ext = "H%03i"%H
    xlabel = "C"
    ylabel = "L"
    extent = [C[0], C[1], L[0], L[1]]
    xticks = C_ticks if C==[0,200] else None
    yticks = L_ticks if L==[0,100] else None
    if stretch: # C normalized to 100%
        name += "_stretch"
        xlabel = "C/Cmax"
        extent = [C[0], 100, L[0], L[1]]
        xticks = [0, 50, 100]
    make_figure(showfig=showfig,dir=dir,name=name,ext=ext,arr=arr,extent=extent,xlabel=xlabel,ylabel=ylabel,xticks=xticks,yticks=yticks,modes=modes,axes=axes,figsize=figsize)
    return arr

#-------
# utils
#-------

def all_planes(slices=['LH','CH','LC'], res=1, dir='.',modes=['crop','clip'], axes=['on','off']):
    """ Generates all the planes """
    if 'LH' in slices:
        LH_planes(C=np.arange(0,201,1), L=[0,100], H=[0,360], res=res, dir=dir, modes=modes, axes=axes)
        LH_plane_max(                   L=[0,100], H=[0,360], res=res, dir=dir, modes=modes, axes=axes)
    if 'CH' in slices:
        CH_planes(L=np.arange(0,101,1), C=[0,200], H=[0,360], res=res, dir=dir, modes=modes, axes=axes)
    if 'LC' in slices:
        LC_planes(H=np.arange(0,360,1), L=[0,100], C=[0,200], res=res, dir=dir, modes=modes, axes=axes, stretch=False)
        LC_planes(H=np.arange(0,360,1), L=[0,100], C=[0,100], res=res, dir=dir, modes=modes, axes=axes, stretch=True)

def display_triplet(L,C,H,newline=False):
    """ Prints the (L,C,H) values """
    sys.stdout.write("L = %6.2f, C = %6.2f, H = %6.2f"%(L,C,H))
    sys.stdout.write("\n") if newline else sys.stdout.write("\r")
    sys.stdout.flush()

L_ticks = np.arange(int(100/25.)+1)*25
H_ticks = np.arange(int(360/90.)+1)*90
C_ticks = np.arange(int(200/50.)+1)*50

def make_figure(showfig,dir,name,ext,arr,extent,xlabel,ylabel,xticks,yticks,modes=['crop','clip'],axes=['on','off'],aspect='equal',figsize=None):
    """ Displays and/or saves a figure """
    if dir != "" and not os.path.exists(dir): os.makedirs(dir)
    for mode in modes:
        fname = dir + "/" + name + "_" + mode
        if showfig or (dir != "" and 'on' in axes):
            figname = "%s_%s_%s"%(name,mode,ext)
            plt.figure(figname,figsize=figsize)
            plt.imshow(arr[mode],origin='lower',extent=extent)
            plt.title(ext)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel, rotation='horizontal')
            if xticks is not None: plt.xticks(xticks)
            if yticks is not None: plt.yticks(yticks)
            if dir != "" and 'on' in axes:
                print "writing %s"%(fname+"_axon_"+ext+".png")
                plt.savefig(fname+"_axon_"+ext+".png", dpi=None, bbox_inches='tight')
                if not showfig: plt.close(figname)
        if dir != "" and 'off' in axes:
            print "writing %s"%(fname+"_axoff_"+ext+".png")
            plt.imsave(arr=arr[mode],origin='lower',fname=fname+"_axoff_"+ext+".png")
