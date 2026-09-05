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

from __future__ import print_function
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

def LH_planes(C=np.arange(0,201,1),L=[0,100],H=[0,360],res=1,showfig=False,dir="LHplanes",name="LHplane",modes=['crop','clip'],axes=['on','off'],figsize=None,dpi=None):
    """ Generates the LH plane for a range of C """
    for Cj in C: LH = LH_plane(Cj,L=L,H=H,res=res,showfig=showfig,dir=dir,name=name,modes=modes,axes=axes,figsize=figsize,dpi=dpi)

def LH_plane(C,L=[0,100],H=[0,360],res=1,showfig=True,dir=".",name="LHplane",modes=['crop','clip'],axes=['on','off'],figsize=None,dpi=None):
    """ Generates an RGB array that samples the LH plane for a given C """
    nL = int((L[1]-L[0])*res+1)
    nH = int((H[1]-H[0])*res+1)
    L_range = np.linspace(L[0],L[1],nL)
    H_range = np.linspace(H[0],H[1],nH)
    LL, HH = np.meshgrid(L_range,H_range,indexing='ij')
    Cmax = {}
    valid = {}
    for gmt in ['sRGB','full']:
        Cmax[gmt] = gamut.Cmax_for_LH(LL,HH,res=res_gamut,gmt=gmt)
        valid[gmt] = np.logical_and(C>=0, C<=Cmax[gmt])[:,:,np.newaxis]
    arr = {}
    for mode in ['crop','clip']:
        arr[mode] = np.zeros((nL,nH,3),dtype=np.float32)
    arr['clip'] = np.where(valid['full'], convert.clip3(convert.LCH2RGB(LL,C,HH)), arr['clip'])
    arr['crop'] = np.where(valid['full'], convert.clip3(convert.LCH2RGB(LL,0,HH)), arr['crop'])
    arr['crop'] = np.where(valid['sRGB'], convert.clip3(convert.LCH2RGB(LL,C,HH)), arr['crop'])
    # figure
    ext = "C%03i"%C
    xlabel = "H"
    ylabel = "L"
    extent = [H[0], H[1], L[0], L[1]]
    xticks = H_ticks if H==[0,360] else None
    yticks = L_ticks if L==[0,100] else None
    make_figure(showfig=showfig,dir=dir,name=name,ext=ext,arr=arr,extent=extent,xlabel=xlabel,ylabel=ylabel,xticks=xticks,yticks=yticks,modes=modes,axes=axes,figsize=figsize,dpi=dpi)
    return arr

def LH_plane_max(L=[0,100],H=[0,360],res=1,showfig=True,dir=".",name="LHplane",kinds=['max','equ'],modes=['crop','clip'],axes=['on','off'],figsize=None,dpi=None):
    """ Generates an RGB array that samples the LH plane at the max possible C
        Max means either the Cmax that accomodates all H for a given L ("equ") or the Cmax of this H for this L ("max")
    """
    nL = int((L[1]-L[0])*res+1)
    nH = int((H[1]-H[0])*res+1)
    L_range = np.linspace(L[0],L[1],nL)
    H_range = np.linspace(H[0],H[1],nH)
    LL, HH = np.meshgrid(L_range,H_range,indexing='ij')
    arr = {}
    for kind in ['equ','max']:
        arr[kind] = {}
        for mode in ['crop','clip']:
            arr[kind][mode] = np.zeros((nL,nH,3),dtype=np.float32)
    Cmax = {}
    # maximal possible chroma for each hue
    for gmt in ['sRGB','full']:
        Cmax[gmt] = gamut.Cmax_for_LH(LL,HH,res=res_gamut,gmt=gmt)
    arr['max']['crop'] = convert.clip3(convert.LCH2RGB(LL,Cmax['sRGB'],HH))
    arr['max']['clip'] = convert.clip3(convert.LCH2RGB(LL,Cmax['full'],HH))
    # maximal possible chroma for all hues
    for gmt in ['sRGB','full']:
        Cmax[gmt] = np.amin(Cmax[gmt],axis=1)
        Cmax[gmt] = np.repeat(Cmax[gmt][:,np.newaxis],nH,axis=-1)
    arr['equ']['crop'] = convert.clip3(convert.LCH2RGB(LL,Cmax['sRGB'],HH))
    arr['equ']['clip'] = convert.clip3(convert.LCH2RGB(LL,Cmax['full'],HH))
    # figure
    xlabel = "H"
    ylabel = "L"
    extent = [H[0], H[1], L[0], L[1]]
    xticks = H_ticks if H==[0,360] else None
    yticks = L_ticks if L==[0,100] else None
    for kind in kinds:
        ext = "Cmax%s"%kind
        make_figure(showfig=showfig,dir=dir,name=name,ext=ext,arr=arr[kind],extent=extent,xlabel=xlabel,ylabel=ylabel,xticks=xticks,yticks=yticks,modes=modes,axes=axes,figsize=figsize,dpi=dpi)
    return arr

#-------
# CH(L)
#-------

def CH_planes(L=np.arange(0,101,1),C=[0,200],H=[0,360],res=1,stretch=False,showfig=False,dir="CHplanes",name="CHplane",modes=['crop','clip'],axes=['on','off'],figsize=None,dpi=None):
    """ Generates the CH plane for a range of L """
    for Li in L: CH = CH_plane(Li,C=C,H=H,res=res,stretch=stretch,showfig=showfig,dir=dir,name=name,modes=modes,axes=axes,figsize=figsize,dpi=dpi)

def CH_plane(L,C=[0,200],H=[0,360],res=1,stretch=False,showfig=True,dir=".",name="CHplane",modes=['crop','clip'],axes=['on','off'],figsize=None,dpi=None):
    """ Generates an RGB array that samples the CH plane for a given L
        (if stretch=True, then C is expressed as a percentage of Cmax)
    """
    nC = int((C[1]-C[0])*res+1)
    nH = int((H[1]-H[0])*res+1)
    H_range = np.linspace(H[0],H[1],nH)
    C_range = np.linspace(C[0],C[1],nC)
    CC_, HH = np.meshgrid(C_range,H_range,indexing='ij')
    Cmax = {}
    for gmt in ['sRGB','full']:
        #Cmax[gmt] = gamut.Cmax_for_LH(L,HH,res=res_gamut,gmt=gmt)
        Cmax[gmt] = gamut.Cmax_for_LH(L,H_range,res=res_gamut,gmt=gmt)
        Cmax[gmt] = np.repeat(Cmax[gmt][np.newaxis,:],nC,axis=0)
    valid = {}
    arr = {}
    for mode in ['crop','clip']:
        arr[mode] = np.zeros((nC,nH,3),dtype=np.float32)
    if stretch:
        # C up to max representable
        CC = CC_/100.*Cmax['sRGB']
        for gmt in ['sRGB','full']: valid[gmt] = np.logical_and(CC>=0, CC<=Cmax[gmt])[:,:,np.newaxis]
        arr['crop'] = np.where(valid['full'], convert.clip3(convert.LCH2RGB(L, 0,HH)), arr['crop'])
        arr['crop'] = np.where(valid['sRGB'], convert.clip3(convert.LCH2RGB(L,CC,HH)), arr['crop'])
        # C up to max possible
        CC = CC_/100.*Cmax['full']
        for gmt in [       'full']: valid[gmt] = np.logical_and(CC>=0, CC<=Cmax[gmt])[:,:,np.newaxis]
        arr['clip'] = np.where(valid['full'], convert.clip3(convert.LCH2RGB(L,CC,HH)), arr['clip'])
    else:
        # C as given
        CC = CC_
        for gmt in ['sRGB','full']: valid[gmt] = np.logical_and(CC>=0, CC<=Cmax[gmt])[:,:,np.newaxis]
        arr['clip'] = np.where(valid['full'], convert.clip3(convert.LCH2RGB(L,CC,HH)), arr['clip'])
        arr['crop'] = np.where(valid['full'], convert.clip3(convert.LCH2RGB(L, 0,HH)), arr['crop'])
        arr['crop'] = np.where(valid['sRGB'], convert.clip3(convert.LCH2RGB(L,CC,HH)), arr['crop'])
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
        yticks = [0, 50, 100] if C==[0,100] else None
    make_figure(showfig=showfig,dir=dir,name=name,ext=ext,arr=arr,extent=extent,xlabel=xlabel,ylabel=ylabel,xticks=xticks,yticks=yticks,modes=modes,axes=axes,figsize=figsize,dpi=dpi)
    return arr

#-------
# LC(H)
#-------

def LC_planes(H=np.arange(0,360,1),L=[0,100],C=[0,200],res=1,stretch=False,showfig=False,dir="LCplanes",name="LCplane",modes=['crop','clip'],axes=['on','off'],figsize=None,dpi=None):
    """ Generates the LC plane for a range of H """
    for Hk in H: LC = LC_plane(Hk,L=L,C=C,res=res,stretch=stretch,showfig=showfig,dir=dir,name=name,modes=modes,axes=axes,figsize=figsize,dpi=dpi)

def LC_plane(H,L=[0,100],C=[0,200],res=1,stretch=False,showfig=True,dir=".",name="LCplane",modes=['crop','clip'],axes=['on','off'],figsize=None,dpi=None):
    """ Generates an RGB array that samples the LC plane for a given H
        (if stretch=True, then C is expressed as a percentage of Cmax)
    """
    nL = int((L[1]-L[0])*res+1)
    nC = int((C[1]-C[0])*res+1)
    L_range = np.linspace(L[0],L[1],nL)
    C_range = np.linspace(C[0],C[1],nC)
    LL, CC_ = np.meshgrid(L_range,C_range,indexing='ij')
    Cmax = {}
    for gmt in ['sRGB','full']:
        #Cmax[gmt] = gamut.Cmax_for_LH(LL,H,res=res_gamut,gmt=gmt)
        Cmax[gmt] = gamut.Cmax_for_LH(L_range,H,res=res_gamut,gmt=gmt)
        Cmax[gmt] = np.repeat(Cmax[gmt][:,np.newaxis],nC,axis=1)
    valid = {}
    arr = {}
    for mode in ['crop','clip']:
        arr[mode] = np.zeros((nL,nC,3),dtype=np.float32)
    if stretch:
        # C up to max representable
        CC = CC_/100.*Cmax['sRGB']
        for gmt in ['sRGB','full']: valid[gmt] = np.logical_and(CC>=0, CC<=Cmax[gmt])[:,:,np.newaxis]
        arr['crop'] = np.where(valid['full'], convert.clip3(convert.LCH2RGB(LL, 0,H)), arr['crop'])
        arr['crop'] = np.where(valid['sRGB'], convert.clip3(convert.LCH2RGB(LL,CC,H)), arr['crop'])
        # C up to max possible
        CC = CC_/100.*Cmax['full']
        for gmt in [       'full']: valid[gmt] = np.logical_and(CC>=0, CC<=Cmax[gmt])[:,:,np.newaxis]
        arr['clip'] = np.where(valid['full'], convert.clip3(convert.LCH2RGB(LL,CC,H)), arr['clip'])
    else:
        # C as given
        CC = CC_
        for gmt in ['sRGB','full']: valid[gmt] = np.logical_and(CC>=0, CC<=Cmax[gmt])[:,:,np.newaxis]
        arr['clip'] = np.where(valid['full'], convert.clip3(convert.LCH2RGB(LL,CC,H)), arr['clip'])
        arr['crop'] = np.where(valid['full'], convert.clip3(convert.LCH2RGB(LL, 0,H)), arr['crop'])
        arr['crop'] = np.where(valid['sRGB'], convert.clip3(convert.LCH2RGB(LL,CC,H)), arr['crop'])
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
        xticks = [0, 50, 100] if C==[0,100] else None
    make_figure(showfig=showfig,dir=dir,name=name,ext=ext,arr=arr,extent=extent,xlabel=xlabel,ylabel=ylabel,xticks=xticks,yticks=yticks,modes=modes,axes=axes,figsize=figsize,dpi=dpi)
    return arr

#-------
# utils
#-------

def all_planes(slices=['LH','CH','LC'], res=1, dir='.',modes=['crop','clip'], axes=['on','off']):
    """ Generates all the planes """
    if 'LH' in slices:
        LH_planes(C=np.arange(0,201,1), L=[0,100], H=[0,360], res=res, dir=dir, modes=modes, axes=axes, showfig=False)
        LH_plane_max(                   L=[0,100], H=[0,360], res=res, dir=dir, modes=modes, axes=axes, showfig=False)
    if 'CH' in slices:
        CH_planes(L=np.arange(0,101,1), C=[0,200], H=[0,360], res=res, dir=dir, modes=modes, axes=axes, showfig=False)
    if 'LC' in slices:
        LC_planes(H=np.arange(0,360,1), L=[0,100], C=[0,200], res=res, dir=dir, modes=modes, axes=axes, showfig=False, stretch=False)
        LC_planes(H=np.arange(0,360,1), L=[0,100], C=[0,100], res=res, dir=dir, modes=modes, axes=axes, showfig=False, stretch=True)

L_ticks = np.arange(int(100/25.)+1)*25
H_ticks = np.arange(int(360/90.)+1)*90
C_ticks = np.arange(int(200/50.)+1)*50

def make_figure(showfig, dir, name, ext, arr, extent, xlabel, ylabel, xticks, yticks, modes=['crop','clip'], axes=['on','off'], aspect='equal', figsize=None, dpi=None):
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
                print("writing %s"%(fname+"_axon_"+ext+".png"))
                plt.savefig(fname+"_axon_"+ext+".png",dpi=dpi,bbox_inches='tight')
                if not showfig: plt.close(figname)
        if dir != "" and 'off' in axes:
            print("writing %s"%(fname+"_axoff_"+ext+".png"))
            plt.imsave(arr=arr[mode],origin='lower',fname=fname+"_axoff_"+ext+".png",dpi=dpi if dpi!=None else 200)
