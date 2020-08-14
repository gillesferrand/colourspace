# -*- coding: utf-8 -*-
"""
Conversion between colour spaces: CIE LCH <-> CIE Lab <-> CIE XYZ <->sRGB
Can use either:
- custom formulas taken from http://www.easyrgb.com/en/math.php
- wrapper to "colorspacious" package from https://pypi.python.org/pypi/colorspacious/
- wrapper to "colour" package from http://colour-science.org
"""

from __future__ import print_function
import numpy as np

convertor = None
LCH2RGB = lambda L,C,H: [np.nan, np.nan, np.nan]
RGB2LCH = lambda R,G,B: [np.nan, np.nan, np.nan]

def set_convertor(name):
    """ Binds conversion function LCH2RGB to the choosen package
        NB: all assume standard D65 illuminant
    """
    global LCH2RGB, RGB2LCH, convertor
    if name not in ['custom', 'colorspacious', 'colourscience']:
        print("Unknown conversion module")
        return
    convertor = name
    if name=='custom':
        LCH2RGB = lambda L,C,H: XYZ2RGB(Lab2XYZ(LCH2Lab((L,C,H))))
        RGB2LCH = lambda R,G,B: Lab2LCH(XYZ2Lab(RGB2XYZ((R,G,B))))
    if name=='colorspacious':
        from colorspacious import cspace_convert
        LCH2RGB = lambda L,C,H: cspace_convert(cspace_convert([L,C,H], "CIELCh", "XYZ100"), "XYZ100", "sRGB1")
        RGB2LCH = lambda R,G,B: cspace_convert(cspace_convert([R,G,B], "sRGB1", "XYZ100"), "XYZ100", "CIELCh")
    if name=='colourscience':
        import colour as colourscience
        illuminant = colourscience.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
        LCH2RGB = lambda L,C,H: colourscience.XYZ_to_sRGB(colourscience.Lab_to_XYZ(colourscience.LCHab_to_Lab([L,C,H]), illuminant=illuminant))
        RGB2LCH = lambda R,G,B: colourscience.Lab_to_LCHab(colourscience.XYZ_to_Lab(colourscience.sRGB_to_XYZ([R,G,B]), illuminant=illuminant))
    print("convertor = '",name,"'")

set_convertor('custom')

# cartesian CIE Lab <-> cylindrical CIE LCH

def LCH2Lab(LCH):
    L,C,H = LCH
    #L = float(L)
    a = C * np.cos(H*2*np.pi/360.)
    b = C * np.sin(H*2*np.pi/360.)
    return (L, a, b)

def Lab2LCH(Lab):
    L,a,b = Lab
    #L = float(L)
    C = np.sqrt(a**2+b**2)
    H = np.arctan2(b,a) * 360/(2*np.pi)
    H = np.where(H<0, H+360, H)
    return (L, C, H)

# perceptual CIE XYZ <-> uniform CIE Lab

# standard illuminants for 2° observer
Xn = {}
Yn = {}
Zn = {}
# D65
Xn['D65'] =  95.047
Yn['D65'] = 100.000
Zn['D65'] = 108.883
# D50
Xn['D50'] =  96.422
Yn['D50'] = 100.000
Zn['D50'] =  82.521
illuminant = 'D65'

def XYZ2Lab(XYZ):
    X,Y,Z = XYZ
    L = 116 *  f_forward(Y/Yn[illuminant]) - 16
    a = 500 * (f_forward(X/Xn[illuminant]) - f_forward(Y/Yn[illuminant]))
    b = 200 * (f_forward(Y/Yn[illuminant]) - f_forward(Z/Zn[illuminant]))
    return (L,a,b)

def Lab2XYZ(Lab):
    L,a,b = Lab
    X = Xn[illuminant] * f_reverse((L+16)/116.+a/500.)
    Y = Yn[illuminant] * f_reverse((L+16)/116.)
    Z = Zn[illuminant] * f_reverse((L+16)/116.-b/200.)
    return (X,Y,Z)

def f_forward(x):
    return np.where(x > (6/29.)**3, x**(1/3.), 1/3.*(29/6.)**2*x+4/29.)

def f_reverse(x):
    return np.where(x > 6/29., x**3, 3*(6/29.)**2*(x-4/29.))

# human CIE XYZ <-> machine sRGB

RGB_max = 1.

def XYZ2RGB(XYZ):
    X,Y,Z = XYZ
    R = RGB_max * gamma_forward( +3.2406 * X/100. -1.5372 * Y/100. -0.4986 * Z/100. )
    G = RGB_max * gamma_forward( -0.9689 * X/100. +1.8758 * Y/100. +0.0415 * Z/100. )
    B = RGB_max * gamma_forward( +0.0557 * X/100. -0.2040 * Y/100. +1.0570 * Z/100. )
    return (R,G,B)

def RGB2XYZ(RGB):
    R,G,B = RGB
    X = 100 * ( +0.4124 * gamma_reverse(R/RGB_max) +0.3576 * gamma_reverse(G/RGB_max) +0.1805 * gamma_reverse(B/RGB_max) )
    Y = 100 * ( +0.2126 * gamma_reverse(R/RGB_max) +0.7152 * gamma_reverse(G/RGB_max) +0.0722 * gamma_reverse(B/RGB_max) )
    Z = 100 * ( +0.0193 * gamma_reverse(R/RGB_max) +0.1192 * gamma_reverse(G/RGB_max) +0.9505 * gamma_reverse(B/RGB_max) )
    return (X,Y,Z)

def gamma_forward(Cln):
    mask = Cln <= 0.0031308
    g = np.empty_like(Cln)
    g[ mask] = 12.92*Cln[mask]
    g[~mask] = 1.055*(Cln[~mask]**(1/2.4))-0.055
    return g
    #return np.where(Cln <= 0.0031308, 12.92*Cln, 1.055*(Cln**(1/2.4))-0.055)

def gamma_reverse(Cnl):
    mask = Cnl <= 0.04045
    g = np.empty_like(Cnl)
    g[ mask] = Cnl[mask]/12.92
    g[~mask] = ((Cnl[~mask]+0.055)/1.055)**2.4
    return g
    #return np.where(Cnl <= 0.04045, Cnl/12.92, ((Cnl+0.055)/1.055)**2.4)

# RGB gamut check

def crop1(R,min=0,max=1):
    R_crop = np.where(R      < min, np.nan, R     )
    R_crop = np.where(R_crop > max, np.nan, R_crop)
    return R_crop

def crop3(RGB,min=0,max=1):
    R,G,B = RGB
    return np.stack((crop1(R,min,max),crop1(G,min,max),crop1(B,min,max)),axis=-1)

def clip1(R,min=0,max=1):
    R_clip = np.where(R      < min, min, R     )
    R_clip = np.where(R_clip > max, max, R_clip)
    return R_clip

def clip3(RGB,min=0,max=1):
    R,G,B = RGB
    return np.stack((clip1(R,min,max),clip1(G,min,max),clip1(B,min,max)),axis=-1)
