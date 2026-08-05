"""Tools to broaden and shift templates. 
   Originally from IronFit by Zhefu Yu"""


import numpy as np 
import astropy
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve, convolve_fft
          
def broaden_template(lbdtpl,ftpl,wtpl,pixtpl=1.,brdspace='linear'):
    '''
    Broaden a template

    Input:
    - lbdtpl,ftpl: iron template
    - wtpl: the smoothing width
    - pixtpl: the pixel size of the iron template (in wavelength unit)
    - brdspace: broaden the template in which space;
                choose between "linear"/"log", i.e., wavelength/velocity space
                (here "log" refers to ln, not log10)
    ***
    if broadening in wavelength space:
      - "lbdtpl","ftpl" should be evenly sampled in linear wavelength space
      - "pixtpl", "wtpl" in wavelength unit
    if broadening in log wavelength (i.e., velocity) space:
      - "lbdtpl","ftpl" should be evenly sampled in LOG wavelength space
      - "wtpl" in km/s
      - "pixtpl" is not used in the calculation. 
        The pixel size in log space is directly calculated using "lbdtpl"
    ***
    
    Output:
    - ftpl_colv: the broadened iron template 
    (same sampling as the input template)
    '''
    #NOTE: the input width for the kernel is in unit of PIXEL, not wavelength!
    if brdspace=='linear':
        wtpl_pix = wtpl / pixtpl
        gkernel = Gaussian1DKernel(stddev=wtpl_pix)
        ftpl_colv = convolve(ftpl,gkernel)
    elif brdspace=='log':
        # Calculate the width in pixels
        c = 2.99792458e5 #km/s
        lnlbd = np.log(lbdtpl)
        pixln = abs(lnlbd[1] - lnlbd[0])
        wtpl_pix = (wtpl/c) / pixln
        # Convolve in the lbd * F_lbd vs. ln(lbd) space
        gkernel = Gaussian1DKernel(stddev=wtpl_pix)
        lbdftpl_colv = convolve(lbdtpl*ftpl,gkernel)
        ftpl_colv = lbdftpl_colv / lbdtpl

    else:
        raise ValueError('Unknown space for broadening!')

    return ftpl_colv


def shift_template(lbdtpl,dx,shfspace='linear'):
    '''
    Shift a template

    Input:
    - lbdtpl: wavelength array of the iron template
    - dx: displacement of the template
    - shfpace: shift the template in which space;
                choose between "linear"/"log", i.e., wavelength/velocity space
                (here "log" refers to ln, not log10)
    ***
    if shifting in wavelength space: "dx" in wavelength unit
    if shifting in log wavelength (i.e., velocity) space: "dx" in km/s
    ***

    Output:
    - lbdtpl_shf: wavelength array of the shifted template
    '''
    if shfspace=='linear':
        lbdtpl_shf = lbdtpl + dx
    elif shfspace=='log':
        c = 2.99792458e5 #km/s
        dlnlbd = dx / c
        lnlbd_shf = np.log(lbdtpl) + dlnlbd
        lbdtpl_shf = np.exp(lnlbd_shf)
    else:
        raise ValueError('Unknown space for shifting!')
    return lbdtpl_shf


