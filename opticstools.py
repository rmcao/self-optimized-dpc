'''
Copyright 2017 Waller Lab
The University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


import numpy as np
from math import factorial

pi = np.pi
naxis = np.newaxis
np_complex_datatype = np.complex128

def cartToNa(point_list_cart, z_offset=0):
    """Function which converts a list of cartesian points to numerical aperture (NA)

    Args:
        point_list_cart: List of (x,y,z) positions relative to the sample (origin)
        z_offset : Optional, offset of LED array in z, mm

    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (Na_x, NA_y)
    """
    yz = np.sqrt(point_list_cart[:, 1] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)
    xz = np.sqrt(point_list_cart[:, 0] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)

    result = np.zeros((np.size(point_list_cart, 0), 2))
    result[:, 0] = np.sin(np.arctan(point_list_cart[:, 0] / yz))
    result[:, 1] = np.sin(np.arctan(point_list_cart[:, 1] / xz))

    return(result)

def cart2Pol(x, y):
    rho          = (x * np.conj(x) + y * np.conj(y))**0.5
    theta        = np.arctan2(np.real(y), np.real(x)).astype(np_complex_datatype)
    return rho, theta

def genBayerCouplingMatrix(image_stack_rgb, pixel_offsets=(0, 0)):
    bayer_coupling_matrix = np.zeros((4,3))

    for color_index, frame in enumerate(image_stack_rgb):
        bayer_coupling_matrix[0, color_index] = np.mean(frame[pixel_offsets[0]::2, pixel_offsets[1]::2])
        bayer_coupling_matrix[1, color_index] = np.mean(frame[pixel_offsets[0] ::2, pixel_offsets[1] + 1::2])
        bayer_coupling_matrix[2, color_index] = np.mean(frame[pixel_offsets[0] + 1::2, pixel_offsets[1]::2])
        bayer_coupling_matrix[3, color_index] = np.mean(frame[pixel_offsets[0] + 1::2, pixel_offsets[1] + 1::2])

    return(bayer_coupling_matrix)

def genZernikeAberration(shape, pixel_size, NA, wavelength, z_coeff = [1], z_index_list = [0], fx_illu=0.0, fy_illu=0.0):
    assert len(z_coeff) == len(z_index_list), "number of coefficients does not match with number of zernike indices!"

    pupil             = genPupil(shape, pixel_size, NA, wavelength, fx_illu=fx_illu, fy_illu=fy_illu)
    fxlin             = _genGrid(shape[1], 1/pixel_size/shape[1], flag_shift = True) - fx_illu
    fylin             = _genGrid(shape[0], 1/pixel_size/shape[0], flag_shift = True) - fy_illu
    fxlin             = np.tile(fxlin[np.newaxis,:], [shape[0], 1])
    fylin             = np.tile(fylin[:, np.newaxis], [1, shape[1]])
    rho, theta        = cart2Pol(fxlin, fylin)
    rho[:, :]        /= NA/wavelength

    def zernikePolynomial(z_index):
        n                    = int(np.ceil((-3.0 + np.sqrt(9+8*z_index))/2.0))
        m                    = 2*z_index - n*(n+2)
        normalization_coeff  = np.sqrt(2 * (n+1)) if abs(m) > 0 else np.sqrt(n+1)
        azimuthal_function   = np.sin(abs(m)*theta) if m < 0 else np.cos(abs(m)*theta)
        zernike_poly         = np.zeros([shape[0], shape[1]], dtype = np_complex_datatype)
        for k in range((n-abs(m))//2+1):
            zernike_poly[:, :]  += ((-1)**k * factorial(n-k))/ \
                                    (factorial(k)*factorial(0.5*(n+m)-k)*factorial(0.5*(n-m)-k))\
                                    * rho**(n-2*k)

        return normalization_coeff * zernike_poly * azimuthal_function

    for z_coeff_index, z_index in enumerate(z_index_list):
        zernike_poly = zernikePolynomial(z_index)

        if z_coeff_index == 0:
            zernike_aberration = np.array(z_coeff).ravel()[z_coeff_index] * zernike_poly
        else:
            zernike_aberration += np.array(z_coeff).ravel()[z_coeff_index] * zernike_poly

    return zernike_aberration * pupil

def genPupil(shape, pixel_size, NA, wavelength, fx_illu = 0.0, fy_illu = 0.0, NA_in = 0.0):
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin        = np.fft.ifftshift(_genGrid(shape[1],1/pixel_size/shape[1]))
    fylin        = np.fft.ifftshift(_genGrid(shape[0],1/pixel_size/shape[0]))
    pupil_radius = NA/wavelength
    pupil        = np.asarray((fxlin[naxis,:] - fx_illu)**2 + (fylin[:,naxis] - fy_illu)**2 <= pupil_radius**2)
    if NA_in != 0.0:
        pupil[(fxlin[naxis,:] - fx_illu)**2 + (fylin[:,naxis]-fy_illu)**2 < pupil_radius**2] = 0.0
    return pupil

def propKernel(shape, pixel_size, wavelength, prop_distance, NA = None, RI = 1.0, fx_illu=0.0, fy_illu=0.0, band_limited=True):
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin        = np.fft.ifftshift(_genGrid(shape[1],1/pixel_size/shape[1]))
    fylin        = np.fft.ifftshift(_genGrid(shape[0],1/pixel_size/shape[0]))
    if band_limited:
        assert NA is not None, "need to provide numerical aperture of the system!"
        Pcrop = genPupil(shape, pixel_size, NA, wavelength, fx_illu = fx_illu, fy_illu = fy_illu)
    else:
        Pcrop = 1.0
    prop_kernel = Pcrop * np.exp(1j*2.0*pi*np.abs(prop_distance)*Pcrop*((RI/wavelength)**2 - (fxlin[naxis,:] - fx_illu)**2 - (fylin[:,naxis] - fy_illu)**2)**0.5)
    prop_kernel = prop_kernel.conj() if prop_distance < 0 else prop_kernel
    return prop_kernel

def _genGrid(size, dx, flag_shift = False):
    """
    This function generates 1D Fourier grid, and is centered at the middle of the array
    Inputs:
        size    - length of the array
        dx      - pixel size
    Optional parameters:
        flag_shift - flag indicating whether the final array is circularly shifted
                     should be false when computing real space coordinates
                     should be true when computing Fourier coordinates
    Outputs:
        kx      - 1D Fourier grid

    """
    xlin = (np.arange(size,dtype='complex128') - size//2) * dx
    if flag_shift:
        xlin = np.roll(xlin, (size)//2)
    return xlin

class Metadata:
    def __init__(self,shape,ps=6.5,mag=8.0,NA=1.0,wavelength=0.5,psz=None,RI=1.0,**kwargs):
        self.dim = shape
        self.mag = mag
        self.ps = ps/mag
        self.psz = psz
        self.NA = NA
        self.RI = RI
        self.wavelength = wavelength
        if len(shape)==1:
            self.dfx = 1.0/(shape[0]*self.ps)
            self.xlin = _genGrid(shape[0],self.ps)
            self.fxlin = np.fft.ifftshift(_genGrid(shape[0],self.dfx))
        elif len(shape)==2:
            self.xlin = _genGrid(shape[1],self.ps)
            self.ylin = _genGrid(shape[0],self.ps)
            self.dfx = 1.0/(shape[1]*self.ps)
            self.dfy = 1.0/(shape[0]*self.ps)
            self.fxlin = np.fft.ifftshift(_genGrid(shape[1],self.dfx))
            self.fylin = np.fft.ifftshift(_genGrid(shape[0],self.dfy))
        elif len(shape) ==3:
            assert psz is not None, "need pixel size on the thrid dimension!"
            self.xlin = _genGrid(shape[2],self.ps)
            self.ylin = _genGrid(shape[1],self.ps)
            self.zlin = _genGrid(shape[0],self.ps)
            self.dfx = 1.0/(shape[2]*self.ps)
            self.dfy = 1.0/(shape[1]*self.ps)
            self.dfz = 1.0/(shape[0]*self.psz)
            self.fxlin = np.fft.ifftshift(_genGrid(shape[2],self.dfx))
            self.fylin = np.fft.ifftshift(_genGrid(shape[1],self.dfy))
            self.fzlin = np.fft.ifftshift(_genGrid(shape[0],self.dfz))
        else:
            print('metadata does not support data beyond 3D!')
        if kwargs:
            print('problem dependent metadata:')
            for param in kwargs:
                print(param + ":" + str(kwargs[param]))
                if param == "z_planes":
                    self.z_planes = kwargs[param]   # through focus stack parameter
                elif param == "NA_in":
                    self.NA_in = kwargs[param]      # DPC parameter
                elif param == "rotation":
                    self.rotation = kwargs[param]   # DPC parameter
                else:
                    print("no parameter called " + param + " in metadata!")
                    return
