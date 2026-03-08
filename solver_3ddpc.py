# Description:
#   Solver for 3D Differential Phase Contrast (DPC) microscopy.
#   Adopted from Michael Chen's 3D DPC code.
# Created by Ruiming Cao on April 26, 2020
# Contact: rcao@berkeley.edu
# Website: https://rmcao.net

import numpy as np
import tensorflow as tf
from opticstools import _genGrid, genZernikeAberration

pi    = np.pi
naxis = np.newaxis
F_2D  = lambda x: np.fft.fft2(x, axes=(0, 1))
IF_2D = lambda x: np.fft.ifft2(x, axes=(0, 1))


def pupilGen(fxlin, fylin, wavelength, na, na_in=0.0):
    '''
    pupilGen create a circular pupil function in Fourier space.
    Inputs:
            fxlin     : 1D spatial frequency coordinate in horizontal direction
            fylin     : 1D spatial frequency coordinate in vertical direction
            wavelength: wavelength of incident light
            na        : numerical aperture of the imaging system
            na_in     : put a non-zero number smaller than na to generate an annular function
    Output:
            pupil     : pupil function
    '''
    pupil = np.array(fxlin[naxis, :]**2+fylin[:, naxis]**2 <= (na/wavelength)**2, dtype="float32")
    if na_in != 0.0:
        pupil[fxlin[naxis, :]**2+fylin[:, naxis]**2 < (na_in/wavelength)**2] = 0.0
    return pupil


class Solver3DDPC:
    '''
    Solver3DDPC provides methods to compute 3D DPC transfer functions and supports
    the illumination pattern optimization workflow.
    '''
    def __init__(self, dpc_imgs, wavelength, na, na_in, pixel_size, pixel_size_z, rotation, RI_medium, dim_z,
                 z_dist=None, na_illum_max=None, calibration_led_fxfy=None):
        '''
        Initialize system parameters and functions for DPC phase microscopy.
        '''
        self.wavelength     = wavelength
        self.na             = na
        self.na_in          = na_in
        self.na_illum_max   = na if na_illum_max is None else na_illum_max
        self.pixel_size     = pixel_size
        self.pixel_size_z   = pixel_size_z
        self.dim_z          = dim_z
        self.dim_img_z      = dpc_imgs.shape[2]
        self.dim_x          = dpc_imgs.shape[1]
        self.dim_y          = dpc_imgs.shape[0]
        self.rotation       = rotation
        self.fxlin          = np.fft.ifftshift(_genGrid(dpc_imgs.shape[1], 1.0/dpc_imgs.shape[1]/self.pixel_size))
        self.fylin          = np.fft.ifftshift(_genGrid(dpc_imgs.shape[0], 1.0/dpc_imgs.shape[0]/self.pixel_size))
        self.fzlin          = np.fft.ifftshift(_genGrid(dim_z, 1.0/dim_z/self.pixel_size_z))
        self.dpc_imgs       = dpc_imgs.astype('float32')
        self.RI_medium      = RI_medium

        self.window         = np.fft.ifftshift(np.hamming(self.dim_z))
        self.pupil          = pupilGen(self.fxlin, self.fylin, self.wavelength, self.na)
        self.pupil_mask     = pupilGen(self.fxlin, self.fylin, self.wavelength, self.na)
        self.pupil_mask_xl  = pupilGen(self.fxlin, self.fylin, self.wavelength, self.na + 0.05)

        zernike_order_min, zernike_order_max = 0, 9
        self.zernike_order_min = zernike_order_min
        self.zernike_order_max = zernike_order_max
        self.zernike_coeffs = np.zeros(zernike_order_max - zernike_order_min + 1).astype(np.float32)
        self.zernike_indices = np.arange(self.zernike_order_min, self.zernike_order_max + 1)
        self.zernike_bases = np.array([genZernikeAberration([self.dim_y, self.dim_x], self.pixel_size, self.na, self.wavelength,
                                                            z_coeff=[1], z_index_list=[i], ) for i in
                                       self.zernike_indices]).astype(np.complex64)

        self.phase_defocus  = self.pupil*2.0*pi*((1.0/wavelength)**2-self.fxlin[naxis, :]**2-self.fylin[:, naxis]**2)**0.5
        self.oblique_factor = self.pupil/4.0/pi/((RI_medium/wavelength)**2-self.fxlin[naxis, :]**2-self.fylin[:, naxis]**2)**0.5
        if np.max(self.dpc_imgs) > 100: # this is a hack
            self.normalization()

        source = []
        for i in range(len(self.rotation)):
            source.append(self.sourceGen(self.rotation[i], self.na_illum_max, self.na_in))
        if calibration_led_fxfy is not None:
            assert type(calibration_led_fxfy)==list
            for fxfy in calibration_led_fxfy:
                source.append(self.sourceGen(0, 0, 0, fxfy))
        self.source = np.array(source)

        if z_dist is None:
            self.equal_spaced = True
        else:
            self.z_lin = z_dist
            self.equal_spaced = False

        self.H_real, self.H_imag = [], []
        self.update_transfer_function(self.source)

        self.setRegularizationParameters()

    def update_transfer_function(self, source):
        self.source = source
        self.H_real, self.H_imag = self.WOTFGen(source)

    def set_pupil(self, zernike_coef=None):
        coef = self.zernike_coeffs if zernike_coef is None else zernike_coef
        self.pupil = np.exp(1.0j * np.sum(self.zernike_bases * coef[:, np.newaxis, np.newaxis], axis=0)) \
                     * self.pupil_mask

    def normalization(self):
        '''
        Normalize the 3D intensity stacks by their average illumination intensities, and subtract the mean.
        '''
        mean_ch = np.tile(np.array([np.mean(self.dpc_imgs[:, :, :, i][self.dpc_imgs[:, :, :, i] > 0])
                  for i in range(self.dpc_imgs.shape[3])])[np.newaxis, np.newaxis, np.newaxis, :], self.dpc_imgs.shape[:3]+ (1,))

        self.dpc_imgs  /= mean_ch
        self.dpc_imgs  -= 1.0

    def get_illu_pattern_bases_LEDdome(self, list_led_fxfy, large_led=False):
        s = np.zeros_like(self.pupil)
        for fxfy in list_led_fxfy:
            if fxfy[1]**2 + fxfy[0]**2 < (self.na_illum_max/self.wavelength)**2 and fxfy[1]**2 + fxfy[0]**2 >= (self.na_in/self.wavelength)**2:
                if large_led:
                    s[(self.fylin[:, np.newaxis] - fxfy[1])**2 + (self.fxlin[np.newaxis,:] - fxfy[0])**2 < 8e-4] = 1.0
                else:
                    s[np.unravel_index(np.argmin((self.fylin[:, np.newaxis] - fxfy[1]) ** 2 + (self.fxlin[np.newaxis, :] - fxfy[0]) ** 2),s.shape)] = 1.0
        s = s * self.pupil_mask
        return s

    def sourceGen(self, rotdegree, na_out, na_in, fxfy=None):
        '''
        Generate DPC source patterns based on the rotation angles and numerical aperture of the illuminations.
        '''
        pupil  = pupilGen(self.fxlin, self.fylin, self.wavelength, na_out, na_in=na_in)
        source = np.zeros((self.dpc_imgs.shape[:2]), dtype='float32')
        if fxfy is not None:
            source[(self.fylin[:, naxis] - fxfy[1]) ** 2 + (self.fxlin[naxis, :] - fxfy[0]) ** 2 < 5e-3] = 1.0
        elif rotdegree < 180:
            source[self.fylin[:, naxis]*np.cos(np.deg2rad(rotdegree))+1e-15>=
                            self.fxlin[naxis, :]*np.sin(np.deg2rad(rotdegree))] = 1.0
            source *= pupil
        else:
            source[self.fylin[:, naxis]*np.cos(np.deg2rad(rotdegree))+1e-15<
                            self.fxlin[naxis, :]*np.sin(np.deg2rad(rotdegree))] = -1.0
            source *= pupil
            source += pupil
        return source

    def sourceFlip(self, source):
        '''
        Flip the sources in vertical and horizontal directions, since the coordinates of the source plane and the pupil plane are opposite.
        '''
        source_flip = np.fft.fftshift(source)
        source_flip = source_flip[::-1, ::-1]
        if np.mod(source_flip.shape[0], 2)==0:
            source_flip = np.roll(source_flip, 1, axis=0)
        if np.mod(source_flip.shape[1], 2)==0:
            source_flip = np.roll(source_flip, 1, axis=1)

        return np.fft.ifftshift(source_flip)

    def WOTFGen(self, source):
        '''
        Generate the absorption (imaginary part) and phase (real part) weak object transfer functions (WOTFs) using the sources and the pupil.
        '''
        dim_x       = self.dim_x
        dim_y       = self.dim_y
        dfx         = 1.0/dim_x/self.pixel_size
        dfy         = 1.0/dim_y/self.pixel_size
        H_real = []
        H_imag = []
        for rot_index in range(source.shape[0]):
            z_lin = np.fft.ifftshift(_genGrid(self.dim_z, self.pixel_size_z))
            prop_kernel = np.exp(
                1.0j * np.abs(z_lin[naxis, naxis, :]) * self.phase_defocus[:, :, naxis])  # propagation on z-distance
            prop_kernel[:,:,z_lin<0] = np.conj(prop_kernel)[:,:,z_lin<0]

            source_flip      = self.sourceFlip(source[rot_index])
            FSP_cFPG         = F_2D(source_flip[:, :, naxis]*self.pupil[:, :, naxis]*prop_kernel)*\
                               F_2D(self.pupil[:, :, naxis]*prop_kernel*self.oblique_factor[:, :, naxis]).conj()
            H_real.append(2.0*IF_2D(1.0j*FSP_cFPG.imag*dfx*dfy))
            H_real[-1] *= self.window[naxis, naxis, :]
            H_real[-1]  = np.fft.fft(H_real[-1], axis=2)*self.pixel_size_z
            H_imag.append(2.0*IF_2D(FSP_cFPG.real*dfx*dfy))
            H_imag[-1] *= self.window[naxis, naxis, :]
            H_imag[-1]  = np.fft.fft(H_imag[-1], axis=2)*self.pixel_size_z
            total_source     = np.sum(source_flip*self.pupil*self.pupil.conj())*dfx*dfy
            H_real[-1] *= 1.0j/total_source
            H_imag[-1] *= 1.0 /total_source
            print("3D weak object transfer function {:02d}/{:02d} has been evaluated.".format(rot_index+1, source.shape[0]), end="\r")
        return np.array(H_real).astype(np.complex64), np.array(H_imag).astype(np.complex64)

    def _V2RI(self, V_real, V_imag):
        '''
        Convert the complex scattering potential (V) into the refractive index. Imaginary part of the refractive index is dumped.
        '''
        wavenumber  = 2.0*pi/self.wavelength
        B           = -1.0*(self.RI_medium**2-V_real/wavenumber**2)
        C           = -1.0*(-1.0*V_imag/wavenumber**2/2.0)**2
        RI_obj      = ((-1.0*B+(B**2-4.0*C)**0.5)/2.0)**0.5

        return RI_obj

    def _V2RI_tf(self, V_real_tf, V_imag_tf):
        '''
        Convert the complex scattering potential (V) into the refractive index. Imaginary part of the refractive index is dumped.
        '''
        wavenumber  = tf.constant(2.0*pi/self.wavelength)
        B           = -1.0*(tf.constant(self.RI_medium**2)-V_real_tf/wavenumber**2)
        C           = -1.0*(-1.0*V_imag_tf/wavenumber**2/2.0)**2
        RI_obj      = tf.sqrt((-1.0*B+(B**2-4.0*C)**0.5)/2.0)

        return RI_obj

    def setRegularizationParameters(self, reg_real=5e-5, reg_imag=5e-5, tau=5e-5, rho=5e-5, lr=1e-2, w_l2=1e2,
                                    w_tv=5e-2, w_nonpos=1e-3):
        '''
        Set regularization parameters for Tikhonov deconvolution.
        '''
        self.reg_real = reg_real
        self.reg_imag = reg_imag
        self.tau      = tau
        self.rho      = rho
        self.lr       = lr
        self.w_l2     = w_l2
        self.w_tv     = w_tv
        self.w_nonpos = w_nonpos

    @tf.function
    def solve_tikhonov_tf_(self, fInt_tf, H_real_tf, H_imag_tf, weights_tf):
        weights_tf = tf.expand_dims(tf.expand_dims(tf.expand_dims(weights_tf, 1), 2), 3)
        AHA_tf = [tf.reduce_sum((tf.math.conj(H_imag_tf) * H_imag_tf * weights_tf), 0) + tf.constant(self.reg_imag, dtype=tf.complex64),
                  tf.reduce_sum((tf.math.conj(H_imag_tf) * H_real_tf * weights_tf), 0),
                  tf.reduce_sum((tf.math.conj(H_real_tf) * H_imag_tf * weights_tf), 0),
                  tf.reduce_sum((tf.math.conj(H_real_tf) * H_real_tf * weights_tf), 0) + tf.constant(self.reg_real, dtype=tf.complex64)]
        determinant_tf = AHA_tf[0] * AHA_tf[3] - AHA_tf[1] * AHA_tf[2]
        AHy_tf         = [tf.reduce_sum(tf.math.conj(H_imag_tf) * fInt_tf * weights_tf, axis=0),
                          tf.reduce_sum(tf.math.conj(H_real_tf) * fInt_tf * weights_tf, axis=0)]
        V_real_tf      = tf.math.real(tf.signal.ifft3d((AHA_tf[0]*AHy_tf[1]-AHA_tf[2]*AHy_tf[0])/determinant_tf))
        V_imag_tf      = tf.math.real(tf.signal.ifft3d((AHA_tf[3]*AHy_tf[0]-AHA_tf[1]*AHy_tf[1])/determinant_tf))
        return V_real_tf, V_imag_tf

    @tf.function
    def WOTFGen_tf(self, sources_tf):
        '''
        Generate the absorption (imaginary part) and phase (real part) weak object transfer functions (WOTFs) using the sources and the pupil.
        '''
        dfx         = tf.constant(1.0/self.dpc_imgs.shape[1]/self.pixel_size, dtype=tf.float32)
        dfy         = tf.constant(1.0/self.dpc_imgs.shape[0]/self.pixel_size, dtype=tf.float32)
        z_lin_tf    = tf.constant(np.fft.ifftshift(_genGrid(self.dim_z, self.pixel_size_z)).real.astype(np.float32))
        phase_defocus_tf = tf.constant(self.phase_defocus[:, :, naxis],dtype=tf.complex64)
        prop_kernel_tf = tf.expand_dims(tf.exp(tf.complex(0.0, z_lin_tf)[tf.newaxis, tf.newaxis, :]*phase_defocus_tf),0) # propagation on z-distance

        sources_tf = tf.complex(sources_tf, 0.0)
        pupil_tf         = tf.expand_dims(tf.expand_dims(tf.constant(self.pupil,dtype=tf.complex64),-1),0)
        oblique_factor_tf= tf.expand_dims(tf.expand_dims(tf.constant(self.oblique_factor, dtype=tf.complex64), -1), 0)
        window_tf        = tf.constant(self.window[naxis,:,naxis, naxis], dtype=tf.complex64)

        FSP_cFPG         = tf.signal.fft2d(tf.transpose(tf.expand_dims(sources_tf,3)*pupil_tf*prop_kernel_tf, (0,3,1,2)))*tf.math.conj(tf.signal.fft2d(tf.transpose(pupil_tf*prop_kernel_tf*oblique_factor_tf, (0,3,1,2))))
        H_real_tf = tf.complex(2.0,0.0)*tf.signal.ifft2d(tf.complex(0.0,1.0)*tf.complex(tf.math.imag(FSP_cFPG)*dfx*dfy, 0.0)* window_tf)
        H_real_tf = tf.signal.fft(tf.transpose(H_real_tf, (0, 2, 3, 1))) * tf.constant(self.pixel_size_z, dtype=tf.complex64)

        H_imag_tf = tf.complex(2.0,0.0)*tf.signal.ifft2d(tf.complex(tf.math.real(FSP_cFPG)*dfx*dfy, 0.0) * window_tf)
        H_imag_tf = tf.signal.fft(tf.transpose(H_imag_tf, (0, 2, 3, 1))) * tf.constant(self.pixel_size_z, dtype=tf.complex64)
        total_source_tf     = tf.reduce_sum(tf.expand_dims(sources_tf,3)*pupil_tf*tf.math.conj(pupil_tf), [1,2,3])*tf.complex(dfx*dfy,0.0)
        H_real_tf = H_real_tf * tf.complex(0.0, 1.0)/tf.expand_dims(tf.expand_dims(tf.expand_dims(total_source_tf,1),2),3)
        H_imag_tf = H_imag_tf * 1.0 / tf.expand_dims(tf.expand_dims(tf.expand_dims(total_source_tf,1),2),3)

        return H_real_tf, H_imag_tf
