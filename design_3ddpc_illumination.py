# Description:
#  Physics based learning to identify the optimal DPC illumination for 3D DPC at a given motion speed (axial sampling
#  rate)
# Created by Ruiming Cao on July 11, 2020
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import numpy as np
import tensorflow as tf
from solver_3ddpc import Solver3DDPC
from opticstools import _genGrid
# tf.config.set_visible_devices([], 'GPU') # cpu only
from multiprocessing import Pool

pi    = np.pi
naxis = np.newaxis


class DesignMotion3DDPCIllumination(Solver3DDPC):
    def __init__(self, dim_yxz, wavelength, na, pixel_size, pixel_size_z, RI_medium,
                 source_mask=None, num_illu=4, intensity_coef=1.0):
        super().__init__(dpc_imgs=np.zeros(dim_yxz + (1,)), wavelength=wavelength, na=na, na_in=0.0,
                         pixel_size=pixel_size, pixel_size_z=pixel_size_z,rotation=[0], RI_medium=RI_medium,
                         dim_z=dim_yxz[2])

        self.is_noise = True
        if intensity_coef == 0:
            self.is_noise = False
            intensity_coef = 1.0

        # calibration may be needed
        self.intensity_coef = intensity_coef
        self.sensor_gain = 2.87
        self.source_to_intensity_ratio = 65.2 * intensity_coef
        self.readout_noise_std = 1.35

        if source_mask is None:
            self.source_mask = self.pupil_mask
        else:
            self.source_mask = source_mask * self.pupil_mask
        self.num_illu = num_illu

    def get_illu_pattern_bases(self, max_na, delta_na, delta_theta, half_circle=False, na_levels=None):
        if half_circle:
            angles = np.arange(pi //delta_theta) * delta_theta - 0.5 * pi
        else:
            angles = _genGrid(2 * pi //delta_theta, delta_theta)

        if na_levels is None:
            na_levels = np.arange(0, max_na // delta_na).astype(float) * delta_na
        else:
            delta_na = na_levels[1:] - na_levels[:-1]
            na_levels = na_levels[:-1]

        bases =  np.array((self.fxlin[naxis, naxis, :] ** 2 + self.fylin[naxis, :, naxis] ** 2 <=
                 ((na_levels+delta_na)[:, naxis, naxis] / self.wavelength) ** 2) *
                          (self.fxlin[naxis, naxis, :] ** 2 + self.fylin[naxis, :, naxis] ** 2 >=
                 (na_levels[:, naxis, naxis] / self.wavelength) ** 2), dtype="float32")
        bases = bases[:,naxis,:,:] * ((np.arctan2(self.fylin[:,naxis].real+1e-5, self.fxlin[naxis,:].real+1e-5)[naxis,] > angles[:,naxis, naxis]) *
                                          (np.arctan2(self.fylin[:,naxis].real+1e-5, self.fxlin[naxis,:].real+1e-5)[naxis,] <= (angles + delta_theta)[:,naxis, naxis]))[naxis,:,:,:]
        bases = np.reshape(bases, [-1, self.dim_y, self.dim_x])

        return bases

    def _generate_sphere(self, xs, ys, zs, delta_RI, delta_attenu):
        RI_real, RI_imag = np.ones((self.dim_y, self.dim_x, self.dim_z)) * self.RI_medium, np.zeros((self.dim_y, self.dim_x, self.dim_z))
        obj_spacing = 15.0
        max_rad, min_rad = 6.0, 3.0
        k_0 = 2 * np.pi / self.wavelength

        num_x = int(self.dim_x * self.pixel_size // obj_spacing)
        num_y = int(self.dim_y * self.pixel_size // obj_spacing)
        num_z = 1
        for i in range(num_x):
            for j in range(num_y):
                for k in range(num_z):
                    radius = np.random.uniform(min_rad, max_rad)
                    center_xyz = [np.random.uniform(obj_spacing * i + max_rad, obj_spacing * i + (obj_spacing - max_rad)) + np.min(xs),
                                  np.random.uniform(obj_spacing * j + max_rad, obj_spacing * j + (obj_spacing - max_rad)) + np.min(ys),
                                  np.random.uniform(obj_spacing * k + max_rad, obj_spacing * k + (obj_spacing - max_rad)) + np.min(zs)]
                    if num_z == 1:
                        center_xyz[2] = 0.5 * (np.max(zs) + np.min(zs)) + np.random.uniform(-2.0, 2.0)

                    # phase volume
                    RI_real[(ys - center_xyz[1])[:, naxis, naxis] ** 2 + (xs - center_xyz[0])[naxis, :, naxis] ** 2 +
                           (zs - center_xyz[2])[naxis, naxis, :] ** 2 < radius ** 2] = self.RI_medium + delta_RI
                    # random little spheres to add complicity
                    for _ in range(int(radius ** 3 // 12)):
                        radius_sub = np.clip(np.random.normal(0.9, 0.3), 0.4, 1.4)
                        distance = np.sqrt(np.random.uniform(0.0, (radius - radius_sub) ** 2))
                        angle1 = np.pi * np.random.uniform(0.0, 2.0)
                        angle2 = np.pi * np.random.uniform(0.0, 2.0)
                        random_xyz = [distance * np.cos(angle1) * np.cos(angle2),
                                      distance * np.sin(angle1) * np.cos(angle2), distance * np.sin(angle2)]
                        random_xyz = np.array(random_xyz) + np.array(center_xyz)

                        RI_real[(ys - random_xyz[1])[:, naxis, naxis] ** 2 + (xs - random_xyz[0])[naxis, :, naxis] ** 2 +
                               (zs - random_xyz[2])[naxis, naxis, :] ** 2 < radius_sub ** 2] = self.RI_medium + 2 * delta_RI

                    V_real = k_0 ** 2 * (self.RI_medium ** 2 - RI_real ** 2)

                    # absorption surface
                    RI_imag[((ys - center_xyz[1])[:, naxis, naxis] ** 2 + (xs - center_xyz[0])[naxis, :, naxis] ** 2 +
                            zs[naxis, naxis, :] ** 2 < radius ** 2) & ((ys - center_xyz[1])[:, naxis, naxis] ** 2 + (xs - center_xyz[0])[naxis, :,naxis] ** 2 + zs[naxis, naxis, :] ** 2 > (radius - 2e-1) ** 2)] = delta_attenu
                    V_imag = -2.0 * (k_0 ** 2) * RI_imag * RI_real
        return V_real, V_imag

    def _generate_multi_layer_sphere(self, xs, ys, zs, delta_RI, num_layer):
        V_real, V_imag = np.zeros((self.dim_y, self.dim_x, self.dim_z)), np.zeros((self.dim_y, self.dim_x, self.dim_z))
        obj_spacing = 5.0
        radius = 2.0
        z_range = 3.0
        num_x = int(self.dim_x * self.pixel_size // obj_spacing)
        num_y = int(self.dim_y * self.pixel_size // obj_spacing)
        for k in range(num_layer):
            z_mid = (np.max(zs) - np.min(zs)) / num_layer * (k + 0.5) + np.min(zs)
            for i in range(num_x):
                for j in range(num_y):
                    center_xyz = [np.random.uniform(obj_spacing * i + radius, obj_spacing * i + (obj_spacing - radius)) + np.min(xs),
                                  np.random.uniform(obj_spacing * j + radius, obj_spacing * j + (obj_spacing - radius)) + np.min(ys),
                                  np.random.uniform(z_mid - z_range, z_mid + z_range)]

                    # phase volume
                    V_real[(ys - center_xyz[1])[:, naxis, naxis] ** 2 + (xs - center_xyz[0])[naxis, :, naxis] ** 2 +
                           (zs - center_xyz[2])[naxis, naxis, :] ** 2 < radius ** 2] = (2 * np.pi / self.wavelength) ** 2 * (self.RI_medium ** 2 - (self.RI_medium + delta_RI) ** 2)

    def generate_object_scattering_potential(self, num_object, delta_RI=1e-2, delta_attenu=0e-3, add_noise=False,
                                             volume_type='sphere'):
        xs = _genGrid(self.dim_x, self.pixel_size)
        ys = _genGrid(self.dim_y, self.pixel_size)
        zs = _genGrid(self.dim_z, self.pixel_size_z)

        V_objects = []
        RI_objects = []
        for _ in range(num_object):
            V_real, V_imag = self._generate_sphere(xs, ys, zs, delta_RI, delta_attenu)
            if add_noise:
                V_real = V_real - np.random.uniform(0.0, 0.25, size=V_real.shape)
            V_objects.append(V_real + 1.0j * V_imag)
            RI_objects.append(self._V2RI(V_real, V_imag))

        return V_objects, RI_objects

    @tf.function
    def _forward(self, sources_tf, V_object_real_tf_, V_object_imag_tf_, undersample_matrix):
        # forward pass
        H_real_tf_, H_imag_tf_ = self.WOTFGen_tf(sources_tf)
        fI_tf = H_real_tf_ * tf.stop_gradient(
            tf.expand_dims(tf.signal.fft3d(tf.complex(V_object_real_tf_, 0.0)), 0)) + H_imag_tf_ * tf.stop_gradient(tf.expand_dims(
                tf.signal.fft3d(tf.complex(V_object_imag_tf_, 0.0)), 0))

        # noise
        noise_signal = tf.random.normal(shape=tf.shape(fI_tf), mean=0.0, stddev=1.0, dtype=tf.float32) * tf.expand_dims(
            tf.sqrt(tf.constant(self.sensor_gain / self.source_to_intensity_ratio, dtype=tf.float32) / tf.maximum(
                tf.reduce_sum(tf.constant(self.pupil_mask_xl[np.newaxis]) * sources_tf, axis=(1, 2), keepdims=True),
                20.0)), 3)
        noise_readout = tf.random.normal(shape=tf.shape(fI_tf), mean=0.0, stddev=1.0, dtype=tf.float32) * tf.expand_dims(tf.constant(self.readout_noise_std / self.source_to_intensity_ratio, dtype=tf.float32) / tf.maximum(tf.reduce_sum(tf.constant(self.pupil_mask_xl[np.newaxis]) * sources_tf, axis=(1,2), keepdims=True), 20.0),3)

        # sampling
        fI_tf_ = tf.cond(tf.constant(self.is_noise),
                         lambda: tf.signal.fft3d(tf.complex(tf.math.real(tf.signal.ifft3d(fI_tf)) + noise_signal + noise_readout, 0.0) * tf.expand_dims(tf.expand_dims(undersample_matrix,1),2)),
                         lambda: tf.signal.fft3d(tf.complex(tf.math.real(tf.signal.ifft3d(fI_tf)) , 0.0) * tf.expand_dims(tf.expand_dims(undersample_matrix,1),2)))

        return fI_tf_, H_real_tf_, H_imag_tf_

    @tf.function
    def _tik_forward_inverse(self,sources_tf, V_object_real_tf_, V_object_imag_tf_, undersample_matrix, weights_tf):
        # forward pass
        fI_tf_, H_real_tf_, H_imag_tf_ = self._forward(sources_tf, V_object_real_tf_, V_object_imag_tf_, undersample_matrix)
        # reconstruction
        V_recon_real_tf, V_recon_imag_tf = self.solve_tikhonov_tf_(fInt_tf=fI_tf_, H_real_tf=H_real_tf_, H_imag_tf=H_imag_tf_, weights_tf=weights_tf)
        return V_recon_real_tf, V_recon_imag_tf, fI_tf_

    def _tik_forward_inverse_mismatch(self, sources_forward_tf, sources_recon_tf, V_object_real_tf_, V_object_imag_tf_, undersample_matrix, weights_tf):
        # forward pass
        fI_tf_, H_real_tf_, H_imag_tf_ = self._forward(sources_forward_tf, V_object_real_tf_, V_object_imag_tf_, undersample_matrix)

        # reconstruction
        H_real_recon_tf, H_imag_recon_tf = self.WOTFGen_tf(sources_recon_tf)
        V_recon_real_tf, V_recon_imag_tf = self.solve_tikhonov_tf_(fInt_tf=fI_tf_, H_real_tf=H_real_recon_tf,
                                                                   H_imag_tf=H_imag_recon_tf, weights_tf=weights_tf)
        return V_recon_real_tf, V_recon_imag_tf, fI_tf_

    def optimize_illu_pattern_tikhonov(self, illu_bases, init_coef=None, V_objects=None, lr=1e-2,
                                       iters=10, batch_size=16, imperfect_mode=False, dims_z=None, shift_range=3):
        # init optimizer
        optimizer_illu = tf.keras.optimizers.Adam(learning_rate=lr)

        # initialize objects
        if V_objects is None:
            V_objects, _ = self.generate_object_scattering_potential(batch_size, delta_RI=5e-3)

        # sub-sampling setting
        if dims_z is None:
            dims_z = [self.dim_img_z]
        assert isinstance(dims_z, list)

        if illu_bases is False:
            is_pixel_source = True
            assert init_coef is not None
            assert init_coef.shape == (self.num_illu, self.dim_y, self.dim_x)
            illu_coef_tf = tf.Variable(init_coef.astype(np.float32), trainable=True)
        else:
            raise NotImplementedError

        num_illu = self.num_illu

        # define undersampling matrix
        undersample_matrices = np.zeros((len(dims_z), num_illu, self.dim_z), dtype=np.complex64)
        for j in range(len(dims_z)):
            sampling_ratio = self.dim_z / dims_z[j]
            if sampling_ratio == 2.0:
                undersample_matrices[j, :int(num_illu/2),::int(sampling_ratio)] = 1.0
                undersample_matrices[j, int(num_illu/2):,1::int(sampling_ratio)] = 1.0
            elif sampling_ratio%num_illu == 0:
                for i in range(num_illu):
                    undersample_matrices[j, i, i * int(sampling_ratio//num_illu)::int(sampling_ratio)] = 1.0
            elif sampling_ratio > 1.0:
                indices = np.round(np.linspace(0.0, self.dim_z - 1, dims_z[j] * num_illu)).astype(int)
                for i in range(num_illu):
                    undersample_matrices[j, i, indices[i::num_illu]] = 1.0
            else:
                undersample_matrices[j, :, ::int(sampling_ratio)] = 1.0
            assert np.all(np.sum(undersample_matrices[j], 1) == dims_z[j])
            undersample_matrices[j] = undersample_matrices[j] * sampling_ratio

        # for imperfect mode
        shift_conv_size = shift_range * 2 -1
        shift_conv_filter = np.zeros((shift_conv_size ** 2))
        shift_conv_filter[shift_conv_size ** 2//2] = 1.0

        error = []
        error_l2 = []
        illu_coefs = []
        error_batch, loss_l2_batch, loss_infnorm_batch = tf.Variable(0.0), tf.Variable(0.0), tf.Variable(0.0)
        for step in range(iters):
            undersample_matrix = undersample_matrices[np.random.randint(undersample_matrices.shape[0])]

            illu_coefs.append(illu_coef_tf.numpy())
            accu_grad_illu_coef = tf.Variable(tf.zeros_like(illu_coef_tf), trainable=False)
            error_batch = 0.0
            loss_l2_batch = 0.0
            loss_infnorm_batch = 0.0

            for i in range(len(V_objects)):
                cur_V = V_objects[i]
                RI_object_tf = tf.constant(self._V2RI(cur_V.real, cur_V.imag), dtype=tf.float32)
                V_object_real_tf = tf.convert_to_tensor(cur_V.real, dtype=tf.float32)
                V_object_imag_tf = tf.convert_to_tensor(cur_V.imag, dtype=tf.float32)
                with tf.GradientTape(persistent=False) as grad_tape:
                    # sources
                    if tf.math.equal(tf.constant(is_pixel_source), tf.constant(True)):
                        sources_tf = illu_coef_tf
                    elif tf.math.equal(tf.constant(len(init_coef.shape)), 1):
                        sources_tf = tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf.expand_dims(illu_coef_tf,0),2),3) * tf.constant(illu_bases),axis=1)
                    else:
                        sources_tf = tf.reduce_sum(tf.expand_dims(tf.expand_dims(illu_coef_tf,2),3) * tf.constant(illu_bases), axis=1)

                    sources_tf = sources_tf * tf.constant(self.source_mask[np.newaxis,:,:])

                    pattern_weights_tf = tf.complex(tf.reduce_sum(sources_tf,(1,2)) * 4.0 / tf.reduce_sum(sources_tf, (0,1,2)), 0.0)

                    if imperfect_mode:
                        varying_intensity_matrix = tf.clip_by_value(tf.random.normal((self.dim_y, self.dim_x), mean=1.0, stddev=0.05), 0.9, 1.1)
                        shift_conv_filter_tf = tf.reshape(tf.cond(tf.less(tf.random.uniform([],0.0,1.0),0.6),
                                                          lambda : shift_conv_filter.astype(np.float32),
                                                          lambda : tf.random.shuffle(shift_conv_filter.astype(np.float32))), (shift_conv_size,shift_conv_size))

                        sources_forward_tf = tf.signal.fftshift(tf.squeeze(tf.nn.conv2d(tf.signal.ifftshift(sources_tf)[:,:,:,tf.newaxis], shift_conv_filter_tf[:,:,tf.newaxis, tf.newaxis], strides=1, padding='SAME'),3))
                        sources_forward_tf = varying_intensity_matrix[tf.newaxis,:,:] * sources_forward_tf

                        # binary
                        if tf.less(tf.random.uniform([],0.0, 1.9), 0.3):
                            sources_forward_tf = tf.where(tf.less(sources_forward_tf, 0.6), tf.zeros_like(sources_forward_tf), tf.ones_like(sources_forward_tf))

                        V_recon_real_tf, V_recon_imag_tf, _ = self._tik_forward_inverse_mismatch(sources_forward_tf, sources_tf, V_object_real_tf, V_object_imag_tf, tf.constant(undersample_matrix), pattern_weights_tf)
                    else:
                        V_recon_real_tf, V_recon_imag_tf, _ = self._tik_forward_inverse(sources_tf, V_object_real_tf, V_object_imag_tf, tf.constant(undersample_matrix), pattern_weights_tf)
                    RI_recon_tf = self._V2RI_tf(V_recon_real_tf, V_recon_imag_tf)

                    # loss
                    loss_l2 = tf.reduce_mean((1e3*RI_recon_tf - 1e3*RI_object_tf)**2)
                    loss_infnorm = 5e-2 * (tf.reduce_sum(tf.maximum(sources_tf - 1.0 - 1e-5, 0.0)) + tf.reduce_sum(tf.maximum(-sources_tf, 1e-5)))
                    loss = loss_l2 + loss_infnorm

                grad = grad_tape.gradient(loss, illu_coef_tf)
                accu_grad_illu_coef.assign_add(grad)

                error_batch = error_batch + loss
                loss_l2_batch = loss_l2_batch + loss_l2
                loss_infnorm_batch = loss_infnorm_batch + loss_infnorm

            optimizer_illu.apply_gradients([(accu_grad_illu_coef, illu_coef_tf)])
            tf.print("Step: ", step, ", loss: ", error_batch, ", loss_l2: ", loss_l2_batch, ", loss_infnorm:", loss_infnorm_batch)
            error.append(error_batch.numpy())
            error_l2.append(loss_l2_batch.numpy())

        illu_coefs.append(illu_coef_tf.numpy())
        return illu_coefs, error_l2
