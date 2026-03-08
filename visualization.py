# Description:
#   Visualization utilities for 3D DPC illumination design results.
# Created for open-source release.
# Contact: rcao@berkeley.edu
# Website: https://rmcao.net

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def visualize_patterns(opt_coefs, source2source_highres_fn, title="", threshold=0.7):
    """
    Visualize the final optimized illumination patterns in a 1×N row figure.

    Args:
        opt_coefs: list of per-iteration coefficient arrays; uses opt_coefs[-1] (final iteration).
                   opt_coefs[-1] has shape (num_illu, H, W).
        source2source_highres_fn: callable that maps a low-res source pattern to high-res for display.
        title: optional figure title string.
        threshold: binarization threshold for displaying the final patterns (default 0.7).
    """
    final_patterns = np.array(opt_coefs[-1])  # shape (num_illu, H, W)
    num_illu = final_patterns.shape[0]

    normalize = mpl.colors.Normalize(vmin=-0.5, vmax=4.1)
    f, axes = plt.subplots(1, num_illu, figsize=(2.5 * num_illu, 2.5))
    if num_illu == 1:
        axes = [axes]

    for i in range(num_illu):
        pat = final_patterns[i] > threshold
        img = np.fft.ifftshift(source2source_highres_fn(pat))[40:610, 40:610][::-1, ::-1]
        axes[i].imshow(img, norm=normalize, cmap='viridis')
        axes[i].axis('off')

    if title:
        f.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_transfer_functions(H_real, final_coefs, design_obj, source2source_highres_fn,
                                  na, wavelength,
                                  fz_ind_min=18, fz_step=6, fz_num=5,
                                  clim=(-0.1, 0.1), threshold=0.7, save_path=None):
    """
    Visualize phase transfer function (TF) slices alongside illumination patterns.

    Args:
        H_real: complex array of shape (num_illu, H, W, Z) — phase weak object TFs.
        final_coefs: list of per-iteration coefficients; uses final_coefs[-1] (shape: num_illu × H × W).
        design_obj: Solver3DDPC object providing fxlin, fylin, fzlin attributes.
        source2source_highres_fn: callable mapping low-res source to high-res for pattern display.
        na: numerical aperture of the imaging system.
        wavelength: illumination wavelength (µm).
        fz_ind_min: starting z-frequency index for TF slice display (default 18).
        fz_step: step between z-frequency indices (default 6).
        fz_num: number of TF slices to display per row (default 5).
        clim: (lower, upper) color limits for TF display (default (-0.1, 0.1)).
        threshold: binarization threshold for pattern display (default 0.7).
        save_path: if provided, save figure to this path at 600 dpi.
    """
    # Compute missing cone geometry
    k_na = 2.0 * np.pi * na / wavelength
    k_0 = 2.0 * np.pi / wavelength
    fz = design_obj.fzlin.copy()
    fx_missingcone = (-np.sqrt((1.0 / wavelength)**2 -
                               (np.abs(fz) + np.sqrt(k_0**2 - k_na**2) / 2.0 / np.pi)**2)
                      + na / wavelength)
    fx_missingcone[np.isnan(fx_missingcone)] = 0.0

    # Prepare TF and pattern arrays
    Hs_vis = np.fft.fftshift(H_real.imag, axes=(1, 2, 3))
    patterns_binary = np.array(final_coefs[-1]) > threshold
    patterns_vis = np.fft.fftshift(patterns_binary, axes=(1, 2))

    vis_num = Hs_vis.shape[0]
    min_na_x = np.min(design_obj.fxlin.real)
    max_na_x = np.max(design_obj.fxlin.real)
    min_na_y = np.min(design_obj.fylin.real)
    max_na_y = np.max(design_obj.fylin.real)
    clim_l, clim_u = clim

    normalize = mpl.colors.Normalize(vmin=-0.5, vmax=4.1)
    f, ax = plt.subplots(vis_num, fz_num + 1,
                         figsize=(1.5 + 1.5 * (fz_num + 1), 1.5 + 1.5 * vis_num),
                         sharex=True, sharey=True)
    # Ensure ax is always 2D
    if vis_num == 1:
        ax = ax[np.newaxis, :]

    frames = []

    for j in range(vis_num):
        # Col 0: illumination pattern (high-res)
        pat_unshifted = np.fft.ifftshift(patterns_vis[j], axes=(0, 1))
        highres_pat = source2source_highres_fn(pat_unshifted)[::-1, ::-1]
        ax[j, 0].imshow(np.fft.ifftshift(highres_pat, axes=(0, 1)),
                        extent=[-1.2, 1.2, -1.2, 1.2], norm=normalize, cmap='viridis')
        ax[j, 0].axis("off")

        # Cols 1..fz_num: phase TF slices
        for i in range(fz_num):
            fz_ind = fz_ind_min + i * fz_step
            frames.append(ax[j, i + 1].imshow(
                Hs_vis[j, :, :, fz_ind], cmap='bwr',
                extent=[min_na_x, max_na_x, min_na_y, max_na_y],
                clim=(clim_l, clim_u)))
            ax[j, i + 1].set_xlim(-1.2, 1.2)
            ax[j, i + 1].set_ylim(-1.2, 1.2)
            ax[j, i + 1].axis("off")
            ax[j, i + 1].set_aspect(1)

            # Missing cone overlay
            mc_size = np.fft.fftshift(fx_missingcone)[fz_ind].real
            circle_fill = plt.Circle((0, 0), mc_size, color='white', linewidth=0, alpha=0.8, fill=True)
            ax[j, i + 1].add_artist(circle_fill)
            circle_contour = plt.Circle((0, 0), mc_size, color='black', linestyle='--', fill=False)
            ax[j, i + 1].add_artist(circle_contour)

            if j == 0:
                fz_val = np.fft.fftshift(fz).real[fz_ind]
                ax[j, i + 1].set_title('fz= {:.02f}µm⁻¹'.format(fz_val))

    ax[0, 0].set_title('illumin. pattern')
    f.tight_layout(pad=0.3)
    f.colorbar(frames[-1], ax=ax, ticks=[clim_l, 0, clim_u],
               fraction=0.02, aspect=5, shrink=0.2)

    if save_path is not None:
        f.savefig(save_path, dpi=600)

    plt.show()
