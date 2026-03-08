# End-to-end Design of LED Array Patterns for 3D Differential Phase Contrast

## Dependencies

**Setup:** `uv venv && source .venv/bin/activate && uv sync && uv run jupyter notebook 3D_DPC_illumination_design.ipynb`

## Run

Follow the `3D_DPC_illumination_design.ipynb` notebook to run step-by-step. The notebook is also runnable in Google Colab.

## LED Positions

The NA of each LED is needed for the optimization. The NA information is loaded from a json file, like the provided example `led_array_pos_na_z65mm.json`. If you are using a LED array from Zack (Sci Microscopy), [the controller firmware](https://github.com/sci-microscopy/illuminate) will print you one.

## Important considerations

The optimization is dependent on the device parameters and noise profile. You will need to plug in your own settings to the notebook and as well as `DesignMotion3DDPCIllumination`:

```python
self.sensor_gain = 2.87
self.source_to_intensity_ratio = 65.2 * intensity_coef
self.readout_noise_std = 1.35
```

## Example patterns

The final patterns used in our paper is attached in `optimized_4patterns.json` as a reference. The optimization is for our 40x, 0.65 NA system.

## Paper

```
@article{cao2022self,
  title={Self-calibrated 3D differential phase contrast microscopy with optimized illumination},
  author={Cao, Ruiming and Kellman, Michael and Ren, David and Eckert, Regina and Waller, Laura},
  journal={Biomedical Optics Express},
  volume={13},
  number={3},
  pages={1671--1684},
  year={2022},
  publisher={Optica Publishing Group}
}
```
