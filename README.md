# Predictions of Peak Warming

***
Neural networks are trained on CMIP6 data to predict the maximum warming that will be reached under different scenarios through 2100.

## Pytorch Code

***
This code was written in python 3.12.0, pytorch 2.1.2, xarray 2024.1.1 and numpy 1.26.4.

### Python Environment

The following python environment was used to implement this code.

```bash
conda create --name env-torch-shash python=3.12.0
conda activate env-torch-shash
conda install numpy scipy pandas matplotlib seaborn statsmodels palettable progressbar2 flake8 jupyterlab black isort jupyterlab_code_formatter xarray scikit-learn cartopy netCDF4 geopandas nc-time-axis pytorch
pip install ipython-autotime cmocean cmasher cmaps captum ipywidgets torchinfo shapely regionmask pyarrow
```

### Run the Code

Run the script using Python and pass the experiment name as an argument. The experiment name is used to specify the config file. For example, if your experiment name is exp101, you would run:

```python train.py exp101```

This will start the script and use the configuration specified in the exp101 config file.

## Credits

***
This work is a collaborative effort between [Dr. Noah Diffenbaugh](https://earth.stanford.edu/people/noah-diffenbaugh#gs.runods) and  [Dr. Elizabeth A. Barnes](https://barnes.atmos.colostate.edu).

### References

[1] Diffenbaugh and Barnes, 2024. "Data-Driven Predictions of Peak Warming under Rapid Decarbonization." Geophysical Research Letters, in press.

### License

This project is licensed under an MIT license.

MIT © [Elizabeth A. Barnes](https://github.com/eabarnes1010)
