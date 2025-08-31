 Cotton-Leaf Point Cloud Completion (C3D)

1) Environment
EN: Python ≥ 3.8, PyTorch (GPU recommended), CUDA matching your PyTorch.

```bash
conda create -n c3d python=3.8 -y
conda activate c3d
pip install torch torchvision  
pip install numpy h5py easydict
```

2) Dataset layout (HDF5)
Place files under `../data_ourh5/TrainPoint/Completion3D_airplane/`:

```
Completion3D_airplane/
  train/|val/|test/
    partial_input/<class>/<id>.h5
    partial_dense/<class>/<id>.h5
    partial_bilinear/<class>/<id>.h5
    drop_dense/<class>/<id>.h5
    gt_half/<class>/<id>.h5
```


3) Configure
Edit `config_c3d.py`*:

EN:
  * Set dataset paths (root & templates)
  * `CONST.DEVICE` (e.g., `"0,1"`), `DIR.OUT_PATH`
  * `CONST.N_INPUT_POINTS` (e.g., 4096)
  * `NETWORK.N_SAMPLING_POINTS` (e.g., 2048)
  * For val/test set `CONST.WEIGHTS` to a checkpoint path

4) Run 
```bash
# Train 
python main_c3d.py

# Validate 
python main_c3d.py --val

# Test
python main_c3d.py --test
```
EN: Logs & checkpoints are saved under `outpath/` (see config).


5) Data making scripts 
EN: Your dataset maker is provided in `makeoursData_ourh5.zip` .
