# Installation

**Step 1.** Conda env
```
conda create -n acemob

```

**Step 2.** Install pytorch

**Step 3.** Conda and Pip requirements
```
conda install -c conda-forge opencv
conda install -c anaconda pandas
conda install -c conda-forge matplotlib
```
```
pip install pyyaml tqdm seaborn protobuf ipython psutil
```

**Step 4.** Get weights
```
mkdir weights
cd weights
wget --no-check-certificate 'https://drive.google.com/u/0/uc?id=1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb&export=download&confirm=t&uuid=2a0362c3-7ff7-4306-bd71-dec6669cb415' -O crowdhuman_yolov5m.pt