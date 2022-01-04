# ROTS
Python Implementation of RObustTraining forTime-Series (ROTS) for the paper: "[Training Robust Deep Models for Time-Series Domain: Novel Algorithms and Theoretical Analysis]" by Taha Belkhouja, Yan Yan, and Janardhan Rao Doppa.

## Setup 
```
pip install -r requirements.txt
```
- By default, data is stored in `experim_path_{dataset_name}`. Directory can be changed in `ROTS.py`

## Run
- Example run
```python ROTS.py --dataset_name=SyntheticControl --K=50 --rots_beta=5e-2```
