# Robust Training for Time-Series
Python Implementation of RObustTraining forTime-Series (RO-TS) for the paper: "[Training Robust Deep Models for Time-Series Domain: Novel Algorithms and Theoretical Analysis]()" by Taha Belkhouja, Yan Yan, and Janardhan Rao Doppa.

## Setup 
```
pip install -r requirement.txt
```
By default, data is stored in `experim_path_{dataset_name}`. Directory can be changed in `RO_TS.py`


## Obtain datasets
- The dataset can be obtained as .zip file from "[The UCR Time Series Classification Repository](http://www.timeseriesclassification.com/dataset.php)".
- Download the .zip file and extract it it in `UCRDatasets/{dataset_name}` directory.
- Run the following command for pre-processing a given dataset while specifying if it is multivariate, for example, on SyntheticControl dataset
```
python preprocess_dataset.py --dataset_name=SyntheticControl --multivariate=False
```
The results will be stored in `Dataset` directory. 

## Run
- Example  training run
```
python RO_TS.py --dataset_name=SyntheticControl --K=10 --rots_beta=5e-1 --rots_lambda=1e-2 --batch=11
```

- Example testing run
```
python test_RO_TS_model.py --dataset_name=SyntheticControl --rots_beta=5e-1 --rots_lambda=1e-2
```

