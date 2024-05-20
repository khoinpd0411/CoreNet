
# CoreNet: Leveraging Context-Aware Representations via MLP Networks for CTR Prediction

This is repo of paper "CoreNet: Leveraging Context-Aware Representations via MLP Networks for CTR Prediction" for RecSys 2024 review process. 

## Datasets
Datasets are downloaded from:
- Criteo, Frappe, MovieLens 2M: https://github.com/WeiyuCheng/AFN-AAAI-20
- Avazu: https://www.kaggle.com/c/avazu-ctr-prediction

## Environment Requirements
The code has been tested running under Python 3.8. The required packaged for reproducing environment could be found in requirements.txt
```
pip install -r requirements.txt
```

## Preprocess
run
```
python src/preprocess/run_preprocess.py --dataset {name_dataset}
```

## Training
run
```
python src/train.py --dataset {name_dataset}
```