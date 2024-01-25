import os

import torch
from torch.optim import Adam
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from preprocess.utils import Bins

from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

from corenet.models import CoreNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='frappe', help='dataset name')
parser.add_argument('--save_dir', type=str, default='./src/weights/corenet.pt', help='dataset name')

if __name__ == "__main__":

    opt = parser.parse_args()
    dataset = opt.dataset

    path = f"./src/data/{dataset}/"
    path_ft = opt.save_dir

    if os.path.exists(path + "train.parquet"):
        train_full = pd.read_parquet(path + "train.parquet")
    else:
        raise Exception(f"Cannot find data path or dataset {dataset} is not supported.")
    
    if dataset == "movielens":
        target = "label"
        sparse_features = ["user", "item", "tag"]

    elif dataset == "frappe":
        target = "label"
        sparse_features = ["user", "item", "daytime", "weekday", "isweekend", 
                           "homework", "cost", "weather", "country", "city"]
        
    elif dataset == "avazu":
        target = "click"
        sparse_features = ["C1", "banner_pos", "site_id", "site_domain", "site_category",
                           "app_id", "app_domain", "app_category",
                           "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
                           "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
                           "month", "dayofweek", "day", "hour_time"]
    
    elif dataset == "criteo":
        target = "label"
        dense_features = ["feature_14", "feature_15", "feature_16", "feature_17", "feature_18",
                         "feature_19", "feature_20", "feature_21", "feature_22", "feature_23",
                         "feature_24", "feature_25", "feature_26", "feature_27", "feature_28",
                         "feature_29", "feature_30", "feature_31", "feature_32", "feature_33",
                         "feature_34", "feature_35", "feature_36", "feature_37", "feature_38",
                         "feature_39"]
        
        sparse_features = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5",
                           "feature_6", "feature_7", "feature_8"]

    
    if dataset in ["frappe", "movielens"]:
        # Label Encoding for sparse features, and do transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            train_full[feat] = lbe.fit_transform(train_full[feat])
        fixlen_feature_columns = [SparseFeat(feat, train_full[feat].nunique(), embedding_dim=10)
                                for feat in sparse_features]
    else:
        woe_bins = Bins()
        woe_bins.add(dense_features, target, train_full)
        train_full = woe_bins.apply(train_full)

        sparse_features = sparse_features + dense_features
        for feat in sparse_features:
            lbe = LabelEncoder()
            train_full[feat] = lbe.fit_transform(train_full[feat])
        fixlen_feature_columns = [SparseFeat(feat, train_full[feat].nunique(), embedding_dim=10)
                                for feat in sparse_features]
        
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    kf = KFold(n_splits = 5)

    for i, (train_index, test_index) in enumerate(kf.split(train_full)):
        train = train_full.iloc[train_index]
        test = train_full.iloc[test_index]
        train = pd.concat([train, test])

        # 4.Define Model,train,predict and evaluate
        train_model_input = {name: train[name] for name in feature_names}

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # device = "cpu"

        es = EarlyStopping(monitor='val_binary_crossentropy', patience=3, verbose=1)
        mdckpt = ModelCheckpoint(
                filepath=path_ft, monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')

        if dataset == "frappe":
            model = CoreNet(linear_feature_columns, dnn_feature_columns, task='binary', device=device, dnn_hidden_units=(300,))
            model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'auc'])
        elif dataset == "movielens":
            model = CoreNet(linear_feature_columns, dnn_feature_columns, task='binary', device=device, dnn_hidden_units=(300,))
            model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'auc'])
        elif dataset == "avazu":
            model = CoreNet(linear_feature_columns, dnn_feature_columns, task='binary', device=device, dnn_hidden_units=(400, 400, 400))
            model.compile(Adam(model.parameters(), 1e-2), "binary_crossentropy", metrics=['binary_crossentropy', 'auc'])
        elif dataset == "criteo":
            model = CoreNet(linear_feature_columns, dnn_feature_columns, task='binary', device=device, dnn_hidden_units=(400, 400, 400))
            model.compile(Adam(model.parameters(), 7.5e-3), "binary_crossentropy", metrics=['binary_crossentropy', 'auc'])
        
        history = model.fit(train_model_input,train[target].values,batch_size=1024,epochs=100,verbose=1,validation_split=0.2,
                            callbacks=[es, mdckpt])