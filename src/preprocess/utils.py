import os
import pandas as pd

from tqdm import tqdm
from optbinning import ContinuousOptimalBinning 

def read_frappe(files):
    df = pd.DataFrame()
    data = []
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().replace("\n", "")
                feats = line.split(" ")

                label = 1 if int(feats[0]) > 0 else 0
                user = int(feats[1].split(":")[0])
                item = int(feats[2].split(":")[0])
                daytime = int(feats[3].split(":")[0])
                weekday = int(feats[4].split(":")[0])
                isweekend = int(feats[5].split(":")[0])
                homework = int(feats[6].split(":")[0])
                cost = int(feats[7].split(":")[0])
                weather = int(feats[8].split(":")[0])
                country = int(feats[9].split(":")[0])
                city = int(feats[10].split(":")[0])
            
                data.append([user, item, daytime, weekday, isweekend, homework, cost, weather, country, city, label])

    df = df.from_records(data, columns = ["user", "item", "daytime", "weekday", "isweekend", "homework", "cost", "weather", "country", "city",
                                          "label"])

    del data
    return df

def read_ml(files):
    df = pd.DataFrame()
    data = []
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().replace("\n", "")
                feats = line.split(" ")

                label = 1 if int(feats[0]) > 0 else 0
                user = int(feats[1].split(":")[0])
                item = int(feats[2].split(":")[0])
                tag = int(feats[3].split(":")[0])
            
                data.append([user, item, tag, label])

    df = df.from_records(data, columns = ["user", "item", "tag",
                                          "label"])

    del data
    return df

def read_avazu(files):
    df = pd.DataFrame([])

    parse_field = ["hour"]
    parse_func = lambda val: pd.datetime.strptime(val, "%y%m%d%H")

    for file in files:
        ratings = pd.read_csv(file, sep = ",", encoding="latin-1", parse_dates=parse_field, date_parser=parse_func)
        df = pd.concat([df, ratings])
    
    df["month"] = df["hour"].dt.microsecond
    df["dayofweek"] = df["hour"].dt.dayofweek
    df["day"] = df["hour"].dt.day
    df["hour_time"] = df["hour"].dt.hour

    cols = ["device_type", "device_conn_type", "banner_pos",
            "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C1"]
    for col in cols:
        df[col] = df[col].astype(str)

    return df

def read_criteo(files) -> pd.DataFrame():
    df_data = []
    for file in files:
        with open(f'data/criteo/{file}') as f:
            data = f.readlines()
            for i,d in enumerate(data):
                d_array = d.strip().split(' ')
                clean_d = []
                for item in d_array:
                    split_item = item.split(":")
                    if len(split_item) > 1:
                        if split_item[1] != '1':
                            clean_d.append(split_item[1])
                        else:
                            clean_d.append(split_item[0])
                    else:
                        clean_d.append(split_item[0])
                df_data.append(clean_d)
                            
    header = ['label'] + [f'feature_{i}' for i in range(1,40)]
    df = pd.DataFrame(df_data, columns=header)
    print("Total Record: ", len(df))
    
    return df

class Bins:
    def __init__(self, exist_dict: dict = None):
        if isinstance(exist_dict, dict):
            self.dict = exist_dict
        else:
            self.dict = {}

    def add(self, conts, target, df):
        for c in conts:
            if c not in self.dict:
                x = df[c].values
                y = df[target]
                self.dict[c] = ContinuousOptimalBinning(
                    name=c, dtype="numerical")
                self.dict[c].fit(x, y)
                binning_table = self.dict[c].binning_table
                binning_table.build()

    def apply(self, df):
        for c in self.dict:
            df[c] = self.dict[c].transform(df[c].values, metric="bins")
        return df