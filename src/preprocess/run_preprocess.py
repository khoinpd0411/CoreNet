from utils import read_frappe, read_ml, read_avazu, read_criteo

from sklearn.utils import shuffle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='frappe', help='dataset name')

def ordering(f):
    if f.startswith("train"):
        return (1, f)
    elif f.startswith("val"):
        return (2, f)
    else:
        return (3, f)

if __name__ == "__main__":
    opt = parser.parse_args()
    datasets = opt.dataset

    path = f'./src/data/{datasets}/'

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith((".libsvm", '.txt'))]
    files = sorted(files, key = ordering)
    print(files)

    files_path = [os.path.join(path, f) for f in files]

    if datasets == "frappe":
        df = read_frappe(files_path)
    elif datasets == "movielens":
        df = read_ml(files_path)
    elif datasets == "avazu":
        df = read_avazu(files_path)
    elif datasets == "criteo":
        df = read_criteo(files_path)
    else:
        raise "Datasets not found"
    
    if datasets in ["movielens", "frappe"]:
        df = shuffle(df, random_state=42)
    
    df.to_parquet(f"./src/data/{datasets}/train.parquet")