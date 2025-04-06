import pandas as pd
from tqdm import tqdm

reader_train_info = pd.read_csv('../train.csv', header=0, on_bad_lines='skip',nrows=10000)

reader_master_info = pd.read_csv('../master.csv', header=0, on_bad_lines='skip',nrows=2000)

reader_train_info.to_csv('../train_low.csv',index=False)

reader_master_info.to_csv('../master_low.csv',index=False)

