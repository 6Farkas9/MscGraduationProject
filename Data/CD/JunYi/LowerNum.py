import pandas as pd
from tqdm import tqdm

reader_train_info = pd.read_csv('./train.csv', header=0, on_bad_lines='skip',nrows=800000)

reader_test_info = pd.read_csv('./test.csv', header=0, on_bad_lines='skip',nrows=200000)

reader_train_info.to_csv('./train.csv',index=False)

reader_test_info.to_csv('./test.csv',index=False)

