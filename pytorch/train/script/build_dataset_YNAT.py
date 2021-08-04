import json
import pandas as pd

data_dir = 'data/ynat-v1'
data_files = {'train':f'{data_dir}/ynat-v1_train.json', 
              'validation':f'{data_dir}/ynat-v1_dev.json'}
features = {'guid': 'Id', 'title':'text', 'label':'label'}


for task, fn in data_files.items():
    with open(fn) as f:
        data = json.load(f)
    data = pd.DataFrame(data).loc[:,features.keys()].rename(columns=features)
    data.to_csv(f"{data_dir}/{task}.csv", index=False)