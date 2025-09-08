import pandas as pd
import numpy as np


data = pd.read_csv('./data/SKEMPI2/SKEMPI2.csv').values.tolist()
data_base = pd.read_csv('./data/SKEMPI2/S4169.csv').values.tolist()

base_dict = dict()
for item in data_base:
    pdb_name = item[0]
    biounit_chains = list(item[1].replace('_',''))
    mutChain_mutation = item[4]
    ddg = float(-item[5])
    
    key = '{}_{}'.format(pdb_name, mutChain_mutation)
    base_dict[key] = ddg


new_data = list()
for item in data:
    pdb_name = item[1]
    biounit_chains = list(item[2]+item[3])
    mutation_list = item[4].split(',')
    ddg = float(item[5])

    if len(mutation_list) == 1 or np.isnan(ddg):
        continue

    skip_flag = False
    ddg_path = list()
    for mut in mutation_list:
        mutChain = mut[1]
        src_aa = mut[0]
        tgt_aa = mut[-1]
        pos = mut[2:-1]
        key = '{}_{}:{}{}{}'.format(pdb_name, mutChain, src_aa, pos, tgt_aa)
        
        if key not in base_dict:
            skip_flag = True
            break

        ddg_path.append('{}{}{}{}={}'.format(src_aa, mutChain, pos, tgt_aa, base_dict[key]))
    
    if skip_flag:
        continue
    
    ddg_path = ','.join(ddg_path)

    new_item = [pdb_name, ''.join(biounit_chains), ','.join(mutation_list), ddg, ddg_path]
    new_data.append(new_item)


df = pd.DataFrame(new_data)
df.columns = ['pdb_name', 'biounit_chains', 'mutation_list', 'ddg', 'ddg_path']
df.to_csv('./data/SKEMPI2/SKEMPI2_MultiMuts.csv', index=False)
        
        
        



