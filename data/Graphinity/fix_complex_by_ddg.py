import pandas as pd
import numpy as np

# data_name = 'Experimental_ddG_645_+Reverse_Mutations_+Non_Binders-cutoff_random-10foldcv'
# data_name = 'Experimental_ddG_645_+Reverse_Mutations_+Non_Binders-cutoff_100-10foldcv'
# data_name = 'Experimental_ddG_645_+Reverse_Mutations_+Non_Binders-cutoff_90-10foldcv'
data_name = 'Experimental_ddG_645_+Reverse_Mutations_+Non_Binders-cutoff_70-10foldcv'

data = pd.read_csv('./data/Graphinity/Experimental_ddG_645_+Reverse_Mutations_+Non_Binders/{}.csv'.format(data_name)).values.tolist()
base_data = pd.read_csv('./data/Graphinity/AB_645.csv').values.tolist()


base_dict = dict()
base_dict_reverse = dict()
for item in base_data:
    pdb_name = item[0]
    biounit_chains = list(item[1].replace('_',''))
    mutChain_mutation = item[4]
    ddg = -float(item[5])

    mutChain, mutation = mutChain_mutation.split(':')
    src_aa = mutation[0]
    pos = mutation[1:-1]
    tgt_aa = mutation[-1]

    key = '{}:{}-{}_{}_{}'.format(pdb_name, mutChain, src_aa, tgt_aa, ddg)
    base_dict[key] = pos

    key = '{}:{}-{}_{}_{}'.format(pdb_name, mutChain, tgt_aa, src_aa, -ddg)
    base_dict_reverse[key] = pos



data_fix = list()
for item in data:
    pdb_name = item[0].upper()
    biounit_chains = list(item[3]+item[4])
    mutChain_mutation = item[1].split('_')[-1]
    ddg = float(item[2])
    mut_chain = mutChain_mutation[1]
    src_aa = mutChain_mutation[0]
    tgt_aa = mutChain_mutation[-1]
    pos = mutChain_mutation[2:-1]  # 这是原始残基编号

    key = '{}:{}-{}_{}_{}'.format(pdb_name, mut_chain, src_aa, tgt_aa, ddg)
    if key in base_dict.keys():
        pos = base_dict[key]
        mutChain_mutation = '{}:{}{}{}'.format(mut_chain, src_aa, pos, tgt_aa)
        item[1] = mutChain_mutation
        data_fix.append(item + ['forward'])
    elif key in base_dict_reverse.keys():
        pos = base_dict_reverse[key]
        mutChain_mutation = '{}:{}{}{}'.format(mut_chain, tgt_aa, pos, src_aa)
        item[1] = mutChain_mutation
        data_fix.append(item + ['backward'])
    else:
        print('Something is wrong for {}'.format(item))

df = pd.DataFrame(data_fix)
df.columns = ['pdb','complex','labels','chain_prot1','chain_prot2','ab_chain','ag_chain','fold_id', 'direction']
df.to_csv('./data/Graphinity/Experimental_ddG_645_+Reverse_Mutations_+Non_Binders/{}_fixed.csv'.format(data_name), index=False)





