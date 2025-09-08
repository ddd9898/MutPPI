from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Bio import SeqIO
import random
from scipy.spatial.transform import Rotation as R
import torch
from torch import nn
# from utils.aaindex2 import aaindex2
from tqdm import tqdm
import itertools

def distance_matrix(current_wild_positions, current_tgt_protein_positions):
    dist_m = np.zeros(shape=(current_wild_positions.shape[0],current_tgt_protein_positions.shape[0]),dtype=np.float32)
    for i in range(current_tgt_protein_positions.shape[0]):
        dist_m[:,i] = np.linalg.norm(current_tgt_protein_positions[i] - current_wild_positions, axis=1)
        
    return dist_m


    

def reduction_item_byNumAA(wl_seq, mt_seq, wl_positions, tgt_proteins_seq, tgt_proteins_positions, num_aa = 20):
    '''
    返回以突变位点为起始点,长度为num_aa的突变位点周围序列(排列顺序尊崇最近邻原则),
    以及突变位点附近的、长度为num_aa的、靶点蛋白质的残基序列
    '''
    ##找到single突变位点
    mut_idx = [i for i in range(len(wl_seq)) if wl_seq[i] != mt_seq[i]]
    
    if len(mut_idx) != 1:
        print('Error: single mutation site not found')
        print(len(mut_idx))
        print(wl_seq)
        print(mt_seq)
        return None
    else:
        mut_idx = mut_idx[0]
    pos_mut = wl_positions[mut_idx]
    
    
    dis_wl = np.linalg.norm(wl_positions - pos_mut, axis=1)
    idx_wl_nearest = np.argsort(dis_wl)[:(num_aa+1)] ##+1是因为要包含突变位点本身
    new_wl_seq = ''.join(np.array(list(wl_seq))[idx_wl_nearest])
    new_wt_seq = ''.join(np.array(list(mt_seq))[idx_wl_nearest])
    new_wl_pos = wl_positions[idx_wl_nearest]
    
    tgt_pos_concat = np.vstack(tgt_proteins_positions)
    dis_tgt = np.linalg.norm(tgt_pos_concat-pos_mut, axis=1)
    idx_tgt_nearest = np.argsort(dis_tgt)[:num_aa]
    idx_tgt_nearest = idx_tgt_nearest[::-1] ##逆序排列，使离突变位点远的位点在序列最前面
    tgt_seq_concat = ''.join(tgt_proteins_seq)
    new_tgt_seq = ''.join(np.array(list(tgt_seq_concat))[idx_tgt_nearest])
    new_tgt_pos = tgt_pos_concat[idx_tgt_nearest]
    
    return new_wl_seq, new_wt_seq, new_wl_pos, new_tgt_seq, new_tgt_pos
    


def acquire_mt_seq(wild_seq_idx, mut, protein_ID):
    '''
    获取突变序列和野生序列, 如果mut==-1, 则获取野生型序列
    wild_seq_idx: eg. '59-P,60-G,61-E,62-L,63-V,64-R,65-T,66-D'
    mut: eg. 'A16D'
    protein_ID: '1E50_A', only used for highlighting errors.
    '''
    # wild_seq_idx = wild_seq_idx.split(',')
    flag_find_error = False

    mt_seq = list()
    wl_seq = list()     

    if mut == -1: #无突变，获取野生型序列
        for item in wild_seq_idx:
            wl_seq.append(item.split('-')[-1])
        wl_seq = ''.join(wl_seq)

        return wl_seq, wl_seq, flag_find_error
    else: #有突变，获取突变型序列
        mut = mut.upper()
        mut_pos = mut[1:-1]
        ori_aa = mut[0]
        tgt_aa = mut[-1]
        for item in wild_seq_idx:
            # print(item)
            # pos,aa = item.split('-')
            pos = item[:-2]
            aa = item[-1]
            if pos == mut_pos:
                if ori_aa == aa:
                    wl_seq.append(ori_aa)
                    mt_seq.append(tgt_aa)
                else:
                    print("Something is wrong for {}, mut={}, but ori aa = {}!".format(protein_ID, mut, item))
                    flag_find_error = True
                    return None, None, flag_find_error
            else:
                wl_seq.append(aa)
                mt_seq.append(aa)
        mt_seq = ''.join(mt_seq)
        wl_seq = ''.join(wl_seq)
        
        if mt_seq == wl_seq:
            print("mt_seq = wl_seq for {}_{}".format(protein_ID, mut))
            flag_find_error = True
            return None, None, flag_find_error
            
        return wl_seq, mt_seq, flag_find_error
    

def mutation_step(mutChain, wild_seq_idx, muts, protein_ID):
    '''
    获取突变步骤中最后一步的突变序列和野生序列
    mutChain: 当前突变链
    wild_seq_idx: 当前mutChain在pdb中的序列信息, eg. '59-P,60-G,61-E,62-L,63-V,64-R,65-T,66-D'
    muts: eg. (B:Y1172W, B:V1173A, A:I921V)
    protein_ID: '1E50_A', only used for highlighting errors.
    '''
    # wild_seq_idx = wild_seq_idx.split(',')
    flag_find_error = False

    mt_seq = list()
    wl_seq = list()
    
    mut_pos_list = list()  
    ori_aa_list = list()
    tgt_aa_list = list()
    for mut in muts:
        mut = mut.upper()
        if mut[0] != mutChain: # 只考虑当前突变链的突变
            continue
        mut_pos_list.append(mut[3:-1]) # 保存突变位置
        ori_aa_list.append(mut[2]) # 保存突变前氨基酸
        tgt_aa_list.append(mut[-1]) # 保存突变后氨基酸
        
    for item in wild_seq_idx:
        # print(item)
        # pos,aa = item.split('-')
        pos = item[:-2]
        aa = item[-1]
        if pos in mut_pos_list: #是突变位点
            idx = mut_pos_list.index(pos)
            if ori_aa_list[idx] == aa: #突变位点检查无误
                if mutChain == muts[-1][0] and pos == mut_pos_list[-1]: ##如果是最后一个突变步骤, 注意区分野生型和突变型序列
                    wl_seq.append(ori_aa_list[idx])
                else:
                    wl_seq.append(tgt_aa_list[idx])
                mt_seq.append(tgt_aa_list[idx])
            else:
                print("Something is wrong for {}, mut={}, but ori aa = {}!".format(protein_ID, muts[idx], item))
                flag_find_error = True
                return None, None, flag_find_error
        else:
            wl_seq.append(aa)
            mt_seq.append(aa)
            
    mt_seq = ''.join(mt_seq)
    wl_seq = ''.join(wl_seq)
        
    if mutChain == muts[-1][0] and mt_seq == wl_seq:
        print("mt_seq = wl_seq for {}_{}".format(protein_ID, mut))
        flag_find_error = True
        return None, None, flag_find_error
            
    return wl_seq, mt_seq, flag_find_error


import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids

def aa_to_onehot(seq):
    onehot = torch.zeros(len(seq), len(AA_LIST))
    for i, aa in enumerate(seq):
        if aa in AA_LIST:
            onehot[i][AA_LIST.index(aa)] = 1
    return onehot

    
    
# 添加ViSNet需要的.z信息：氨基酸字母到整数ID
aa_to_id = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}
def seq_to_id(seq):
    return torch.tensor([aa_to_id.get(res, 0) for res in seq], dtype=torch.long)  # 默认未知氨基酸为A(0)



def get_biounit_chains():
    ##biounit chains
    biounit_chains_dict = dict()
    tempdata = pd.read_csv('./data/SKEMPI2/S4169.csv').values.tolist()
    for item in tempdata:
        pdb_name = item[0]
        biounit_chains = list(item[1].replace('_',''))
        biounit_chains_dict[pdb_name] = biounit_chains
    biounit_chains_dict['2QJB'] = ['A', 'B', 'C', 'D']
    biounit_chains_dict['2QJA'] = ['A', 'B', 'C', 'D']
    biounit_chains_dict['2QJ9'] = ['A', 'B', 'C', 'D']
    biounit_chains_dict['1MQ8'] = ['A', 'B']
    biounit_chains_dict['1XXM'] = ['A', 'C']
    biounit_chains_dict['2PYE'] = ['D', 'E']
    biounit_chains_dict['3HG1'] = ['A', 'C', 'D', 'E']
    biounit_chains_dict['1QAB'] = ['A', 'B', 'C', 'D', 'E', 'F']
    biounit_chains_dict['4UYP'] = ['A', 'B', 'C', 'D']
    biounit_chains_dict['4UYQ'] = ['A', 'B']
    biounit_chains_dict['2VN5'] = ['A', 'B']
    biounit_chains_dict['3BDY'] = ['H', 'L', 'V']
    biounit_chains_dict['3BE1'] = ['H', 'L', 'A']
    biounit_chains_dict['2NY7'] = ['H', 'L', 'G']
    biounit_chains_dict['3IDX'] = ['H', 'L', 'G']
    biounit_chains_dict['2C5D'] = ['A', 'B', 'C', 'D']
    biounit_chains_dict['1OHZ'] = ['A', 'B']
    biounit_chains_dict['1Y4A'] = ['E', 'I']
    biounit_chains_dict['2NOJ'] = ['A', 'B']
    biounit_chains_dict['2ABZ'] = ['B', 'E']
    biounit_chains_dict['2CCL'] = ['A', 'B']
    biounit_chains_dict['3UIH'] = ['A', 'B']
    biounit_chains_dict['3SE4'] = ['A', 'B', 'C']
    biounit_chains_dict['4GU0'] = ['A', 'C', 'E']


    return biounit_chains_dict



class StructureGraphAugDataset(Dataset):
    def __init__(self, dataname = 'SM_ZEMu', UseMutPath = False, NumAA = 10):
        super(StructureGraphAugDataset,self).__init__()
        
        self.mutationPPI_data = list()
        self.UseMutPath = UseMutPath
        
        
        if dataname == 'SM_ZEMu' or dataname == 'SM595' or dataname == 'SM1124':
            ##Load structure data
            pos_idx_list = np.load('./data/test/multiple/{}-position.npy'.format(dataname), allow_pickle = True).item()
            data = pd.read_csv('./data/test/multiple/{}.csv'.format(dataname)).values.tolist()
            ##biounit chains
            biounit_chains_dict = get_biounit_chains()

        # ##Debug
        # data = data[:10]

        Error_count = 0
        for item in data: 
            if dataname == 'SM_ZEMu' or dataname == 'SM595' or dataname == 'SM1124':
                pdb_name = item[0][:4]
                biounit_chains = biounit_chains_dict[pdb_name]
                mutations = item[2].replace('_',',')
                ddg = item[3]
                
            
            ## 获取记录的所有突变步骤
            muts_step_all = list()
            for mutation in mutations.split(','):
                mutChain = mutation[1]
                src_aa = mutation[0]
                tgt_aa = mutation[-1]
                pos = mutation[2:-1]
                mutation = "{}:{}{}{}".format(mutChain, src_aa, pos, tgt_aa)
                
                muts_step_all.append(mutation)
            
            
            if self.UseMutPath: ##获取所有突变步骤的组合（突变路径）
                # muts_step_all = list(itertools.permutations(muts_step_all))
                # random.shuffle(muts_step_all)
                # muts_step_all = muts_step_all[:3] #防止路径组合数量过多

                muts_step_all = [muts_step_all, muts_step_all[::-1]] ## 添加反向突变路径
            else: ## 默认采用记录的突变路径
                muts_step_all = [muts_step_all]
            
            for muts_step in muts_step_all: # eg. (B:Y1172W, B:V1173A, A:I921V)
                mutation_data_step_list = list()
                # mutation_data_step_inverse_list = list()
                for mut_step_idx in range(len(muts_step)):
                    ###Mutated protein
                    mutChain = muts_step[mut_step_idx][0]
                    protein_ID = pdb_name + '_' + mutChain

                    wl_positions = pos_idx_list[protein_ID]
                    if UseMutPath:
                        wl_seq, mt_seq, flag_find_error = mutation_step(mutChain = mutChain, wild_seq_idx = list(wl_positions.keys()), muts = muts_step[:(mut_step_idx+1)], protein_ID = protein_ID)
                    else:
                        wl_seq, mt_seq, flag_find_error = acquire_mt_seq(wild_seq_idx = list(wl_positions.keys()), mut = muts_step[mut_step_idx].split(':')[-1], protein_ID = protein_ID)
                    wl_positions_np = np.array(list(wl_positions.values()))
                    if flag_find_error:
                        break

                    ###Target proteins
                    tgt_proteins_seq = list()
                    tgt_proteins_positions = list()
                    for biounit_chain in biounit_chains:
                        if biounit_chain == mutChain:
                            continue
                        target_protein_ID = pdb_name + '_' + biounit_chain
                            
                        target_structure = pos_idx_list[target_protein_ID]
                        
                        if UseMutPath:
                            target_sequence, _, flag_find_error = mutation_step(mutChain = biounit_chain, wild_seq_idx = list(target_structure.keys()), muts = muts_step[:(mut_step_idx+1)], protein_ID = target_protein_ID)
                        else:
                            target_sequence, _, _ = acquire_mt_seq(wild_seq_idx = list(target_structure.keys()), mut = -1, protein_ID = target_protein_ID)
                        if flag_find_error:
                            break
                        target_structure_np = np.array(list(target_structure.values()))
                        
                        
                        # target_proteins.append({'sequence':target_sequence, 'structure':target_structure_np})
                        tgt_proteins_seq.append(target_sequence)
                        tgt_proteins_positions.append(target_structure_np)
                    if flag_find_error:
                        break
                            
                    ori_wl_seq, ori_mt_seq = wl_seq, mt_seq
                    # if NumAA > 0:
                    #     wl_seq, mt_seq, wl_positions_np, tgt_proteins_seq, tgt_proteins_positions = reduction_item_byNumAA(wl_seq, mt_seq, wl_positions_np, tgt_proteins_seq, tgt_proteins_positions, NumAA)
                    tgt_proteins_seq = ''.join(tgt_proteins_seq)
                    tgt_proteins_positions = np.vstack(tgt_proteins_positions)

                    ###mutant sequence
                    # 拼接所有序列和坐标
                    all_seq = mt_seq + tgt_proteins_seq
                    all_positions = np.concatenate((wl_positions_np, tgt_proteins_positions), axis=0)
                    # 构造节点特征
                    node_features = aa_to_onehot(all_seq)
                    # node_features = aa_to_onehot_add_mutMarker(mt_seq, tgt_proteins_seq)
                    # z_ids = seq_to_id(''.join(all_seq)) 
                    # 计算 pairwise 距离
                    dist_matrix = cdist(all_positions, all_positions)
                    adjacency = (dist_matrix < 7).astype(np.float32)  # 距离小于 7Å 认为有边
                    # edge_index 和边特征（距离）
                    edge_index, _ = dense_to_sparse(torch.tensor(adjacency))
                    edge_attr = torch.norm(
                        torch.tensor(all_positions)[edge_index[0]] - torch.tensor(all_positions)[edge_index[1]],
                        dim=1
                    ).unsqueeze(1)  # shape [num_edges, 1]
                    # 构造 PyG 的图
                    data_mt = Data(
                        x=node_features,              # 节点特征：one-hot
                        # z=z_ids,                      # 加入原子ID, <非基本信息>
                        # pos=torch.tensor(all_positions, dtype=torch.float),  # 加入坐标, <非基本信息>
                        mutInfo = pdb_name + '_' + muts_step[mut_step_idx],
                        seq=ori_mt_seq,
                        edge_index=edge_index,       # 边连接
                        edge_attr=edge_attr,         # 边特征：距离
                        y=torch.tensor([ddg], dtype=torch.float)  # 回归目标
                    )
                    
                    ###wild sequence
                    # 拼接所有序列和坐标
                    all_seq = wl_seq + tgt_proteins_seq
                    # 构造节点特征
                    node_features = aa_to_onehot(all_seq)
                    # node_features = aa_to_onehot_add_mutMarker(wl_seq, tgt_proteins_seq)
                    # z_ids = seq_to_id(''.join(all_seq)) 
                    # 计算 pairwise 距离
                    dist_matrix = cdist(all_positions, all_positions)
                    adjacency = (dist_matrix < 7).astype(np.float32)  # 距离小于 7Å 认为有边
                    # edge_index 和边特征（距离）
                    edge_index, _ = dense_to_sparse(torch.tensor(adjacency))
                    edge_attr = torch.norm(
                        torch.tensor(all_positions)[edge_index[0]] - torch.tensor(all_positions)[edge_index[1]],
                        dim=1
                    ).unsqueeze(1)  # shape [num_edges, 1]
                    # 构造 PyG 的图
                    data_wl = Data(
                        x=node_features,              # 节点特征：one-hot
                        # z=z_ids,                      # 加入原子ID, <非基本信息>
                        # pos=torch.tensor(all_positions, dtype=torch.float),  # 加入坐标, <非基本信息>
                        mutInfo = pdb_name + '_' + muts_step[mut_step_idx],
                        seq = ori_wl_seq,
                        edge_index=edge_index,       # 边连接
                        edge_attr=edge_attr,         # 边特征：距离
                        y=torch.tensor([ddg], dtype=torch.float)  # 回归目标
                    )
                    mutation_data_step_list.extend([data_wl, data_mt])
                    
                    # # 数据增强（如果需要）
                    # if self.UseMutPath:
                    #     data_wl_aug = data_wl.clone()
                    #     data_wl_aug.y *= -1
                    #     data_mt_aug = data_mt.clone()
                    #     data_mt_aug.y *= -1
                    #     mutation_data_step_inverse_list.extend([data_mt_aug, data_wl_aug])
                if flag_find_error:
                    Error_count = Error_count + 1
                    continue
                # 存入结果
                self.mutationPPI_data.append(mutation_data_step_list)
                
                # if self.UseMutPath:
                #     self.mutationPPI_data.append(mutation_data_step_inverse_list)
                

                    
        print("Delete {} samples due to not aligned between sequence and structure data!".format(Error_count))

    def __len__(self):
        return len(self.mutationPPI_data)

    def __getitem__(self, i):
        
        return self.mutationPPI_data[i]  # 直接返回 PyG 图结构, (data_wl, data_mt)

from torch_geometric.data import Batch
def my_graph_aug_collate_fn(batch):
    N = len(batch[0])
    # N = np.max([ len(data) for data in batch ] )
    output = [ list() for n in range(N)]
    for item in batch:
        for i in range(N):
            output[i].append(item[i])
    
    for i in range(N):
        output[i] = Batch.from_data_list(output[i])
    return output


