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



def distance_matrix(current_wild_positions, current_tgt_protein_positions):
    dist_m = np.zeros(shape=(current_wild_positions.shape[0],current_tgt_protein_positions.shape[0]),dtype=np.float32)
    for i in range(current_tgt_protein_positions.shape[0]):
        dist_m[:,i] = np.linalg.norm(current_tgt_protein_positions[i] - current_wild_positions, axis=1)
        
    return dist_m

def reduction_item_byInterfaceDist(wl_seq, mt_seq, current_wild_positions, current_tgt_proteins_seq, current_tgt_proteins_positions, interface_dist=7):
    '''
    删减靶点蛋白的非结合口袋部分(>7A)
    current_wild_positions:  (Num_AA * 3)
    current_tgt_proteins_seq: Num_seq
    current_tgt_proteins_positions: Num_proteins * (Num_AA * 3)
    '''
    dist_m_2 = 0
    new_seqs_list = list()
    new_posi_list = list()
    for tgt_protein_idx in range(len(current_tgt_proteins_seq)):
        current_tgt_protein_positions = current_tgt_proteins_positions[tgt_protein_idx]
        current_tgt_protein_seq = current_tgt_proteins_seq[tgt_protein_idx]
        
        dist_m = distance_matrix(current_wild_positions, current_tgt_protein_positions)
        dist_m_1 = np.sum(dist_m < interface_dist, axis=0, keepdims=False)>0 # Num_AA
        dist_m_2 = dist_m_2 + np.sum(dist_m < interface_dist, axis=1, keepdims=False)>0 # Num_AA
        
        new_seqs = ''.join(np.array(list(current_tgt_protein_seq))[dist_m_1])
        new_posi = current_tgt_protein_positions[dist_m_1]

        if len(new_seqs) > 0 :
            new_seqs_list.append(new_seqs)
            new_posi_list.append(new_posi)
    
    ##New mutant/wild protein information
    dist_m_2 = (dist_m_2>0)
    new_wild_positions = current_wild_positions[dist_m_2]
    new_wl_seq = ''.join(np.array(list(wl_seq))[dist_m_2])
    new_mt_seq = ''.join(np.array(list(mt_seq))[dist_m_2])
    
    new_tgt_seq = ''.join(new_seqs_list)
    new_tgt_pos = np.vstack(new_posi_list)
    
    return new_wl_seq, new_mt_seq,new_wild_positions, new_tgt_seq, new_tgt_pos


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
    
# # 添加ViSNet需要的.z信息：氨基酸字母到整数ID
# aa_to_id = {
#     'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
#     'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
#     'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
#     'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
# }
# def seq_to_id(seq):
#     return torch.tensor([aa_to_id.get(res, 0) for res in seq], dtype=torch.long)  # 默认未知氨基酸为A(0)


class StructureGraphDataset(Dataset):
    def __init__(self, datapath = 'S645', train_flag = True, fold = 0, reduction = 10, reduction_mode = 0):
        '''
        reduction_mode: 0: no reduction, 1: reduction_item_byInterfaceDist, 2: reduction_item_byNumAA,
        '''
        super(StructureGraphDataset,self).__init__()
        
        # self.ESM2_embedding = ESM2feature(pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320, device = 'cuda')
        
        self.mutationPPI_data = list()
        # self.augment = augment
        

        ##Load structure data
        pos_idx_list = np.load('./data/SKEMPI2/S645_aa-position.npy', allow_pickle = True).item()
        data = pd.read_csv(datapath).values.tolist()
        # data = data[160:160+10] ## debug
    
            
        Error_count = 0
        for item in data: 
            pdb_name = item[0].upper()
            biounit_chains = list(item[3]+item[4])
            mutChain_mutation = item[1]
            ddg = float(item[2])
            fold_flag = int(item[-2])
            direction = item[-1]
            
            if train_flag and (fold_flag == fold):
                continue
            
            if (not train_flag) and (fold_flag != fold):
                continue
        
                
            ###Mutated protein
            mutChain, mutation = mutChain_mutation.split(':')
            protein_ID = pdb_name + '_' + mutChain
                
            wl_positions = pos_idx_list[protein_ID]
            wl_seq, mt_seq, flag_find_error = acquire_mt_seq(wild_seq_idx = list(wl_positions.keys()), mut = mutation, protein_ID = protein_ID)
            wl_positions_np = np.array(list(wl_positions.values()))
            if flag_find_error:
                Error_count = Error_count + 1
                continue

            ###Target proteins
            tgt_proteins_seq = list()
            tgt_proteins_positions = list()
            for biounit_chain in biounit_chains:
                if biounit_chain == mutChain:
                    continue
                target_protein_ID = pdb_name + '_' + biounit_chain
                    
                target_structure = pos_idx_list[target_protein_ID]
                target_sequence, _, _ = acquire_mt_seq(wild_seq_idx = list(target_structure.keys()), mut = -1, protein_ID = target_protein_ID)
                target_structure_np = np.array(list(target_structure.values()))
                
                
                # target_proteins.append({'sequence':target_sequence, 'structure':target_structure_np})
                tgt_proteins_seq.append(target_sequence)
                tgt_proteins_positions.append(target_structure_np)

            ori_wl_seq, ori_mt_seq = wl_seq, mt_seq
            
            
            if reduction != 0:
                ## 约简氨基酸残基
                if reduction_mode == 0:
                    tgt_proteins_seq = ''.join(tgt_proteins_seq)
                    tgt_proteins_positions = np.vstack(tgt_proteins_positions)
                elif reduction_mode == 1:
                    wl_seq, mt_seq, wl_positions_np, tgt_proteins_seq, tgt_proteins_positions = reduction_item_byInterfaceDist(wl_seq, mt_seq, wl_positions_np, tgt_proteins_seq, tgt_proteins_positions, interface_dist = reduction)
                elif reduction_mode == 2:
                    wl_seq, mt_seq, wl_positions_np, tgt_proteins_seq, tgt_proteins_positions = reduction_item_byNumAA(wl_seq, mt_seq, wl_positions_np, tgt_proteins_seq, tgt_proteins_positions, num_aa = reduction)
             
                ## 添加数据
                if direction == 'forward':
                    self.append_graph_item(wl_seq, mt_seq, wl_positions_np, tgt_proteins_seq, tgt_proteins_positions, ori_wl_seq, ori_mt_seq, ddg)
                else:
                    self.append_graph_item(mt_seq, wl_seq, wl_positions_np, tgt_proteins_seq, tgt_proteins_positions, ori_mt_seq, ori_wl_seq, ddg)


        print("Delete {} samples due to not aligned between sequence and structure data!".format(Error_count))

    def append_graph_item(self, new_wl_seq, new_mt_seq, new_wl_positions_np, new_tgt_proteins_seq, new_tgt_proteins_positions, ori_wl_seq, ori_mt_seq, ddg):
        # 拼接所有坐标
        all_positions = np.concatenate((new_wl_positions_np, new_tgt_proteins_positions), axis=0)
        data_mt = self.acquire_graph_data(new_mt_seq + new_tgt_proteins_seq, all_positions, ddg) ##mutant sequence
        data_wl = self.acquire_graph_data(new_wl_seq + new_tgt_proteins_seq, all_positions, ddg) ###wild sequence
        self.mutationPPI_data.append((data_wl, data_mt, ori_wl_seq, ori_mt_seq)) # 存入结果
        # if self.augment: # 数据增强（如果需要）
        #     data_wl_aug = data_wl.clone()
        #     data_wl_aug.y *= -1
        #     data_mt_aug = data_mt.clone()
        #     data_mt_aug.y *= -1
        #     self.mutationPPI_data.append((data_mt_aug, data_wl_aug, ori_mt_seq, ori_wl_seq))

    def acquire_graph_data(self, all_seq, all_positions, ddg):
        # 构造节点特征
        node_features = aa_to_onehot(all_seq)
        # 
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
        data_graph = Data(
            x=node_features,              # 节点特征：one-hot
            # z=z_ids,                      # 加入原子ID, <非基本信息>
            # pos=torch.tensor(all_positions, dtype=torch.float),  # 加入坐标, <非基本信息>
            edge_index=edge_index,       # 边连接
            edge_attr=edge_attr,         # 边特征：距离
            y=torch.tensor([ddg], dtype=torch.float)  # 回归目标
        )
        
        return data_graph


    def __len__(self):
        return len(self.mutationPPI_data)

    def __getitem__(self, i):
        
        return self.mutationPPI_data[i]  # 直接返回 PyG 图结构, (data_wl, data_mt)


from torch_geometric.data import Batch
def my_graph_collate_fn(batch):
    # batch: List[(data_wl, data_mt, ori_wl_seq, ori_mt_seq)]
    data_wl_list, data_mt_list, ori_wl_seq_list, ori_mt_seq_list = zip(*batch)
    batch_wl = Batch.from_data_list(data_wl_list)
    batch_mt = Batch.from_data_list(data_mt_list)
    return batch_wl, batch_mt, ori_wl_seq_list, ori_mt_seq_list



