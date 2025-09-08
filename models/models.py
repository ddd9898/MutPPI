import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, GATConv, global_add_pool, BatchNorm, global_mean_pool, global_max_pool
import torch.nn as nn
import numpy as np


class baseline(nn.Module):
    def __init__(self, in_channels=20, hidden_channels=128, dim=512, heads=4):
        super().__init__()
        self.relu = nn.ReLU()

        # --- GIN 层（提取结构特征） ---
        self.gin1_wl = GINConv(nn.Sequential(nn.Linear(in_channels, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn1_wl = BatchNorm(dim)
        self.gin2_wl = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn2_wl = BatchNorm(dim)
        
        self.gin1_mt = GINConv(nn.Sequential(nn.Linear(in_channels, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn1_mt = BatchNorm(dim)
        self.gin2_mt = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn2_mt = BatchNorm(dim)

        # --- GAT 层（加权邻居聚合） ---
        self.gat_wl = GATConv(dim, dim // heads, heads=heads, dropout=0.1)
        self.gat_mt = GATConv(dim, dim // heads, heads=heads, dropout=0.1)


        # --- Readout ---
        self.fc_wl = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, hidden_channels))
        self.fc_mt = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, hidden_channels))

        # --- ddG 预测器 ---
        self.regressor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
          

    def forward(self, wl, mt):
        # --- wild-type ---
        x_wl = self.relu(self.bn1_wl(self.gin1_wl(wl.x, wl.edge_index)))
        x_wl = self.relu(self.bn2_wl(self.gin2_wl(x_wl, wl.edge_index)))
        x_wl = self.gat_wl(x_wl, wl.edge_index)
        x_wl = global_add_pool(x_wl, wl.batch)
        x_wl = self.fc_wl(x_wl)

        # --- mutant ---
        x_mt = self.relu(self.bn1_mt(self.gin1_mt(mt.x, mt.edge_index)))
        x_mt = self.relu(self.bn2_mt(self.gin2_mt(x_mt, mt.edge_index)))
        x_mt = self.gat_mt(x_mt, mt.edge_index)
        x_mt = global_add_pool(x_mt, mt.batch)
        x_mt = self.fc_mt(x_mt)

        # --- ddG prediction ---
        x = torch.cat([x_wl, x_mt], dim=1)
        ddg = self.regressor(x).squeeze(1)
        return ddg




class GINGATRegressor(nn.Module):
    def __init__(self, in_channels=20, hidden_channels=128, dim=512, heads=4):
        super().__init__()
        self.relu = nn.ReLU()

        # --- GIN 层（提取结构特征） ---
        self.gin1_wl = GINConv(nn.Sequential(nn.Linear(in_channels, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn1_wl = BatchNorm(dim)

        self.gin2_wl = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn2_wl = BatchNorm(dim)

        # --- GAT 层（加权邻居聚合） ---
        self.gat_wl = GATConv(dim, dim // heads, heads=heads, dropout=0.1)

        # --- Readout ---
        self.fc_wl = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, hidden_channels))

        # --- ddG 预测器 ---
        self.regressor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
          

    def forward(self, wl, mt):
        # --- wild-type ---
        x_wl = self.relu(self.bn1_wl(self.gin1_wl(wl.x, wl.edge_index)))
        x_wl = self.relu(self.bn2_wl(self.gin2_wl(x_wl, wl.edge_index)))
        x_wl = self.gat_wl(x_wl, wl.edge_index)
        x_wl = global_add_pool(x_wl, wl.batch)
        x_wl = self.fc_wl(x_wl)

        # --- mutant ---
        x_mt = self.relu(self.bn1_wl(self.gin1_wl(mt.x, mt.edge_index)))
        x_mt = self.relu(self.bn2_wl(self.gin2_wl(x_mt, mt.edge_index)))
        x_mt = self.gat_wl(x_mt, mt.edge_index)
        x_mt = global_add_pool(x_mt, mt.batch)
        x_mt = self.fc_wl(x_mt)

        # --- ddG prediction ---
        x = torch.cat([x_wl, x_mt], dim=1)
        ddg = self.regressor(x).squeeze(1)
        return ddg



from transformers import AutoTokenizer, AutoModel
class FusionEnsembleRegressor(nn.Module):
    def __init__(self, in_channels=20, hidden_channels=128, dim=512, heads=4,
                 pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320,finetune = 0):
        super().__init__()
        self.relu = nn.ReLU()
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.pretrained_model  = AutoModel.from_pretrained(pretrained_model)

        ##设置ESM2的参数是否冻结（用于序列的编码）
        if finetune == 0: # 冻结pretrained_model参数
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        elif finetune == 1: # 微调pretrained_model最后一层参数
            for name,param in self.pretrained_model.named_parameters():
                if 'esm2_t6_8M_UR50D' in pretrained_model and 'encoder.layer.5.' not in name:
                    param.requires_grad = False
                if 'esm2_t12_35M_UR50D' in pretrained_model and 'encoder.layer.11.' not in name:
                    param.requires_grad = False
                if 'esm2_t30_150M_UR50D' in pretrained_model and 'encoder.layer.29.' not in name:
                    param.requires_grad = False
                if 'esm2_t33_650M_UR50D' in pretrained_model and 'encoder.layer.32.' not in name:
                    param.requires_grad = False
                if 'esm2_t36_3B_UR50D' in pretrained_model and 'encoder.layer.35.' not in name:
                    param.requires_grad = False
                if 'esm2_t48_15B_UR50D' in pretrained_model and 'encoder.layer.47.' not in name:
                    param.requires_grad = False
        elif finetune == 2: # 微调pretrained_model全部参数
            for param in self.pretrained_model.parameters():
                param.requires_grad = True

        # --- GIN 层（提取结构特征） ---
        self.gin1_wl = GINConv(nn.Sequential(nn.Linear(in_channels, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn1_wl = BatchNorm(dim)

        self.gin2_wl = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn2_wl = BatchNorm(dim)

        # --- GAT 层（加权邻居聚合） ---
        self.gat_wl = GATConv(dim, dim // heads, heads=heads, dropout=0.1)

        # --- Readout ---
        self.fc_wl = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, hidden_channels))

        # --- ddG 预测器 ---
        self.regressor_ave = nn.Sequential(
            nn.Linear(2 * hidden_channels + hidden_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
        self.regressor_min = nn.Sequential(
            nn.Linear(2 * hidden_channels + hidden_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
        self.regressor_max = nn.Sequential(
            nn.Linear(2 * hidden_channels + hidden_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, wl, mt, wl_seq, mt_seq):
        # --- wild-type ---
        x_wl = self.relu(self.bn1_wl(self.gin1_wl(wl.x, wl.edge_index)))
        x_wl = self.relu(self.bn2_wl(self.gin2_wl(x_wl, wl.edge_index)))
        x_wl = self.gat_wl(x_wl, wl.edge_index)
        x_wl = global_add_pool(x_wl, wl.batch)
        x_wl = self.fc_wl(x_wl)

        # --- mutant ---
        x_mt = self.relu(self.bn1_wl(self.gin1_wl(mt.x, mt.edge_index)))
        x_mt = self.relu(self.bn2_wl(self.gin2_wl(x_mt, mt.edge_index)))
        x_mt = self.gat_wl(x_mt, mt.edge_index)
        x_mt = global_add_pool(x_mt, mt.batch)
        x_mt = self.fc_wl(x_mt)
        
        
        ##Sequence Embedding
        max_sequence_length = np.max([len(seq) for seq in wl_seq])
        tokenizer1 = self.tokenizer(wl_seq,
                                    truncation=True,
                                    padding=True,
                                    max_length=max_sequence_length,
                                    add_special_tokens=False)
        input1_ids=torch.tensor(tokenizer1['input_ids']).to(self.pretrained_model.device)
        attention_mask1=torch.tensor(tokenizer1['attention_mask']).to(self.pretrained_model.device)
        wl_output1=self.pretrained_model(input_ids=input1_ids,attention_mask=attention_mask1) 
        
        tokenizer2 = self.tokenizer(mt_seq,
                                    truncation=True,
                                    padding=True,
                                    max_length=max_sequence_length,
                                    add_special_tokens=False)
        input2_ids=torch.tensor(tokenizer2['input_ids']).to(self.pretrained_model.device)
        attention_mask2=torch.tensor(tokenizer2['attention_mask']).to(self.pretrained_model.device)
        mt_output2=self.pretrained_model(input_ids=input2_ids,attention_mask=attention_mask2)
        
        diff_seq_embedding = mt_output2.last_hidden_state - wl_output1.last_hidden_state
        
        
        x_seq_feature_ave = torch.mean(diff_seq_embedding,dim=1)
        x_seq_feature_min = torch.min(diff_seq_embedding,dim=1).values
        x_seq_feature_max = torch.max(diff_seq_embedding,dim=1).values

        # --- ddG prediction ---
        x = torch.cat([x_wl, x_mt, x_seq_feature_ave], dim=1)
        ddg_ave = self.regressor_ave(x).squeeze(1)
        
        x = torch.cat([x_wl, x_mt, x_seq_feature_min], dim=1)
        ddg_min = self.regressor_min(x).squeeze(1)
        
        x = torch.cat([x_wl, x_mt, x_seq_feature_max], dim=1)
        ddg_max = self.regressor_max(x).squeeze(1)
        
        return ddg_ave, ddg_min, ddg_max


if __name__ == '__main__':
    mdoel = FusionEnsembleRegressor()