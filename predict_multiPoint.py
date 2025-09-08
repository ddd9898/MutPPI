import numpy as np
import torch
from torch import nn
import pandas as pd
from torch_geometric.loader import DataLoader
import argparse
import logging
import random
import os
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
from models.models import GINGATRegressor, FusionEnsembleRegressor
from utils.dataloader_multi import StructureGraphAugDataset, my_graph_aug_collate_fn
from utils.evaluate import evaluate


def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  

    parser.add_argument('--Model', dest='Model', type=int, default=0,
                        help='Model',metavar='E')
    
    parser.add_argument('--reduction', dest='reduction', type=int, default=10,
                        help='reduction',metavar='E')
    
    parser.add_argument('--Rmode', dest='Rmode', type=int, default=0,
                        help='Rmode',metavar='E')
    
    parser.add_argument('--epoch', dest='epoch', type=int, default=500,
                        help='epoch',metavar='E')
    

    return parser.parse_args()


def predicting_multiPoints(model, device, loader, Model_type):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_results = list()

    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    test_loss = 0
    with torch.no_grad():
        for data in loader:
        # for (data_wl, data_mt, ori_wl_seq, ori_mt_seq) in loader:
            y_ddg_sum = list()
            y_ddg_ave_sum = list()
            y_ddg_min_sum = list()
            y_ddg_max_sum = list()

            mutations = list()
            for n in range(int(len(data)/2)):
                data_wl = data[n*2]
                data_mt = data[n*2+1]
                ori_wl_seq = data_wl.seq
                ori_mt_seq = data_mt.seq
                    
                #Get input
                data_wl = data_wl.to(device)
                data_mt = data_mt.to(device)

                mutations.append(data_wl.mutInfo[0].split('_')[1])
                
                gt_ddg_sum = torch.tensor(data_wl.y).to(device)
                if Model_type == 0:
                    y_ddg = model(data_wl, data_mt)
                    y_ddg_sum.append(y_ddg.unsqueeze(1))
                  
                elif Model_type == 1:
                    y_ddg_ave, y_ddg_min, y_ddg_max = model(data_wl, data_mt, ori_wl_seq, ori_mt_seq)
                    y_ddg_ave_sum.append(y_ddg_ave.unsqueeze(1))
                    y_ddg_min_sum.append(y_ddg_min.unsqueeze(1))
                    y_ddg_max_sum.append(y_ddg_max.unsqueeze(1))
                    
            ###Calculate loss
            if Model_type == 0:
                y_ddg_list = torch.cat(y_ddg_sum, dim=1).cpu().numpy().flatten().tolist()

                y_ddg_sum = torch.sum(torch.cat(y_ddg_sum, dim=1), dim=1)
            elif Model_type == 1:
                y_ddg_list = (torch.cat(y_ddg_ave_sum, dim=1) + torch.cat(y_ddg_min_sum, dim=1) + torch.cat(y_ddg_max_sum, dim=1)) / 3
                y_ddg_list = y_ddg_list.cpu().numpy().flatten().tolist()

                y_ddg_ave_sum = torch.sum(torch.cat(y_ddg_ave_sum, dim=1), dim=1)
                y_ddg_min_sum = torch.sum(torch.cat(y_ddg_min_sum, dim=1), dim=1)
                y_ddg_max_sum = torch.sum(torch.cat(y_ddg_max_sum, dim=1), dim=1)
                
                y_ddg_sum = (y_ddg_ave_sum + y_ddg_min_sum + y_ddg_max_sum) / 3

            #
            loss = F.mse_loss(y_ddg_sum,gt_ddg_sum)
            test_loss = test_loss + loss.item()
            
            total_preds = torch.cat((total_preds, y_ddg_sum.cpu()), 0)
            total_labels = torch.cat((total_labels, gt_ddg_sum.cpu()), 0)

            pdb_name = data_wl.mutInfo[0].split('_')[0]
            total_results.append([pdb_name, ','.join(mutations), y_ddg_list, y_ddg_sum.cpu().numpy().flatten().tolist(), gt_ddg_sum.cpu().numpy().flatten().tolist()])
            
        test_loss = test_loss/len(loader)
            
    return test_loss, total_labels.numpy().flatten(),total_preds.numpy().flatten(), total_results


if __name__ == '__main__':
    
    #Get argument parse
    args = get_args()

    #Test setting
    BATCH_SIZE = 1 ##In the testing phase, it is necessary to set this to 1 to ensure that the mutation data is loaded correctly. This is because the test data does not separate mutations of different quantities.
    REDUCTION_MODE = args.Rmode

    UseMutPath = False

    if args.Model == 0:
        model_name = 'GINGATRegressor'

    elif args.Model == 1:
        model_name = 'FusionEnsembleRegressor(finetune-1,ESM2-650M)'


    #Step 1:Prepare dataloader
    testdataset_aug2 = StructureGraphAugDataset(dataname='SM_ZEMu', UseMutPath = UseMutPath, NumAA=args.reduction)
    test_loader_aug2 = DataLoader(testdataset_aug2, batch_size=BATCH_SIZE, collate_fn = my_graph_aug_collate_fn, shuffle=False, pin_memory=True)
    
    testdataset_aug3 = StructureGraphAugDataset(dataname='SM595', UseMutPath = UseMutPath, NumAA=args.reduction)
    test_loader_aug3 = DataLoader(testdataset_aug3, batch_size=BATCH_SIZE, collate_fn = my_graph_aug_collate_fn, shuffle=False, pin_memory=True)
    
    testdataset_aug4 = StructureGraphAugDataset(dataname='SM1124', UseMutPath = UseMutPath, NumAA=args.reduction)
    test_loader_aug4 = DataLoader(testdataset_aug4, batch_size=BATCH_SIZE, collate_fn = my_graph_aug_collate_fn, shuffle=False, pin_memory=True)

    #Step 2: Set  model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if args.Model == 0:
        model = GINGATRegressor(in_channels=20).to(device)
   
    elif args.Model == 1:  
        model = FusionEnsembleRegressor(in_channels=20, finetune=1, 
                                        pretrained_model=r'./models/esm2_t33_650M_UR50D',
                                        hidden_size=1280).to(device)
    
    p_ave = 0
    for randomseed in ['34', '42', '1998','2025','3407']:
        print('##############randomseed{}################'.format(randomseed))
        
        #Output name
        add_name = '_Train-S4169_ReductionMode{}-{}_RandomSeed-{}'.format(REDUCTION_MODE, args.reduction, randomseed)
        # add_name = '_Train-S4169+augment_ReductionMode{}-{}_RandomSeed-{}'.format(REDUCTION_MODE, args.reduction, randomseed)
        model_file_name =  './output/checkpoint/' + model_name + add_name
        
        model_path = model_file_name +'_epoch'+str(args.epoch)+'.model'
        # model_path = model_file_name +'_best.model'
        weights = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(weights)


        ##预测一轮多点数据
        test_loss, g,p, total_results = predicting_multiPoints(model, device, test_loader_aug2, args.Model)
        Pearson_value, Kendall_value, rmse = evaluate(g,p)
        print("SM_ZEMu: testLoss={:.4f}, Pearson_value={:.4f}, Kendall_value={:.4f}, rmse={:.4f}".format(
            test_loss, Pearson_value, Kendall_value, rmse))

        total_results = pd.DataFrame(total_results, columns=['pdb_name', 'mutations', 'y_ddg_list', 'y_ddg_sum', 'gt_ddg_sum'])
        total_results.to_csv('./output/results/total_results_SM_ZEMu-randomseed{}_UseMutPath{}.csv'.format(randomseed, UseMutPath), index=False)

        ##预测一轮多点数据
        test_loss, g,p, total_results = predicting_multiPoints(model, device, test_loader_aug3, args.Model)
        Pearson_value, Kendall_value, rmse = evaluate(g,p)
        print("SM595: testLoss={:.4f}, Pearson_value={:.4f}, Kendall_value={:.4f}, rmse={:.4f}".format(
            test_loss, Pearson_value, Kendall_value, rmse))

        total_results = pd.DataFrame(total_results, columns=['pdb_name', 'mutations', 'y_ddg_list', 'y_ddg_sum', 'gt_ddg_sum'])
        total_results.to_csv('./output/results/total_results_SM595-randomseed{}_UseMutPath{}.csv'.format(randomseed, UseMutPath), index=False)

        ##预测一轮多点数据
        test_loss, g,p, total_results = predicting_multiPoints(model, device, test_loader_aug4, args.Model)
        Pearson_value, Kendall_value, rmse = evaluate(g,p)
        print("SM1124: testLoss={:.4f}, Pearson_value={:.4f}, Kendall_value={:.4f}, rmse={:.4f}".format(
            test_loss, Pearson_value, Kendall_value, rmse))

        total_results = pd.DataFrame(total_results, columns=['pdb_name', 'mutations', 'y_ddg_list', 'y_ddg_sum', 'gt_ddg_sum'])
        total_results.to_csv('./output/results/total_results_SM1124-randomseed{}_UseMutPath{}.csv'.format(randomseed, UseMutPath), index=False)
            