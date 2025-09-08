import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import argparse
import logging
import random
import os
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
from models.models import GINGATRegressor, FusionEnsembleRegressor
from utils.dataloader import StructureGraphDataset, my_graph_collate_fn
from utils.evaluate import evaluate
from copy import deepcopy
import pandas as pd

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




def predicting(model, device, loader, Model_type):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    test_loss = 0
    with torch.no_grad():
        # for data in tqdm(loader):
        for (data_wl, data_mt, ori_wl_seq, ori_mt_seq) in loader:
            #Get input
            data_wl = data_wl.to(device)
            data_mt = data_mt.to(device)
            
            #Calculate output
            if Model_type == 0:
                y_ddg = model(data_wl, data_mt)
                
            elif Model_type == 1:
                y_ddg_ave, y_ddg_min, y_ddg_max = model(data_wl, data_mt, ori_wl_seq, ori_mt_seq)
                y_ddg = (y_ddg_ave + y_ddg_min + y_ddg_max) / 3
                
            total_preds = torch.cat((total_preds, y_ddg.cpu()), 0)
            
            #Ground truth
            gt_ddg = torch.tensor(data_wl.y).to(device)
            loss = F.mse_loss(y_ddg,gt_ddg)

            
            test_loss = test_loss + loss.item()
            
            total_labels = torch.cat((total_labels, gt_ddg.cpu()), 0)


        test_loss = test_loss/len(loader)
            
    return test_loss, total_labels.numpy().flatten(),total_preds.numpy().flatten()


if __name__ == '__main__':
    
    #Get argument parse
    args = get_args()

    #Test setting
    BATCH_SIZE = 32 
    REDUCTION_MODE = args.Rmode

    

    if args.Model == 0:
        model_name = 'GINGATRegressor'
   
    elif args.Model == 1:
        model_name = 'FusionEnsembleRegressor(finetune-1,ESM2-650M)'
    

    #Step 1:Prepare dataloader
    testdataset_1  = StructureGraphDataset(dataname='S645', augment = False, reduction=args.reduction, reduction_mode=REDUCTION_MODE)
    
    test_loader_1 = DataLoader(testdataset_1, batch_size=BATCH_SIZE, collate_fn = my_graph_collate_fn, shuffle=False, pin_memory=True)
    
    #Step 2: Set  model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if args.Model == 0:
        model = GINGATRegressor(in_channels=20).to(device)
    elif args.Model == 1:  
        model = FusionEnsembleRegressor(in_channels=20, finetune=1, 
                                        pretrained_model=r'./models/esm2_t33_650M_UR50D',
                                        hidden_size=1280).to(device)
    
    
    
    models_list = list()
    randomseed_list = ['34', '42', '1998','2025','3407']
    for randomseed in randomseed_list:
        #Output name
        add_name = '_Train-S4169_ReductionMode{}-{}_RandomSeed-{}'.format(REDUCTION_MODE, args.reduction, randomseed)
        # add_name = '_Train-S4169+augment_ReductionMode{}-{}_RandomSeed-{}'.format(REDUCTION_MODE, args.reduction, randomseed)
        
        model_file_name =  './output/checkpoint/' + model_name + add_name
        
        model_path = model_file_name +'_epoch'+str(args.epoch)+'.model'
        weights = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(weights)

        models_list.append(deepcopy(model))
        
    for test_name, test_loader in [('S645',test_loader_1)]:
        print('##############{}################'.format(test_name))
        p_ave = 0
        for idx, model in enumerate(models_list):
            #Test
            test_loss, g,p = predicting(model, device, test_loader, args.Model)
            
            p_ave = p_ave + p
            Pearson_value, Kendall_value, rmse = evaluate(g,p)
                
            print("testLoss={:.4f}, Pearson_value={:.4f}, Kendall_value={:.4f}, rmse={:.4f}".format(
                test_loss, Pearson_value, Kendall_value, rmse))

            ##保存结果
            total_results = [[p[n], g[n]] for n in range(len(p))]
            total_results = pd.DataFrame(total_results, columns=['y_ddg', 'gt_ddg'])
            total_results.to_csv('./output/results/total_results_{}-randomseed{}.csv'.format(test_name, randomseed_list[idx]), index=False)
        
        
        p_ave = p_ave / 5
        Pearson_value, Kendall_value, rmse = evaluate(g,p_ave)
        print("Ensemble: testLoss={:.4f}, Pearson_value={:.4f}, Kendall_value={:.4f}, rmse={:.4f}".format(
                test_loss, Pearson_value, Kendall_value, rmse))



                

            