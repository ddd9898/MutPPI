import numpy as np
import torch
from torch import nn
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import argparse
import logging
# from torch.utils.tensorboard import SummaryWriter
import random
import os
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
from models.models import GINGATRegressor, FusionEnsembleRegressor
from utils.dataloader import StructureGraphDataset, my_graph_collate_fn
from utils.evaluate import evaluate


def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--Model', dest='Model', type=int, default=0,
                        help='Model',metavar='E')
    
    parser.add_argument('--reduction', dest='reduction', type=int, default=20,
                        help='reduction',metavar='E')

    parser.add_argument('--randomseed', dest='randomseed', type=int, default=1998,
                        help='randomseed',metavar='E')
    

    return parser.parse_args()



from itertools import cycle
def train(model, device, train_loader, train_loader_aug, optimizer, epoch, Model_type):
    '''
    training function at each epoch
    '''
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    logging.info('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    train_loss = 0
    # for batch_idx, (batch_main, batch_aug) in enumerate(zip(train_loader, cycle(train_loader_aug))):
    for batch_idx, batch_main in enumerate(train_loader):
        
        ############Calculate output###############
        
        optimizer.zero_grad()
        ##################单点数据ddg训练的loss
        data_wl, data_mt, ori_wl_seq, ori_mt_seq = batch_main
        #Get input
        data_wl = data_wl.to(device)
        data_mt = data_mt.to(device)

        gt_ddg = torch.tensor(data_wl.y).to(device)
        if Model_type == 0:
            y_ddg = model(data_wl, data_mt)

            # 对data_wl在第0维进行shuffle，假设data_wl是一个Batch类型的图数据
            perm = torch.randperm(data_wl.num_graphs)
            data_middle = data_wl.__class__.from_data_list([data_wl.get_example(i) for i in perm])
            y_ddg_middle = model(data_wl, data_middle) + model(data_middle, data_mt)
            ###Calculate loss
            loss_main = F.mse_loss(y_ddg,gt_ddg) + F.mse_loss(y_ddg_middle,gt_ddg)   
            
        elif Model_type == 1:
            y_ddg_ave, y_ddg_min, y_ddg_max = model(data_wl, data_mt, ori_wl_seq, ori_mt_seq)

            ###Calculate loss
            loss_main = (F.mse_loss(y_ddg_ave,gt_ddg) + F.mse_loss(y_ddg_min,gt_ddg) + F.mse_loss(y_ddg_max,gt_ddg))/3

            # 随机打乱data_wl和ori_wl_seq，顺序保持一致
            perm = torch.randperm(data_wl.num_graphs)
            data_middle = data_wl.__class__.from_data_list([data_wl.get_example(i) for i in perm])
            ori_middle_seq = [ori_wl_seq[i] for i in perm]
            y_ddg_ave1, y_ddg_min1, y_ddg_max1 = model(data_wl, data_middle, ori_wl_seq, ori_middle_seq)
            y_ddg_ave2, y_ddg_min2, y_ddg_max2 = model(data_middle, data_mt, ori_middle_seq, ori_mt_seq)

            y_ddg_ave_middle = y_ddg_ave1 + y_ddg_ave2
            y_ddg_min_middle = y_ddg_min1 + y_ddg_min2
            y_ddg_max_middle = y_ddg_max1 + y_ddg_max2
            loss_main = loss_main + (F.mse_loss(y_ddg_ave_middle,gt_ddg) + F.mse_loss(y_ddg_min_middle,gt_ddg) + F.mse_loss(y_ddg_max_middle,gt_ddg))/3
 

        loss = loss_main
        train_loss = train_loss + loss.item()

        #Optimize the model
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            logging.info('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                            batch_idx * BATCH_SIZE,
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))
    
    train_loss = train_loss / len(train_loader)
    return train_loss

def predicting(model, device, loader, Model_type):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    logging.info('Make prediction for {} samples...'.format(len(loader.dataset)))
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



def set_seed(seed = 1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True


if __name__ == '__main__':
    #Train setting
    BATCH_SIZE = 8
    LR = 0.00005 #0.0005
    LOG_INTERVAL = 20000 
    NUM_EPOCHS = 20
    AUGMENT = True
    

    #Get argument parse
    args = get_args()
    set_seed(seed = args.randomseed)

    if args.Model == 0:
        model_name = 'GINGATRegressor'     
    elif args.Model == 1:
        model_name = 'FusionEnsembleRegressor(finetune-1,ESM2-650M)'



    #Set log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #Output name
    add_name = '_Train-S4169+augment_ReductionMode{}-{}_RandomSeed-{}'.format(0, args.reduction, args.randomseed)


    logfile = './output/log/log_' + model_name + add_name + '.txt'
    fh = logging.FileHandler(logfile,mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


    #Step 1:Prepare dataloader
    traindataset = StructureGraphDataset(dataname='S4169', augment = AUGMENT, reduction=args.reduction, reduction_mode=0)
    testdataset  = StructureGraphDataset(dataname='S645', augment = False, reduction=args.reduction, reduction_mode=0)
    train_loader = DataLoader(traindataset, batch_size=BATCH_SIZE, collate_fn = my_graph_collate_fn, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(testdataset, batch_size=BATCH_SIZE, collate_fn = my_graph_collate_fn, shuffle=False, pin_memory=True)

    #Step 2: Set  model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if args.Model == 0:
        model = GINGATRegressor(in_channels=20).to(device)

        ## 装载旧模型
        pretrained_model_name = 'GINGATRegressor_Train-S4169_ReductionMode{}-{}_RandomSeed-{}'.format(0, args.reduction, args.randomseed)
        model_file_name =  './output/checkpoint/' + pretrained_model_name
        
        model_path = model_file_name +'_epoch'+str(500)+'.model'
        weights = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(weights,strict=True)

    elif args.Model == 2:
        model = FusionEnsembleRegressor(in_channels=20, finetune=1, 
                                        pretrained_model=r'./models/esm2_t33_650M_UR50D',
                                        hidden_size=1280).to(device)
        
        ## 装载旧模型
        pretrained_model_name = 'FusionEnsembleRegressor(finetune-1,ESM2-650M)_Train-S4169_ReductionMode{}-{}_RandomSeed-{}'.format(0, args.reduction, args.randomseed)
        model_file_name =  './output/checkpoint/' + pretrained_model_name
        
        model_path = model_file_name +'_epoch'+str(100)+'.model'
        weights = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(weights,strict=True)
        
    #Step 3: Train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) #0.001
    #
                                                    
    logging.info(f'''Starting training:
    Model_name:      {model_name}
    Epochs:          {NUM_EPOCHS}
    Batch size:      {BATCH_SIZE}
    Learning rate:   {LR}
    Training size:   {len(traindataset)}
    Validating size: {len(testdataset)}
    Device:          {device.type}
    ''')
    
    
    
    model_file_name =  './output/checkpoint/' + model_name + add_name
    

    for epoch in range(NUM_EPOCHS):
        #Train
        # train_loss = train(model, device, train_loader, train_loader_aug, optimizer, epoch, args.Model)
        train_loss = train(model, device, train_loader, None, optimizer, epoch, args.Model)

        
        #Test
        test_loss, g,p = predicting(model, device, test_loader, args.Model)
        Pearson_value, Kendall_value, rmse = evaluate(g,p) 
        logging.info("Epoch {}: S645: trainloss={:.4f}, testLoss={:.4f}, Pearson_value={:.4f}, Kendall_value={:.4f}, rmse={:.4f}".format(
            epoch, train_loss, test_loss, Pearson_value, Kendall_value, rmse))
        
        
        torch.save(model.state_dict(), model_file_name +'_current.model')
        if epoch == 9:
            torch.save(model.state_dict(), model_file_name +'_epoch'+str(epoch+1)+'.model')
      
    torch.save(model.state_dict(), model_file_name +'_epoch'+str(epoch+1)+'.model')






            

        