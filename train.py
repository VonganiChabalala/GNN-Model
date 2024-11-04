import os
import sys
import pandas as pd
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
from util import config, file_dir
from graph import Graph
from dataset import HazeData



from model.MLP import MLP
from model.LSTM import LSTM
from model.GRU import GRU
from model.GC_LSTM import GC_LSTM
from model.nodesFC_GRU import nodesFC_GRU
from model.PM25_GNN import PM25_GNN
from model.PM25_GNN_nosub import PM25_GNN_nosub
#from model.PM25_GCNs import PM25_GCNs
import arrow
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
import glob
import shutil
import wandb
#wandb.init(settings=wandb.Settings(console='off')) 
wandb.init(project="Vongani_GNN", entity="vonganichabalala8")
#os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]= "20"
#wandb.config1 = {
  #"learning_rate": 0.001,
  #"epochs": 100,
  #"batch_size": 128
#}

torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

graph = Graph()
city_num = graph.node_num

batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
hist_len = config['train']['hist_len']
pred_len = config['train']['pred_len']
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
results_dir = file_dir['results_dir']
dataset_num = config['experiments']['dataset_num']
exp_model = config['experiments']['model']
exp_repeat = config['train']['exp_repeat']
save_npy = config['experiments']['save_npy']
criterion = nn.MSELoss()

train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')
in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1]
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std
#pm25_mean_val, pm25_std_val = val_data.pm25_mean, val_data.pm25_std
#pm25_mean_train, pm25_std_train = train_data.pm25_mean, train_data.pm25_std

def get_metric(predict_epoch, label_epoch):
    haze_threshold = 10
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return rmse, mae, csi, pod, far
    
    
def get_metric_train(predict_epoch, label_epoch):
    haze_threshold = 25
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi_train = hit / (hit + falsealarm + miss)
    pod_train = hit / (hit + miss)
    far_train = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae_train = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse_train = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return rmse_train, mae_train, csi_train, pod_train, far_train
    
def get_metric_val(predict_epoch, label_epoch):
    haze_threshold = 10
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi_val = hit / (hit + falsealarm + miss)
    pod_val = hit / (hit + miss)
    far_val = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae_val = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse_val = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return rmse_val, mae_val, csi_val, pod_val, far_val


def get_exp_info():
    exp_info =  '============== Train Info ==============\n' + \
                'Dataset number: %s\n' % dataset_num + \
                'Model: %s\n' % exp_model + \
                'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
                'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
                'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
                'City number: %s\n' % city_num + \
                'Use metero: %s\n' % config['experiments']['metero_use'] + \
                'batch_size: %s\n' % batch_size + \
                'epochs: %s\n' % epochs + \
                'hist_len: %s\n' % hist_len + \
                'pred_len: %s\n' % pred_len + \
                'weight_decay: %s\n' % weight_decay + \
                'early_stop: %s\n' % early_stop + \
                'lr: %s\n' % lr + \
                '========================================\n'
    return exp_info


def get_model():
    if exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim)
    elif exp_model == 'LSTM':
        return LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GRU':
        return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'nodesFC_GRU':
        return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GC_LSTM':
        return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
    elif exp_model == 'PM25_GNN':
        return PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_nosub':
        return PM25_GNN_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    else:
        raise Exception('Wrong model name!')


def train(train_loader, model, optimizer):

    model.train()
    wandb.watch(model, log='all', log_freq=100)
    predict_list = []
    label_list = []
    time_list = []
    train_loss = 0
    for batch_idx, data in tqdm(enumerate(train_loader)):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        pm25_pred_train = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1)* pm25_std + pm25_mean * pm25_std + pm25_mean
        pm25_label_train = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_train)
        label_list.append(pm25_label_train)
        time_list.append(time_arr.cpu().detach().numpy())


    train_loss /= batch_idx + 1
    predict_epoch_train = np.concatenate(predict_list, axis=0)
    label_epoch_train = np.concatenate(label_list, axis=0)
    time_epoch_train = np.concatenate(time_list, axis=0)
    predict_epoch_train[predict_epoch_train < 0] = 0

    #wandb.log({"train_loss": train_loss})
    return train_loss,predict_epoch_train, label_epoch_train,time_epoch_train


def val(val_loader, model):
    model.val()
    wandb.watch(model, log='all', log_freq=100)
    predict_list = []
    label_list = []
    time_list = []
    val_loss = 0
    for batch_idx, data in tqdm(enumerate(val_loader)):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        val_loss += loss.item()
        
        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1)* pm25_std + pm25_mean * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        time_list.append(time_arr.cpu().detach().numpy())


    val_loss /= batch_idx + 1
    predict_epoch_val = np.concatenate(predict_list, axis=0)
    label_epoch_val = np.concatenate(label_list, axis=0)
    time_epoch_val = np.concatenate(time_list, axis=0)
    predict_epoch_val[predict_epoch_val < 0] = 0

    #wandb.log({"val_loss": val_loss})
    wandb.log({"val_loss": val_loss})
    return val_loss,predict_epoch_val, label_epoch_val,time_epoch_val 


def test(test_loader, model):
    model.test()
    wandb.watch(model, log='all', log_freq=100)
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        test_loss += loss.item()

        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        time_list.append(time_arr.cpu().detach().numpy())

    test_loss /= batch_idx + 1

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0

    #wandb.log({"test_loss": test_loss})
    return test_loss, predict_epoch, label_epoch, time_epoch


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def main():
    exp_info = get_exp_info()
    print(exp_info)

    exp_time = arrow.now().format('YYYYMMDDHHmmss')

    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, csi_list, pod_list, far_list = [], [], [], [], [], [], [], []
    rmse_list_train, mae_list_train, csi_list_train, pod_list_train, far_list_train = [], [], [], [], []
    rmse_list_val, mae_list_val, csi_list_val, pod_list_val, far_list_val = [], [], [], [], []
    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

        model = get_model()
        model = model.to(device)
        model_name = type(model).__name__

        print(str(model))

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), '%02d' % exp_idx)
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')

        val_loss_min = 100000
        best_epoch = 0

        train_loss_, val_loss_ = 0, 0
        val_loss_exp=[]
        train_loss_exp=[]
        #test_train_exp=[]
        epochss=[]
        for epoch in range(epochs):
            print('\nTrain epoch %s:' % (epoch))
            
            train_loss,predict_epoch_train, label_epoch_train,time_epoch_train = train(train_loader, model, optimizer)
            val_loss,predict_epoch_val, label_epoch_val,time_epoch_val = val(val_loader, model)
            wandb.log({"val_loss": val_loss})

	        
            print('train_loss: %.4f' % train_loss)
            print('val_loss: %.4f' % val_loss)

            
            #if train_loss < val_loss_min:
            	#rmse_train, mae_train, csi_train, pod_train, far_train = get_metric(predict_epoch_train, label_epoch_train)
            predict_epoch_train_, label_epoch_train_,time_epoch_train_=predict_epoch_train, label_epoch_train,time_epoch_train
            if save_npy:
                np.save(os.path.join(exp_model_dir, 'train_losss.npy'), train_loss)
                np.save(os.path.join(exp_model_dir, 'val_losss.npy'), val_loss)
                #np.save(os.path.join(exp_model_dir, 'time_train.npy'), time_epoch_train_)

            if epoch - best_epoch > early_stop:
                break
            val_loss_list=[]
            #if epoch%10==0:
                #val_loss=val_loss.cpu().detach().numpy()
                #val_loss_list.append(val_loss)
            train_loss_exp.append(train_loss)
            val_loss_exp.append(val_loss)
            epochss.append(epoch)

            #rmse_train, mae_train, csi_train, pod_train, far_train = get_metric(predict_epoch_train, label_epoch_train)
            #rmse_val, mae_val, csi_val, pod_val, far_val = get_metric(predict_epoch_val, label_epoch_val)
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                print('Minimum val loss!!!')
                torch.save(model.state_dict(), model_fp)
                print('Save model: %s' % model_fp)

                test_loss, predict_epoch, label_epoch, time_epoch = test(test_loader, model)
                print('val_loss: %.4f' % test_loss)
                wandb.log({"train_loss": train_loss}) 
                train_loss_, val_loss_ = train_loss, val_loss
                predict_epoch_val_, label_epoch_val_,time_epoch_val_=predict_epoch_val, label_epoch_val,time_epoch_val
                #predict_epoch_train_, label_epoch_train_,time_epoch_train_=predict_epoch_train, label_epoch_train,time_epoch_train
                rmse, mae, csi, pod, far = get_metric(predict_epoch, label_epoch)
                
                print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far))

                if save_npy:
                    np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
                    np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)
                    np.save(os.path.join(exp_model_dir, 'time.npy'), time_epoch)
                    #np.save(os.path.join(exp_model_dir, 'predict_train.npy'), predict_epoch_train_)
                    #np.save(os.path.join(exp_model_dir, 'label_train.npy'), label_epoch_train_)
                    #np.save(os.path.join(exp_model_dir, 'time_train.npy'), time_epoch_train_)
                    #np.save(os.path.join(exp_model_dir, 'predict_val.npy'), val_loss_list)
                    #np.save(os.path.join(exp_model_dir, 'label_val.npy'), label_epoch_val_)
                    #np.save(os.path.join(exp_model_dir, 'time_val.npy'), time_epoch_val_)                    
                    
        df = pd.DataFrame(train_loss_exp) 
        df.to_csv('train_loss_exp'+str(exp_idx)+'.csv')   
        df = pd.DataFrame(val_loss_exp) 
        df.to_csv('val_loss_exp'+str(exp_idx)+'.csv')
        df = pd.DataFrame(epochss) 
        df.to_csv('epochs'+str(exp_idx)+'.csv') 
        #wandb.log({"train_loss": train_loss_exp,"val_loss": val_loss_exp})             
        #wandb.log({"train_loss": train_loss},{"val_loss": val_loss})
        print("READ ME: ",test_loss)
        print("READ ME: ",train_loss)
        print("READ ME: ",val_loss)
        train_loss_list.append(train_loss_)
        #df = pd.DataFrame(lst)
        val_loss_list.append(val_loss_)
        #df = pd.DataFrame(lst)
        test_loss_list.append(test_loss)
        #df = pd.DataFrame(lst)
        rmse_list.append(rmse)
        mae_list.append(mae)
        csi_list.append(csi)
        pod_list.append(pod)
        far_list.append(far)

        wandb.log({"rmse": rmse,"mae": mae,"csi": csi,"pod": pod,"far": far}) 
        
        #rmse_list_train.append(rmse_train)
        #mae_list_train.append(mae_train)
        #csi_list_train.append(csi_train)
        #pod_list_train.append(pod_train)
        #far_list_train.append(far_train)
        
        #rmse_list_val.append(rmse_val)
        #mae_list_val.append(mae_val)
        #csi_list_val.append(csi_val)
        #pod_list_val.append(pod_val)
        #far_list_val.append(far_val)        
        

        print('\nNo.%2d experiment results:' % exp_idx)
        print(
            'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f' % (
            train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far))
    #df = pd.DataFrame(train_loss_list)
    #df.to_csv('train_loss_list.csv')
    
    #df = pd.DataFrame(val_loss_list)
    #df.to_csv('val_loss_list.csv')
    
    #df = pd.DataFrame(test_loss_list)
    #df.to_csv('test_loss_list.csv')   
    
    exp_metric_str = '---------------------------------------\n' + \
                     'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                     'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
                     'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                     'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
                     'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
                     'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list)) + \
                     'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list)) + \
                     'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list))

    metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metric.txt')
    with open(metric_fp, 'w') as f:
        f.write(exp_info)
        f.write(str(model))
        f.write(exp_metric_str)

    print('=========================\n')
    print(exp_info)
    print(exp_metric_str)
    print(str(model))
    print(metric_fp)


if __name__ == '__main__':
    main()
