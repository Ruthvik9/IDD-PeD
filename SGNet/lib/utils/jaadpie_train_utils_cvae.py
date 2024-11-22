import sys
import os
import re
import os.path as osp
import numpy as np
import pickle
import time
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data

from lib.utils.eval_utils import eval_jaad_pie, eval_jaad_pie_cvae
from lib.utils.data_utils import bbox_denormalize,cxcywh_to_x1y1x2y2
from lib.losses import cvae, cvae_multi

def generate_filenames(initial_string, obs_length=15, pred_length=45):
    # Generates filenames given the fact that the 'initial_string' is the last frame fo the observation sequence.
    # Find the number in the string using regular expression
    match = re.search(r'(\d+)(\.png)$', initial_string)
    if not match:
        raise ValueError("The initial string must contain a number followed by .png")
    
    base = initial_string[:match.start(1)]
    number = int(match.group(1))
    extension = match.group(2)
    
    filenames = []
    
    # Generate 14 frames before the initial frame (counting backward)
    for i in range(obs_length - 1, 0, -1):
        filenames.append(f"{base}{number - i}{extension}")
    
    # Add the initial frame
    filenames.append(f"{base}{number}{extension}")
    
    # Generate 45 frames after the initial frame (counting forward)
    for i in range(1, pred_length + 1):
        filenames.append(f"{base}{number + i}{extension}")
    
    return filenames

def postprocess_boxes(bboxes):
    # Denormalize the boxes
    bboxes = bbox_denormalize(bboxes,W=1920,H=1440)
    # Convert them to the original x1y1x2y2 format.
    bboxes = cxcywh_to_x1y1x2y2(bboxes)
    return bboxes

def generate_vis_results(index,img_file_all,input_traj_all,target_traj_all,predictions_all,eval_results):
    """
    Generate the vis results given an image
    """
    print("File names")
    print(generate_filenames(img_file_all[index]))
    print("Best prediction")
    print(np.array2string(postprocess_boxes(predictions_all[index][:,eval_results['best_indices'][index],:]),separator=","))
    print("Worst prediction")
    print(np.array2string(postprocess_boxes(predictions_all[index][:,eval_results['worst_indices'][index],:]),separator=","))
    print("Ground truth sequence")
    gt_45 = target_traj_all[index]
    ob_15 = input_traj_all[index]
    final_gt = np.concatenate((ob_15,gt_45),axis=0)
    print(np.array2string(postprocess_boxes(final_gt),separator=","))

def train(model, train_gen, criterion, optimizer, device):
    model.train() # Sets the module in training mode.
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _  = model(inputs=input_traj, map_mask=None, targets=target_traj)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            goal_loss = criterion(all_goal_traj, target_traj)

            train_loss = goal_loss + cvae_loss + KLD_loss.mean()

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
    total_goal_loss /= len(train_gen.dataset)
    total_cvae_loss/=len(train_gen.dataset)
    total_KLD_loss/=len(train_gen.dataset)
    
    return total_goal_loss, total_cvae_loss, total_KLD_loss

def val(model, val_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    model.eval()
    loader = tqdm(val_gen, total=len(val_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(inputs=input_traj, map_mask=None, targets=None,training=False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            

            goal_loss = criterion(all_goal_traj, target_traj)


            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

    val_loss = total_goal_loss/len(val_gen.dataset)\
         + total_cvae_loss/len(val_gen.dataset) + total_KLD_loss/len(val_gen.dataset)
    return val_loss

def test(model, test_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    MSE_15 = 0
    MSE_05 = 0 
    MSE_10 = 0 
    FMSE = 0 
    FIOU = 0
    CMSE = 0 
    CFMSE = 0

    img_file_all = []
    input_traj_all = []
    target_traj_all = []
    predictions_all = []
    best_indices_mse_15 = []
    worst_indices_mse_15 = []
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)
            img_file = data['cur_image_file']

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(inputs=input_traj, map_mask=None, targets=None, training=False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)


            goal_loss = criterion(all_goal_traj, target_traj)

            test_loss = goal_loss + cvae_loss

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            cvae_dec_traj = cvae_dec_traj.to('cpu').numpy()
            # print("It ij what it ij")
            # print("shape of input traj",input_traj_np.shape) # 64,15,4
            # print("Shape of target_traj",target_traj_np[:,-1,:,:].shape) # 64,45,4
            # print("shape of cvae_dec_traj",cvae_dec_traj[:,-1,:,:,:].shape) # 64,45,20,4
            # print("End of it ij what it ij")
            img_file_all.extend(img_file)
            input_traj_all.append(input_traj_np)
            target_traj_all.append(target_traj_np[:,-1,:,:])
            predictions_all.append(cvae_dec_traj[:,-1,:,:,:])
            batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE, batch_CMSE, batch_CFMSE, batch_FIOU,best_indices,worst_indices =\
                eval_jaad_pie_cvae(input_traj_np, target_traj_np[:,-1,:,:], cvae_dec_traj[:,-1,:,:,:])
            # print(best_indices.shape,worst_indices.shape) # (64,) (64,)
            best_indices_mse_15.append(best_indices)
            worst_indices_mse_15.append(worst_indices)
            MSE_15 += batch_MSE_15
            MSE_05 += batch_MSE_05
            MSE_10 += batch_MSE_10
            FMSE += batch_FMSE
            CMSE += batch_CMSE
            CFMSE += batch_CFMSE
            FIOU += batch_FIOU

    
    input_traj_all = np.concatenate(input_traj_all,axis=0)
    target_traj_all = np.concatenate(target_traj_all,axis=0)
    predictions_all = np.concatenate(predictions_all,axis=0)
    best_indices_mse_15 = np.concatenate(best_indices_mse_15,axis=0)
    worst_indices_mse_15 = np.concatenate(worst_indices_mse_15,axis=0)
    performance_indices = {}
    performance_indices['best_indices'] = best_indices_mse_15
    performance_indices['worst_indices'] = worst_indices_mse_15
    all_vis_results = {}
    all_vis_results['img_all'] = img_file_all
    all_vis_results['input_t'] = input_traj_all
    all_vis_results['target_t'] = target_traj_all
    all_vis_results['perf_indices'] = performance_indices
    all_vis_results['predictions'] = predictions_all
    with open("/scratch/ruthvik/SGNet.pytorch/all_test_results.pkl","wb") as f:
        pickle.dump(all_vis_results,f)
    # print("Pls work omg")
    # print(input_traj_all.shape) # 6223,15,4
    # print(target_traj_all.shape) # 6223,45,4
    # print(predictions_all.shape) # 6223,45,20,4
    # print(len(img_file_all)) # 6223
    generate_vis_results(10,img_file_all,input_traj_all,target_traj_all,predictions_all,performance_indices)
            

    
    MSE_15 /= len(test_gen.dataset)
    MSE_05 /= len(test_gen.dataset)
    MSE_10 /= len(test_gen.dataset)
    FMSE /= len(test_gen.dataset)
    FIOU /= len(test_gen.dataset)
    
    CMSE /= len(test_gen.dataset)
    CFMSE /= len(test_gen.dataset)
    

    test_loss = total_goal_loss/len(test_gen.dataset) \
         + total_cvae_loss/len(test_gen.dataset) + total_KLD_loss/len(test_gen.dataset)
    return test_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE


def weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.001)
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
