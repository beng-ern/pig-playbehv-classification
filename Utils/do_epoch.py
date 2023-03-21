import time
import random
import numpy as np
import pandas as pd

import torch

from Utils.metric import *
from Utils.fgsm import FGSM_Attack



def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def get_true_label(df, video_path):
    label = [df.loc[df['video_path']==p, 'label'].unique()[0] for p in video_path]
    label = [0 if l==0 else 1 for l in label]
    return label

def Train(train_loader, models, criterion, optimizers, feature_collector, device, noisy=None, log=None, args=None):
    
    info_dict = {'loss_for_backward':[], 'true_labels':[], 'each_recon_loss':[], 'n_recon_loss':[], 'an_recon_loss':[], 'ori_an_recon_loss':[], 'FGSM_recon_loss': [], 'video_path':[]}
       
    # set mode: train
    for key in models:
        models[key].train()
        
    
    # AMP
    scaler = torch.cuda.amp.GradScaler()
    
    #----- do one epoch
    
    for video, labels, video_path in train_loader:
        
        video = video.to(device)
        labels = labels.to(device)       
        
        # FGSM setting
        # if (args.fgsm_type=='video') and (sum(labels==0)>1):
        #     video.requires_grad = True    # 이게 문제였음..!!

                    #   -> feature_x 에 대한 grad 를 사용할 건데
                    #  video가 requires_grad=True 된 후, 사용되지 않아서 retain_graph 후,
                    #  loss.backward()에서 free 되지 않은 듯!!
        
        
        with torch.cuda.amp.autocast():
            x = models['Spatial_Encoder'](video.permute(0,2,1,3,4))
            x = x.permute(1,0,2)

            x_recon, _ = models['Temp_EncDec'](x)
            x = x.permute(1,0,2)


            if (args.fgsm_type=='feature') and (sum(labels==0)>0):
                x.retain_grad()

            output = criterion(x_recon, x)
                
            
            loss, n_recon_loss, an_recon_loss, each_recon_loss, ori_an_recon_loss = cal_criterion(output, labels, device, args)
            
            # variational inference
            kl_loss = models['Temp_EncDec'].kl_loss()
            loss += kl_loss


        if (args.FGSM) and (args.fgsm_type=='feature') and (sum(labels==0)>0):
            optimizers['Spatial_Encoder'].zero_grad()
            optimizers['Temp_EncDec'].zero_grad()
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizers['Spatial_Encoder'])
            scaler.step(optimizers['Temp_EncDec'])
            scaler.update()
            
            
        else:
            optimizers['Spatial_Encoder'].zero_grad()
            optimizers['Temp_EncDec'].zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizers['Spatial_Encoder'])
            scaler.step(optimizers['Temp_EncDec'])
            scaler.update()
        
        
        #----- FGSM
        
        if (args.FGSM):
            # if (args.fgsm_type=='video') and (sum(labels==0)==args.batch_size):            

            #     fgsm_loss = FGSM_Attack(models, video, labels, criterion, device, args)               

            #     optimizers['Spatial_Encoder'].zero_grad()
            #     optimizers['Temp_EncDec'].zero_grad()
            #     scaler.scale(fgsm_loss).backward()
            #     scaler.step(optimizers['Spatial_Encoder'])
            #     scaler.step(optimizers['Temp_EncDec'])
            #     scaler.update()

            #     info_dict['FGSM_recon_loss'].append(fgsm_loss.item())

            if (args.fgsm_type=='feature') and (sum(labels==0)>0):

                fgsm_loss = FGSM_Attack(models, x, labels, criterion, feature_collector, device, args)

                optimizers['Spatial_Encoder'].zero_grad()
                optimizers['Temp_EncDec'].zero_grad()
                scaler.scale(fgsm_loss).backward()    
                scaler.step(optimizers['Spatial_Encoder'])            
                scaler.step(optimizers['Temp_EncDec'])
                scaler.update()

                info_dict['FGSM_recon_loss'].append(fgsm_loss.item())


        #----- just for record
        info_dict['loss_for_backward'].append(loss.item())

        if n_recon_loss.item()!=0:
            info_dict['n_recon_loss'].append(n_recon_loss.item())
        if an_recon_loss.item()!=0:
            info_dict['an_recon_loss'].append(an_recon_loss.item())
            info_dict['ori_an_recon_loss'].append(ori_an_recon_loss.item())
            
            
        info_dict['each_recon_loss'].extend(each_recon_loss.detach().cpu().numpy().tolist())
        info_dict['video_path'].extend(video_path)
        info_dict['true_labels'].extend(labels.cpu().numpy().tolist())

    ##----- just for record
    #----- loss for epoch
    epoch_loss = np.mean(info_dict['loss_for_backward'])  # need to be returned !!  from 'Train' function
    epoch_n_recon_loss = np.mean(info_dict['n_recon_loss'])
    epoch_an_recon_loss = np.mean(info_dict['an_recon_loss'])
    epoch_ori_an_recon_loss = np.mean(info_dict['ori_an_recon_loss'])
    
    if info_dict['FGSM_recon_loss']:
        epoch_FGSM_recon_loss = np.mean(info_dict['FGSM_recon_loss'])
        
    
    
    #----- Metric Calculation
    metric = Prediction(info_dict['each_recon_loss'], info_dict['true_labels'])
    
    #vis_hist(info_dict['each_recon_loss'], info_dict['true_labels'], args)
    
    
    if 'unsup' not in args.exp:
        _, _, tot_recall, normal_recall, anomaly_recall, best_tot_recall_threshold, auc, tot_precision, normal_precision, anomaly_precision, dict_, auprc = metric.get_prediction(args, log)
    else:
        tot_recall, normal_recall, anomaly_recall, best_tot_recall_threshold, auc, tot_precision, normal_precision, anomaly_precision, auprc = 0,0,0,0,0,0,0,0, 0
        dict_ = {'fpr':[0,0], 'fnr':[0,0]}

    
    
    # log
    log(f"[+] Train Loss: {epoch_loss :.4f}, Tr. Normal Recon Loss: {epoch_n_recon_loss :.4f}, Tr. Anomaly Recon Loss: {epoch_an_recon_loss :.4f}, Tr. Ori_an_recon_loss: {epoch_ori_an_recon_loss :.4f}")

    log(f"[+] Ta. AUC: {auc :.4f}, Ta. AUPRC: {auprc :.4f}, Ta. FPR: {dict_['fpr'][1] :.3f}, Ta. FNR: {dict_['fnr'][1] :.3f}\n")
    if info_dict['FGSM_recon_loss']:
        log(f"[+] Tr. FGSM_recon_loss: {epoch_FGSM_recon_loss :.4f}\n")

    
    return epoch_loss



def Valid(valid_loader, models, criterion, device, log=None, args=None):
    
    info_dict={'loss_for_backward':[], 'true_labels':[], 'each_recon_loss':[], 'n_recon_loss':[], 'an_recon_loss':[], 'ori_an_recon_loss':[], 'video_path':[]}
       
    # set mode: test
    for key in models:
        models[key].eval()
    
    #----- do one epoch

    for video, labels, video_path in valid_loader:

        
        video = video.to(device)
        labels = labels.to(device)

        
        with torch.no_grad():
            with torch.cuda.amp.autocast():

                x = models['Spatial_Encoder'](video.permute(0,2,1,3,4))
                x = x.permute(1,0,2)

                x_recon, _ = models['Temp_EncDec'](x)
                x = x.permute(1,0,2)

                output = criterion(x_recon, x)
                    
                loss, n_recon_loss, an_recon_loss, each_recon_loss, ori_an_recon_loss = cal_criterion(output, labels, device, args)
                
                # variational inference
                kl_loss = models['Temp_EncDec'].kl_loss()
                loss += kl_loss


        #----- just for record
        info_dict['loss_for_backward'].append(loss.item())
        
        if n_recon_loss.item()!=0:
            info_dict['n_recon_loss'].append(n_recon_loss.item())
        if an_recon_loss.item()!=0:
            info_dict['an_recon_loss'].append(an_recon_loss.item())
            info_dict['ori_an_recon_loss'].append(ori_an_recon_loss.item())
            
        info_dict['true_labels'].extend(labels.detach().cpu().numpy().tolist())
        info_dict['each_recon_loss'].extend(each_recon_loss.detach().cpu().numpy().tolist())
        info_dict['video_path'].extend(video_path)

    ##----- just for record
    #----- loss for epoch
    epoch_loss = np.mean(info_dict['loss_for_backward'])  # need to be returned !!  from 'Train' function
    epoch_n_recon_loss = np.mean(info_dict['n_recon_loss'])
    epoch_an_recon_loss = np.mean(info_dict['an_recon_loss'])
    epoch_ori_an_recon_loss = np.mean(info_dict['ori_an_recon_loss'])
    
    
    #----- AORUC Calculation
    metric = Prediction(info_dict['each_recon_loss'], info_dict['true_labels'])
    _, _, tot_recall, normal_recall, anomaly_recall, best_tot_recall_threshold, auc, tot_precision, normal_precision, anomaly_precision, dict_, auprc = metric.get_prediction(args, log)
    
    #vis_hist(info_dict['each_recon_loss'], info_dict['true_labels'], args)
    
    
    # log
    log(f"[+] Validation Loss: {epoch_loss :.4f}, Va. Normal Recon Loss: {epoch_n_recon_loss :.4f}, Va. Anomaly Recon Loss: {epoch_an_recon_loss :.4f}, Va. Ori_an_recon_loss: {epoch_ori_an_recon_loss :.4f}")

    log(f"[+] Va. AUC: {auc :.4f}, Va. AUPRC: {auprc :.4f}, Va. FPR: {dict_['fpr'][1] :.3f}, Va. FNR: {dict_['fnr'][1] :.3f}")
    
    return epoch_loss, tot_recall, anomaly_recall, best_tot_recall_threshold, auc, auprc






def cal_criterion(criterion_output, labels, device, args=None):

    normal_idx = labels==0
    anomaly_idx = labels!=0

    each_recon_loss = []
    normal_list = []
    anomaly_list = []
    ori_an_recon_loss = []

    for loss_mat, label in zip(criterion_output, labels):
        each_recon_loss.append(loss_mat.mean())
    
    # normal
    for i, lab in enumerate(labels):
        if lab == 0:
            normal_list.append(each_recon_loss[i])
            anomaly_list.append(torch.zeros(1, device=device))

            ori_an_recon_loss.append(torch.zeros(1, device=device))
    # anomaly
        else:
            anomaly_list.append(1/(each_recon_loss[i]))
            normal_list.append(torch.zeros(1, device=device))

            ori_an_recon_loss.append(each_recon_loss[i])


    n_recon_loss = sum(normal_list)/(len(labels)+1e-5)                #len(labels)     # need to be decreased
    an_recon_loss = args.ld*sum(anomaly_list)/(len(labels)+1e-5)     #len(labels) # need to be decreased
    loss_for_backward = n_recon_loss + an_recon_loss

    ori_an_recon_loss = sum(ori_an_recon_loss)/(sum(anomaly_idx)+1e-5)  # need to be increased

    each_recon_loss = torch.tensor(each_recon_loss, device=device) 

    return loss_for_backward, n_recon_loss, an_recon_loss, each_recon_loss, ori_an_recon_loss
