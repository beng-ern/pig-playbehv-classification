import os
import torch
import copy
from tqdm import tqdm_notebook
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
from tqdm import tqdm_notebook
import time
import numpy as np
import shutil

import torch.nn.functional as F

from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, precision_score, confusion_matrix, average_precision_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_val(model, params, log=None):
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    robust_dl=params["robust_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    path2cp=params["path2cp"]
    esct=params["esct"]
    
    
    loss_history={
        "train": [],
        "val": [],
        "robust":[]
    }
    
    # store accuracy, AUC, MCC, F1 score, specificity, sensitivity etc.
    metric_history={
        "train_acc": [],
        "val_acc": [],
        "val_auc": [],
        "val_mcc": [],
        "val_sen": [],
        "val_spe": [],
        "val_f1": [],
        "val_prec": [],
        "val_npv": [],
        "robust_acc" : [],
        "robust_mcc": [],
        "robust_sen": [],
        "robust_spe": [],
        "robust_f1": [],
        "robust_prec": [],
        "robust_npv": []
    }
    
    
    
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')
    best_val_auc=0.0
    best_val_acc=0.0
    best_val_mcc=0.0
    
    early_stop_count=0
    
    pred_label_dict = {}
    robust_pred_label_dict = {}
    
    from_=time.time()
    
    for epoch in range(num_epochs):
              
        current_lr=get_lr(opt)
        log('Epoch {}/{}, current lr={}'.format(epoch+1, num_epochs, current_lr))
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        
        since = time.time()
        
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)
        metric_history["train_acc"].append(train_metric)
        
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_auc, true_label, pred_label=loss_epoch(model,loss_func,val_dl,sanity_check,opt=None,val_flag=True)
        
        model.eval()             
        with torch.no_grad():
            robust_loss, robust_acc, robust_auc, robust_true_label, robust_pred_label=loss_epoch(model,loss_func,robust_dl,sanity_check,opt=None,val_flag=True)
        
        end=time.time()
        
        # store the prediction results each epoch
        pred_label_dict[epoch+1] = pred_label
        robust_pred_label_dict[epoch+1] = robust_pred_label
        
        
        # calculate the metrics and confusion matrix
        log("\n Test/validation set metrics:")
        val_mcc, val_sen, val_spe, val_f1, val_prec, val_npv = cal_metric(true_label, pred_label, log) 
        
        log("\n Robust test set metrics:")
        robust_mcc, robust_sen, robust_spe, robust_f1, robust_prec, robust_npv = cal_metric(robust_true_label, robust_pred_label, log)           
        is_best_loss=best_loss>val_loss
        
        is_best_acc=best_val_acc<val_acc
        
        is_best_auc=best_val_auc<val_auc
        
        is_best_mcc=best_val_mcc<val_mcc
        
        save_checkpoint(model.state_dict(), is_best_loss, is_best_auc, is_best_acc, is_best_mcc, path2cp, path2weights)
        
        if is_best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
#             torch.save(model.state_dict(), path2weights)
            print("Copied model weights with best loss!")
            log("--- Best Val Loss ---")
            
        if is_best_auc:
            best_val_auc = val_auc
            log("--- Best AUC ---")
            
        if is_best_acc:
            best_val_acc = val_acc
            log("--- Best ACC ---")
                    
        if is_best_mcc:
            best_val_mcc = val_mcc
            log("--- Best MCC ---")            
        
        # store testing metrics
        loss_history["val"].append(val_loss)
        metric_history["val_acc"].append(val_acc)
        metric_history["val_auc"].append(val_auc)
        metric_history["val_mcc"].append(val_mcc)
        metric_history["val_sen"].append(val_sen)
        metric_history["val_spe"].append(val_spe)
        metric_history["val_f1"].append(val_f1)
        metric_history["val_prec"].append(val_prec)
        metric_history["val_npv"].append(val_npv)

        loss_history["robust"].append(robust_loss)
        metric_history["robust_acc"].append(robust_acc)
        metric_history["robust_mcc"].append(robust_mcc)
        metric_history["robust_sen"].append(robust_sen)
        metric_history["robust_spe"].append(robust_spe)
        metric_history["robust_f1"].append(robust_f1)
        metric_history["robust_prec"].append(robust_prec)
        metric_history["robust_npv"].append(robust_npv)
        
        # decrease the learning rate when the accuracy does not increase for N epochs
        lr_scheduler.step(val_acc)
        
        # load the model with best validation loss when LR is changed
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)
            log("Loading model weights with best val loss!")
        
        log("train loss: %.6f, dev loss: %.6f, train accuracy: %.4f, val accuracy: %.4f, val AUC: %.4f, robust ACC: %.4f" %(train_loss,val_loss,100*train_metric,100*val_acc, val_auc, robust_acc))
        
#         log("train loss: %.6f, dev loss: %.6f, train accuracy: %.4f, val accuracy: %.4f, val AUC: %.4f" %(train_loss,val_loss,100*train_metric,100*val_acc, val_auc))
        
        print("train loss: %.6f, dev loss: %.6f, train accuracy: %.4f, val accuracy: %.4f" %(train_loss,val_loss,100*train_metric,100*val_acc))
        print("-"*10) 
        
        log(f'\nEpoch Running Time: {int((end-since)//60)}m {int((end-since)%60)}s\n')
        
        # early stopping
        if is_best_loss:
            early_stop_count=0
        else:
            early_stop_count+=1

        log(f'Early_stop_count: {early_stop_count}\n\n')

        if early_stop_count==esct:
            log(f'\nEarly Stopped because validation loss does not decrease for {esct} epochs')
            break
        
    to_=time.time()
    log(f'\nTotal Running Time: {int((to_-from_)//60)}m {int((to_-from_)%60)}s')    
    
    log('\n\n ======= Metric Report =======\n')
    log(f"Train Accuracy {metric_history['train_acc']} \n")
    log(f"Train Loss {loss_history['train']} \n")
    
    log(f"Validation Loss {loss_history['val']} \n")
    log(f"Validation Accuracy {metric_history['val_acc']} \n")
    log(f"Validation MCC {metric_history['val_mcc']} \n")
    
    log(f"Robust Loss {loss_history['robust']} \n")
    log(f"Robust Accuracy {metric_history['robust_acc']} \n")
    log(f"Robust MCC {metric_history['robust_mcc']} \n")
    
    
    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history, pred_label_dict, robust_pred_label_dict   
#     return model, loss_history, metric_history, pred_label_dict

# get learning rate 
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    pred_label = output.argmax(dim=1)
    prob = F.softmax(output, dim=1)[:, 1]
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects, prob, pred_label

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b, prob_b, pred_label_b = metrics_batch(output,target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b, prob_b, pred_label_b
    

def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None,val_flag=False):
    running_loss=0.0
    running_acc=0.0
    
    
    if val_flag:
        len_data = len(dataset_dl.dataset)
        true_label= np.empty(1)
        predict_prob= np.empty(1)
        pred_label= np.empty(1)
        
#         for xb, coord_seq_b, yb in tqdm_notebook(dataset_dl):
        for xb, yb in tqdm_notebook(dataset_dl):            
            true_label = np.append(true_label, yb.numpy())

            xb=xb.to(device)
#             coord_seq_b=coord_seq_b.to(device)
            yb=yb.to(device)
#             output=model(xb, coord_seq_b)
            output=model(xb)        
            loss_b,acc_b,prob_b,pred_b=loss_batch(loss_func, output, yb, opt)
            running_loss+=loss_b

            predict_prob = np.append(predict_prob, prob_b.cpu().numpy())
            pred_label = np.append(pred_label, pred_b.cpu().numpy())

            if acc_b is not None:
                running_acc+=acc_b
            if sanity_check is True:
                break
                
        loss=running_loss/float(len_data)
        acc=running_acc/float(len_data)
        
        true_label = true_label[1:]
        predict_prob = predict_prob[1:]
        pred_label = pred_label[1:]
    
        auc=roc_auc_score(true_label, predict_prob)
        
        return loss, acc, auc, true_label, pred_label

        
    else:
        len_data = dataset_dl.sampler.num_samples
#         for xb, coord_seq_b, yb in tqdm_notebook(dataset_dl):
        for xb, yb in tqdm_notebook(dataset_dl):            
            xb=xb.to(device)
#             coord_seq_b=coord_seq_b.to(device)
            yb=yb.to(device)
#             output=model(xb, coord_seq_b)
            output=model(xb)        
            loss_b,acc_b,prob_b,pred_b=loss_batch(loss_func, output, yb, opt)
            running_loss+=loss_b

            if acc_b is not None:
                running_acc+=acc_b
            if sanity_check is True:
                break
                
        loss=running_loss/float(len_data)
        acc=running_acc/float(len_data)
        
        return loss, acc



def cal_metric(true_label_tot, pred_label_tot, log=None):
                     
    cnf_matrix = confusion_matrix(true_label_tot, pred_label_tot)
    
    # print(cnf_matrix)
    #[[1 1 3]
    # [3 2 2]
    # [1 3 1]]

    # return the counts based on the group assigned as 'positive' class
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

#     FP = cnf_matrix[0][1]
#     FN = cnf_matrix[1][0]
#     TP = cnf_matrix[1][1]
#     TN = cnf_matrix[0][0]
    
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    # F1 score
    F1 = (2*PPV*TPR) / (PPV+TPR)
    # Matthew corelation coefficient
    matt_cor = (TP*TN - FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    
    if log:
        log("\n        pred 0  pred 1")
        log(f"true 0 [ {cnf_matrix[0][0]} , {cnf_matrix[0][1]} ]")
        log(f"true 1 [ {cnf_matrix[1][0]} , {cnf_matrix[1][1]} ]\n")
        log(f"Recall: {TPR}")
        log(f"Specificity: {TNR}")
        log(f"Precision: {PPV}")
        log(f"FPR: {FPR}")
        log(f"FNR: {FNR}")
        log(f"F1 score: {F1}")
        log(f"Matthew correlation coefficient: {matt_cor} \n")
        
#     dict_ = {'recall': TPR, 'precision': PPV, 'fpr': FPR, 'fnr': FNR}
#     return dict_
    return matt_cor[0], TPR[1], TNR[1], F1[1], PPV[1], NPV[1]



def plot_loss(loss_hist, metric_hist):

    num_epochs= len(loss_hist["train"])

    plt.title("Train-Val Loss")
    plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

    plt.title("Train-Val Accuracy")
    plt.plot(range(1,num_epochs+1), metric_hist["train_acc"],label="train")
    plt.plot(range(1,num_epochs+1), metric_hist["val_acc"],label="val")
    plt.plot(range(1,num_epochs+1), metric_hist["robust_acc"],label="robust")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show() 
    
    
def save_checkpoint(state, is_best_loss, is_best_auc, is_best_acc, is_best_mcc, cp_file=None, weight_file=None):    
    filename = cp_file+".pt"
    
    torch.save(state, filename)

    if is_best_loss:
        shutil.copyfile(filename, weight_file+"_best_loss.pt")
        
    if is_best_auc:
        shutil.copyfile(filename, weight_file+"_best_AUC.pt")   
    
    if is_best_acc:
        shutil.copyfile(filename, weight_file+"_best_accuracy.pt")
        
    if is_best_mcc:
        shutil.copyfile(filename, weight_file+"_best_MCC.pt")