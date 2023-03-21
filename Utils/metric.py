import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os.path as osp

from sklearn.metrics import recall_score, multilabel_confusion_matrix, roc_auc_score, accuracy_score, precision_score, confusion_matrix, average_precision_score


# Confusion Table & Accuracy
class Prediction:
    def __init__(self, total_recon_loss, true_label_tot, test=False, best_recall_threshold=None):
        self.total_recon_loss=total_recon_loss
        self.true_label_tot=true_label_tot
        
        self.test=test
        self.best_recall_threshold=best_recall_threshold
                
    def single_threshold(self, threshold):
        
        # pred_label_tot
        self.pred_label_tot=[]
        for error in self.total_recon_loss:
            if (threshold<error):
                self.pred_label_tot.append(1)
            else:
                self.pred_label_tot.append(0)
                
        # true_anomaly_bool
        self.bool_true_an=[]
        for i in self.true_label_tot:
            if i==1:
                self.bool_true_an.append(True)
            else:
                self.bool_true_an.append(False)
                
        # true_anomaly_label
        self.true_label_an=pd.Series(self.true_label_tot).loc[self.bool_true_an].to_list()
        # pred_anomaly_label
        self.pred_label_an=[]
        self.anomaly_error=pd.Series(self.total_recon_loss).loc[self.bool_true_an]
        
        for error in self.anomaly_error:
            if (threshold<error):
                self.pred_label_an.append(1)
            else:
                self.pred_label_an.append(0)
        
        
    def cal_for_curve(self, threshold):        
        self.single_threshold(threshold)
        normal_recall, anomaly_recall=recall_score(self.true_label_tot, self.pred_label_tot, average=None, zero_division=0)
        tot_recall = (normal_recall+anomaly_recall)/2
#         anomaly_recall=accuracy_score(self.true_label_an, self.pred_label_an)
        normal_precision, anomaly_precision = precision_score(self.true_label_tot, self.pred_label_tot, average=None, zero_division=0)
        tot_precision = (normal_precision+anomaly_precision)/2        
                
        return tot_recall, normal_recall, anomaly_recall, tot_precision, normal_precision, anomaly_precision
    
    
    def get_prediction(self, args, log):
        
        if not self.test:
            self.max_loss=max(self.total_recon_loss)
            self.min_loss=min(self.total_recon_loss)
            step=(self.max_loss-self.min_loss)/2000
            self.threshold=[self.min_loss+i*step for i in range(1,2000)]
            
            # AUROC
            
            self.total_probability=((np.array(self.total_recon_loss)-self.min_loss)/(self.max_loss-self.min_loss)).tolist()

            try:
                self.auc=roc_auc_score(self.true_label_tot, self.total_probability)
            except:
                self.total_probability = np.nan_to_num(self.total_probability)
                self.auc=roc_auc_score(self.true_label_tot, self.total_probability)
                
            self.auprc = average_precision_score(self.true_label_tot, self.total_probability)

            self.total_recall_list=[]
            self.anomaly_recall_list=[]
            
            for num in self.threshold:            
                tot_recall, _, anomaly_recall, _, _, _ = self.cal_for_curve(num)    
                self.total_recall_list.append(tot_recall)
                self.anomaly_recall_list.append(anomaly_recall)
                
            # best total recall
            self.max_recall=max(self.total_recall_list)
            max_idx=self.total_recall_list.index(self.max_recall)
            self.best_tot_recall_threshold=self.threshold[max_idx]
            
            # best anomaly recall
            self.max_recall=max(self.anomaly_recall_list)
            max_idx=self.anomaly_recall_list.index(self.max_recall)
            self.best_anomaly_recall_threshold=self.threshold[max_idx]            


            # Final result
            tot_recall, normal_recall, anomaly_recall, tot_precision, normal_precision, anomaly_precision = self.cal_for_curve(self.best_tot_recall_threshold)
            
            
            rest_metric_dict = cal_metric(self.true_label_tot, self.pred_label_tot, log)
            

            return self.true_label_tot, self.pred_label_tot, tot_recall, normal_recall, anomaly_recall, self.best_tot_recall_threshold, self.auc, tot_precision, normal_precision, anomaly_precision, rest_metric_dict, self.auprc
        
        else:
            
            self.max_loss=max(self.total_recon_loss)
            self.min_loss=min(self.total_recon_loss)
           
            # AUROC
            self.total_probability=((np.array(self.total_recon_loss)-self.min_loss)/(self.max_loss-self.min_loss)).tolist()
            self.auc=roc_auc_score(self.true_label_tot, self.total_probability)
            tot_recall, anomaly_recall = self.cal_for_curve(self.best_recall_threshold)
#             tot_recall, _=self.cal_for_curve(self.best_recall_threshold)
#             _, anomaly_recall=self.cal_for_curve(self.best_recall_threshold)
            return tot_recall, anomaly_recall, self.best_recall_threshold, self.auc






def vis_hist(total_recon_loss, true_label_tot, args):
    result_df = pd.DataFrame({'true':true_label_tot, 'recon_loss':total_recon_loss})
    
    sns.distplot(result_df[result_df['true']==0]['recon_loss'], color = 'blue', hist=True, kde=False, rug=False, label = 'Normal')

    sns.distplot(result_df[result_df['true']==1]['recon_loss'], color = 'red', hist=True, kde=False, rug=False, label = 'Aggressive')

    plt.legend(title = 'Type')
    plt.title(f'epoch {args.epoch} - {args.sess}')
    plt.xlabel('Recon Error')

    save_path = 'D:/Google Drive/Mine/graduate_research/code/video_anomaly/pig_anomaly/log/hist'
    dst_path = os.path.join(save_path, args.exp)

    if not osp.exists(dst_path):
        os.mkdir(dst_path)

    plt.savefig(osp.join(dst_path, f'epoch_{args.epoch}_{args.sess}'))
    
    plt.close()
    
    
    
    
def cal_metric(true_label_tot, pred_label_tot, log=None):
    cnf_matrix = confusion_matrix(true_label_tot, pred_label_tot)
    
    if log:
        log("\n        pred 0  pred 1")
        log(f"true 0 [ {cnf_matrix[0][0]} , {cnf_matrix[0][1]} ]")
        log(f"true 1 [ {cnf_matrix[1][0]} , {cnf_matrix[1][1]} ]\n")
    
    # print(cnf_matrix)
    #[[1 1 3]
    # [3 2 2]
    # [1 3 1]]

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

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
    
    dict_ = {'recall': TPR, 'precision': PPV, 'fpr': FPR, 'fnr': FNR}
    return dict_
        
