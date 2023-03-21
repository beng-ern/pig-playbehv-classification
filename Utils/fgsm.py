import torch
import torch.nn as nn
from torch.autograd import Variable

from Utils import do_epoch


def attack(data, epsilon, grad, args):
    if args.fgsm_type=='video':
        sign_grad = grad.sign()
        perturbed_data = data + epsilon * sign_grad

        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        return perturbed_data
    
    elif args.fgsm_type=='feature':
        sign_grad = grad.sign()
        perturbed_data = data + epsilon * sign_grad

        return perturbed_data       

        
def FGSM_Attack(models, video, labels, criterion, feature_collector, device, args):
    
    # if args.fgsm_type=='video':
        
    #     normal_idx = labels==0
    #     normal_video = video[normal_idx]
    #     grad = video.grad.data[normal_idx]

    #     label = labels[normal_idx]
    #     label = torch.ones_like(label)  # to be treated as anomaly


    #     new_anomaly = torch.empty_like(normal_video)

    #     # attack -> create new anomaly
    #     for i, (data, grad_) in enumerate(zip(normal_video, grad)):
    #         perturbed = attack(data, args.eps, grad_, args)
    #         new_anomaly[i] = perturbed    

    #     with torch.cuda.amp.autocast():

    #         x = models['Spatial_Encoder'](new_anomaly.permute(0,2,1,3,4))
    #         x = x.permute(1,0,2)

    #         x_recon = models['Temp_EncDec'](x)
    #         x = x.permute(1,0,2)                                

    #         output = criterion(x_recon, x)

    #         loss, _, _, _, _ = do_epoch.cal_criterion(output, label, device, args)


    #         kl_loss = models['Temp_EncDec'].kl_loss()
    #         loss += kl_loss

    #         return loss
            
    # elif args.fgsm_type=='feature':
        
    normal_idx = labels==0
    normal_video = video[normal_idx]
    grad = video.grad.data[normal_idx]

    label = labels[normal_idx]
    label = torch.ones_like(label)  # to be treated as anomaly


    new_anomaly = torch.empty_like(normal_video)

    # attack -> create new anomaly
    for i, (data, grad_) in enumerate(zip(normal_video, grad)):
        perturbed = attack(data, args.eps, grad_, args)
        new_anomaly[i] = perturbed


    with torch.cuda.amp.autocast():
        new_anomaly = new_anomaly.permute(1,0,2)
        new_anomaly_recon, _ = models['Temp_EncDec'](new_anomaly)
        new_anomaly = new_anomaly.permute(1,0,2)

        output = criterion(new_anomaly_recon, new_anomaly)
        loss, _, _, _, _ = do_epoch.cal_criterion(output, label, device, args)

        kl_loss = models['Temp_EncDec'].kl_loss()
        loss += kl_loss

        return loss