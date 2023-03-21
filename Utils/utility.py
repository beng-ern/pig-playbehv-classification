import torch
import torch.nn as nn
import torch.optim as optim

from Models import STTAE
import configs




def create_models(base_model='VGG19', device=None, args=None):

    models = {}   
    
    # Spatial Encoder
    models['Spatial_Encoder'] = STTAE.Encoder(base_model=base_model, frame_diff=args.fd, enc_dim=args.dim)
    # Temporal Encoder_Decoder
    models['Temp_EncDec'] = STTAE.VariationalRAE(configs.Configs, batch_size=args.batch_size, device=device, dec_dim=args.dim)


    for key in models:
        models[key].to(device)

    return models


def set_optimizers(models, args):
    optimizers = {}
    lr_schedulers = {}
    
    # Spatial Encoder
    optimizers['Spatial_Encoder'] = optim.SGD(models["Spatial_Encoder"].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Temporal Encoder_Decoder
    optimizers['Temp_EncDec'] = optim.SGD(models["Temp_EncDec"].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    lr_schedulers['Spatial_Encoder'] = optim.lr_scheduler.ReduceLROnPlateau(optimizers['Spatial_Encoder'], patience = 5)
    lr_schedulers['Temp_EncDec'] = optim.lr_scheduler.ReduceLROnPlateau(optimizers['Temp_EncDec'], patience = 5)
    
#     # Spatial Encoder
#     optimizers['Spatial_Encoder'] = optim.Adam(models['Spatial_Encoder'].parameters(), lr=args.lr)
    
#     # Temporal Encoder_Decoder
#     optimizers['Temp_EncDec'] = optim.Adam(models['Temp_EncDec'].parameters(), lr=args.lr)
    
#     lr_schedulers['Spatial_Encoder'] = optim.lr_scheduler.ExponentialLR(optimizers['Spatial_Encoder'], gamma=0.95)
#     lr_schedulers['Temp_EncDec'] = optim.lr_scheduler.ExponentialLR(optimizers['Temp_EncDec'], gamma=0.95)
    
    return optimizers, lr_schedulers

