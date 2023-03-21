import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
    

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()

        self.attn_1 = nn.Linear(feature_dim, feature_dim)
        self.attn_2 = nn.Linear(feature_dim, 1)

        # inititalize
        nn.init.xavier_uniform_(self.attn_1.weight)
        nn.init.xavier_uniform_(self.attn_2.weight)
        self.attn_1.bias.data.fill_(0.0)
        self.attn_2.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Input x is encoder output
        return_attention decides whether to return
        attention scores over the hidden states
        """
#         frame_length = x.shape[1]

        self_attention_scores = self.attn_2(torch.tanh(self.attn_1(x)))
        
        return self_attention_scores
    

    
# need to change this class if want to load the trained model with attention module
class Resnt34LSTM(nn.Module):
    def __init__(self):
        super(Resnt34LSTM, self).__init__()
        num_classes = 2
        pretrained = True
        self.rnn_hidden_size = 60
        rnn_num_layers = 1
        
        baseModel = models.resnet34(pretrained=pretrained)  
        num_features = baseModel.fc.in_features
        baseModel.avgpool = nn.AdaptiveMaxPool2d(output_size=(1,1))
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.attention = Attention(self.rnn_hidden_size)
        self.rnn = nn.LSTM(num_features, self.rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(self.rnn_hidden_size*20, num_classes)

        
    def forward(self, x):        
        b_z, ts, c, h, w = x.shape
        
        
        ii = 0
        y = self.baseModel((x[:,ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(0))
        for ii in range(1, ts):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(0), (hn, cn))
            output = torch.cat((output.view(b_z,-1,self.rnn_hidden_size), out.view(b_z,-1,self.rnn_hidden_size)), dim=1)

        out_att_score = self.attention(output)
        out_att_score_sftmx = F.softmax(out_att_score, dim=1)
        out_agg_att = out_att_score_sftmx * output
        final_out = self.fc1(out_agg_att.view(b_z, -1))
        
        
#         final_out = self.fc1(output.view(b_z,-1)) 
        
        return final_out 
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x    


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "hirescam":HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}

#     model = models.resnet34(pretrained=True)
    model = Resnt34LSTM()
#     checkpoint = torch.load("D:/0_Graduation Thesis/models/weights/Resnet34_LSTM60_MAXpool_WS_OS_allHS_20220530_best_accuracy.pt")
    checkpoint = torch.load("D:/0_Graduation Thesis/models/weights/Resnet34_LSTM60_MAXpool_WSOS_allattendedHS_20220604_best_accuracy.pt")
    model.load_state_dict(checkpoint)
    model = model.baseModel #only CNN backbone
    
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
#     target_layers = [model.layer4]
    target_layers = [model.layer4]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    cv2.imwrite(f'{args.method}_gb.jpg', gb)
    cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)