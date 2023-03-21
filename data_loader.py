import os
import pickle


import torch
import cv2
from torch.utils.data import Dataset
from torchvision.datasets.video_utils import VideoClips
import numpy as np

from ast import literal_eval


# Return the cached VideoCilp objecet (pickle)
class VideoDataset(Dataset):
    
    def __init__(self, video_clip_path, dataframe, frame_diff=None, video_transform=None):
        self.video_clip_path = video_clip_path
        self.df = dataframe
        self.video_transform = video_transform
        self.frame_diff = frame_diff
        
        with open(str(self.video_clip_path), 'rb') as f:
            self.video_clips = pickle.load(f)
        
    def getitem_from_raw_video(self, index):
        video, _, _, _ = self.video_clips.get_clip(index)
        video_idx, clip_idx = self.video_clips.get_clip_location(index)
        video_path = self.video_clips.video_paths[video_idx]
        label = self._get_label(video_path)
#         centroid_y_std, area_std = self.get_std(video_path)
#         coord_seq = self.get_coordinates_seq(video_path)
#         pen_id = self.get_penID(video_path)
        
        
        if self.video_transform:            
            video = video.numpy()
            video = self.video_transform(video)
            video = video.permute(1,0,2,3) #to switch the index of number of frames and number of channels
            
            
        return video, label
#         return video, label, video_path, centroid_y_std, area_std
    
    def _get_label(self, video_path):
        label = self.df[self.df['video_path']==video_path]['label'].unique().item()
        
        if (label == 0) or (label ==0.):
            return 0
        else:
            return 1
    
    def get_std(self, video_path):
#         centroid_x_std = self.df[self.df['video_path']==video_path]['centroid_x_std']
        centroid_y_std = self.df[self.df['video_path']==video_path]['centroid_y_std'].item()
        area_std = self.df[self.df['video_path']==video_path]['area_std'].item()
#         centroid_std = np.array([centroid_x_std, centroid_y_std], dtype='float32')
        return centroid_y_std, area_std
    
    def get_coordinates_seq(self, video_path):
        coord_seq = self.df[self.df['video_path']==video_path]['coord_seq'].apply(literal_eval).item()
        coord_seq_arr = np.array(coord_seq, dtype='float32')
        return coord_seq_arr
    
    def get_penID(self, video_path):
        penID = self.df[self.df['video_path']==video_path]['pen_ID'].item()
        
        # one-hot encode the pigpen ID
        if (penID == 3) or (penID ==3.):
            return np.array([0,1])
        else:
            return np.array([1,0])    
            
    
    # added this for using 'torchsampler.ImbalancedDataSampler'
    def get_labels(self):
        return self.df['label'].tolist()
    
    def __getitem__(self, index):
        video, label  = self.getitem_from_raw_video(index)
#         return video, label, video_path  # video: (N, T, H, W, C) --not sure why is this comment here in Hyunsoo's code
        return video, label

    def __len__(self):
        return len(self.df)

    