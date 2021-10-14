import os
from pathlib import Path

from scipy import io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

class LDCDataset(Data.Dataset):
    def __init__(self, folderPath=None,langFolder='aeng', 
                 dataAttribute=None, is_test=False, task=None, classNum=0):
        super(LDCDataset, self).__init__()
        
        self.folderPath = folderPath 
        self.classNum = classNum
        self.dataAttribute = dataAttribute   
        self.lang = langFolder    
            
        if is_test:
            if task is None:
                print("Error: Please give your task folder name!")
                return
            self.langFolderPath = os.path.join(self.folderPath, self.lang, task)
        else:
            self.langFolderPath = os.path.join(self.folderPath, self.lang)
        
        self.audio_paths = list(Path(self.langFolderPath).glob("**/*.mat"))
        print("lang path is:", self.langFolderPath, " size:", len(self.audio_paths), "class: ", self.classNum)       
        
    def __len__(self):
        return len(self.audio_paths)
        
    def __getitem__(self, index):
        file_path = self.audio_paths[index]
        fileObj = sio.loadmat(file_path, verify_compressed_data_integrity=False)
        fileFeats = fileObj[self.dataAttribute][:, :-1] # Exluding the class label for each frame

        return torch.from_numpy(fileFeats), self.classNum