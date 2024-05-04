from torch.utils.data import Dataset, DataLoader

class PklDataLoader(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        xyz = self.data[index,0:3]
        pose = self.data[index,3:6]
        rgb = self.data[index,6:]
        return xyz, pose, rgb