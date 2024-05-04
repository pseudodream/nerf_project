import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import KiloNeRF
from custom_data_loader import PklDataLoader
from utils import render_pixel

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    ## Set training hyperparameters
    setup_seed(18786)
    batch_size = 1024
    num_epoch = 16
    lr = 5e-4
    DEVICE = 'cuda'

    ## Load dataset
    train_dataset = torch.from_numpy(np.load('dataset/training_data.pkl', allow_pickle=True))
    train_dataset = PklDataLoader(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    ## Load model
    model = KiloNeRF(DEVICE=DEVICE).to(DEVICE)
    ## Define the criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)
    ## Training and evaluation
    for epoch in range(num_epoch):
        train_loss = 0.0
        psnr = 0.0
        for batch in tqdm(train_loader):
            xyz, pose, rgb = batch
            xyz, pose, rgb = xyz.to(DEVICE), pose.to(DEVICE), rgb.to(DEVICE)
            optimizer.zero_grad()
            c = render_pixel(model, xyz, pose, DEVICE=DEVICE, bins_num=144)
            loss = criterion(c, rgb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            psnr += (-10. * torch.log(loss).to(DEVICE) / torch.log(torch.Tensor([10.])).to(DEVICE)).item()
        train_loss_avg = train_loss / len(train_loader)
        psnr_avg = psnr / len(train_loader)
        print(f'Epoch {epoch} with training accurency {train_loss_avg}')
        print(f'Epoch {epoch} with PSNR {psnr_avg}')
        scheduler.step()
    torch.save(model.state_dict(), 'model.pth')
        