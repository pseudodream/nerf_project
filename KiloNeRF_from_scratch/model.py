import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init

class KiloNeRF(nn.Module):
    def __init__(self, N = 16, pos_encoding_loc = 10, pos_encoding_dir = 4, scale = 3.0, DEVICE='cuda'):
        super(KiloNeRF, self).__init__()
        self.N = N
        self.scale = scale
        self.pos_encoding_loc = pos_encoding_loc
        self.pos_encoding_dir = pos_encoding_dir
        self.DEVICE = DEVICE

        self.weight1 = nn.Parameter(torch.zeros(N, N, N, pos_encoding_loc * 6 + 3, 32))
        self.bias1 = nn.Parameter(torch.zeros(N, N, N, 1, 32))
        self.weight2 = nn.Parameter(torch.zeros(N, N, N, 32, 33))
        self.bias2 = nn.Parameter(torch.zeros(N, N, N, 1, 33))
        self.weight3 = nn.Parameter(torch.zeros(N, N, N, 32, 32))
        self.bias3 = nn.Parameter(torch.zeros(N, N, N, 1, 32))

        self.weight4 = nn.Parameter(torch.zeros(N, N, N, 32 + pos_encoding_dir * 6 + 3, 32))
        self.bias4 = nn.Parameter(torch.zeros(N, N, N, 1, 32))

        self.weight5 = nn.Parameter(torch.zeros(N, N, N, 32, 3))
        self.bias5 = nn.Parameter(torch.zeros(N, N, N, 1, 3))

        # Parameter initialization
        init.kaiming_uniform_(self.weight1, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.weight2, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.weight4, mode='fan_in', nonlinearity='relu')

        init.xavier_uniform_(self.weight3)
        init.xavier_uniform_(self.weight5)

        init.constant_(self.bias1, 1e-2)
        init.constant_(self.bias2, 1e-2)
        init.constant_(self.bias3, 1e-2)
        init.constant_(self.bias4, 1e-2)
        init.constant_(self.bias5, 1e-2)

    def position_encoding(self, x, L):
        output = [x]
        for j in range(L):
            output.append(torch.sin(2 ** j * x))
            output.append(torch.cos(2 ** j * x))
        return torch.cat(output, dim=1).unsqueeze(1)

    def forward(self, x, d):
        color = torch.zeros_like(x) # [batch_size, 3]
        density = torch.zeros((x.shape[0],1)).to(self.DEVICE) # [batch_size, 1]
        mask = (x[:, 0].abs() < (self.scale / 2)) & (x[:, 1].abs() < (self.scale / 2)) & (
                x[:, 2].abs() < (self.scale / 2)) # [batch_size]
        
        embbed_x = self.position_encoding(x[mask], self.pos_encoding_loc) # [masked_batch_size, 1, 63]
        embbed_d = self.position_encoding(d[mask], self.pos_encoding_dir) # [masked_batch_size, 1, 27]
        
        # idx
        i = ((x[mask] / (self.scale / self.N)) + self.N / 2).to(torch.long) # [masked_batch_size, 3]
        i = torch.clamp(i, 0, self.N-1) # [masked_batch_size, 3]

        output = torch.bmm(embbed_x, self.weight1[i[:,0], i[:,1], i[:,2]]) + self.bias1[i[:,0], i[:,1], i[:,2]] # 5098, 1, 32
        output = torch.relu(output) # 5098, 1, 32

        output = torch.bmm(output, self.weight2[i[:,0], i[:,1], i[:,2]]) + self.bias2[i[:,0], i[:,1], i[:,2]] # 5098, 1, 33
        output = torch.relu(output) # 5098, 1, 33
        
        sigma = torch.relu(output.squeeze(1))[:,0].unsqueeze(1) # 5098, 1, 33
        output = output[:,:,1:]
        assert sigma.shape == (i.shape[0], 1)

        output = torch.bmm(output, self.weight3[i[:,0], i[:,1], i[:,2]]) + self.bias3[i[:,0], i[:,1], i[:,2]] # 5098, 5098, 32
        output = torch.cat((output, embbed_d), dim=2)

        output = torch.bmm(output, self.weight4[i[:,0], i[:,1], i[:,2]]) + self.bias4[i[:,0], i[:,1], i[:,2]]
        output = torch.relu(output)

        output = torch.bmm(output, self.weight5[i[:,0], i[:,1], i[:,2]]) + self.bias5[i[:,0], i[:,1], i[:,2]]
        output = torch.sigmoid(output)
        c = output.squeeze(1)
        assert c.shape == (i.shape[0], 3)

        density[mask] = sigma
        color[mask] = c

        return color, density