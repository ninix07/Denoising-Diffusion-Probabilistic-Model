import torch
from torch.optim import Adam
from Noise_Scheduler.noise_scheduler import NoiseScheduler
from DatasetLoader.DatasetLoader import MnistDataset, CIFARDataset
from torch.utils.data import DataLoader
from Model.Unet import Unet
from tqdm import tqdm
import numpy as np
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train():
    scheduler= NoiseScheduler()
    dataset= CIFARDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    model = Unet().to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists('train_result'):
        os.mkdir('train_result')
    
    # Load checkpoint if found
    if os.path.exists(os.path.join('train_result',"ddpm.ckpt")):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join('train_result',"ddpm.ckpt"), map_location=device))
    # Specify training parameters
    num_epochs = 60
    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()
    
    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, 1000, (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.forward(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        torch.save(model.state_dict(), os.path.join('train_result',"ddpm.ckpt"))
    
    print('Done Training ...')


if __name__ == "__main__":
    train()