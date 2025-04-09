import torch
import torchvision
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from Model.Unet import Unet
from Noise_Scheduler.noise_scheduler import NoiseScheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, num_images):
    os.makedirs('samples', exist_ok=True)

    for n in range(num_images):
        xt = torch.randn((1, 3, 28, 28)).to(device)
        for i in tqdm(reversed(range(1000)), desc=f"Sampling image {n+1}/{num_images}"):
            noise_pred = model(xt, torch.tensor([i]).to(device))
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.tensor(i).to(device))

        # Process and save the final x0_pred
        im = torch.clamp(x0_pred, -1., 1.).detach().cpu()
        im = (im + 1) / 2  # scale to [0, 1]
        im = im.squeeze(0)  # remove batch dimension
        img = torchvision.transforms.ToPILImage()(im)
        img.save(os.path.join('samples', f'sample_{n+1}.png'))
        img.close()





    
model = Unet().to(device)
model.load_state_dict(torch.load(os.path.join('train_result',"ddpm.ckpt"), map_location=device))
model.eval()
scheduler = NoiseScheduler()
with torch.no_grad():
    sample(model, scheduler,10)


