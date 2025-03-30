import torch

class NoiseScheduler:
    def __init__(self,timestep=1000,beta_start=10e-4,beta_end=0.02):
        self.timestep= timestep #markov chain size
        self.beta_start=beta_start
        self.beta_end=beta_end

        self.betas= torch.linspace(beta_start,beta_end,timestep)
        self.alphas=1-self.betas
        self.cumulative_product= torch.cumprod(self.alphas,dim=0)
        self.sqrt_cumulative_product= torch.sqrt(self.cumulative_product)
        self.sqrt_one_minus_cumulative_product= torch.sqrt(1-self.cumulative_product)
    
    
    def forward(self, image, noise, t):
        ''''
        Forward noise prediction using:
        x_t= sqrt_cum_alpha * x_0 + sqrt_one_minus_cum_alpha * noise
        i.e add noise to the given image according to the timestep.
        '''
        #assume image is batch of image
        image_shape= image.shape 
        batch_size= image_shape[0]
        # reshape the 't'  noise scaling factor to match the batch size
        sqrt_cum_prod= self.sqrt_cumulative_product[t].reshape(batch_size)
        sqrt_one_minus_cum_prod=self.sqrt_one_minus_cumulative_product[t].reshape(batch_size)
        #extend the shape of noise scaling factor to [batch_size,1,1,1]
        for _ in range(len(image_shape)-1):
            sqrt_cum_prod=sqrt_cum_prod.unsqueeze(-1)
            sqrt_one_minus_cum_prod=sqrt_one_minus_cum_prod.unsqueeze(-1)
        return sqrt_cum_prod* image + sqrt_one_minus_cum_prod*noise

    def sample_prev_timestep(self,x_t, noise_pred, t ):
        ''''
        Reverse process to sample x_t-1 from x_t for reconstructing clean image/ noisy image as per necessity.
        '''
        #compute x_0 using the same forward equation by rearranging terms
        x_0 = (x_t - self.sqrt_one_minus_cumulative_product[t]*noise_pred )/self.sqrt_cumulative_product[t]
        x_0 = torch.clamp(x_0, -1,1)
        #for sampling compute mean 
        mean= x_t- ((self.betas[t] *noise_pred)/self.sqrt_one_minus_cumulative_product[t])
        mean= mean/ torch.sqrt(self.alphas[t])

        if t==0:
            return mean, x_0
        else: 
            varaince = (1-self.alpha_cum_prod[t-1])/(1-self.alpha_cum_prod[t])
            varaince= varaince *self.betas[t]
            sigma = torch.sqrt(varaince)
            z=torch.randn(x_t.shape).to(x_t.device)
            return mean +sigma*z, x_0 #: Add noise to the mean, ensuring stochasticity.
