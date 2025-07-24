import torch
import numpy as np
# Contains implementations of diffusion schedulers, 
# responsible for gettting the diffusio sptesp, adding noise to the image, and stepping 
# during the generation process. 

class KarrasScheduler:
    """Karras diffusion scheduler as used in the Consistency Models implementation (https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py)
    Note that we define the number of diffusion steps, not the levels. 
    Args:
        diffusion_steps: Number of diffusion steps during sampling
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        sigma_data: Noise level of the data.
        rho: Exponent for the schedule.
        clip_sample: Whether to clip the samples during sampling.
        num_train_timesteps: Number of diffusion steps during training.
        prediction_type: Type of prediction. Can be "epsilon" or "sigma".
        variance_type: ? Can be "fixed_small", "fixed_large", or "learned".
        device: Device to use.
        dtype: Data type to use.
    """
    def __init__(self, 
            diffusion_steps=1000, 
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=0.5,
            rho=7,
            clip_sample=True,
            num_train_timesteps=100,
            prediction_type="epsilon",
            variance_type="fixed_small",
            device="cuda", 
            dtype=torch.float32, 
            ):
        # Karras noise level parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

        # Training parameters
        self.num_train_timesteps = num_train_timesteps

        # Sampling parameters
        self.num_diff_steps_sampling = diffusion_steps

        # Other parameters
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.variance_type = variance_type
        self.device = device
        self.dtype = dtype

        self.training_sigmas = self.calculate_karras_sigmas(self.num_train_timesteps)

    def calculate_karras_sigmas(self, num_steps):
        """Calculates the sigmas for the Karras diffusion schedule.
        sigm_min = sigma_0 > sigma_1 > ... > sigma_{num_steps - 1} = sigma_max
        """
        # Special case for num_sampling_sigmas = 1, since we don't want to divide by 0
        if num_steps == 1:
            sigmas = torch.tensor([self.sigma_max], device=self.device, dtype=self.dtype)  
        else:
            # Doesn't incldude num_train_timesteps, since we sample t from 0 to num_diff_steps (exclusive)
            # during training.
            indices = torch.arange(num_steps, device=self.device, dtype=self.dtype)
            sigmas = (self.sigma_max ** (1 / self.rho) + indices / (num_steps - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
            ))**self.rho
        # Append 0 to the end
        return torch.cat([sigmas, torch.zeros(1, device=self.device, dtype=self.dtype)])
    
    def get_scaling_cm(self, sigma: torch.Tensor):
        """Returns scaling given a set of step indices
        Args:
            indices: (batch_size,)
        Returns:
            c_skip: (batch_size,)
            c_out: 
        """
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
    
    def step(self, x, index, generator):
        """Adds noise during Consistency Sampling
        Args:
            x: (batch_size, pred_horizon, action_dim)
            index: (batch_size,)
            generator: Random number generator.
        Returns:     
            x: (batch_size, pred_horizon, action_dim)
        """

        sigma = self.sampling_sigmas[index]
        z = torch.randn(x.shape,
                        dtype=x.dtype,
                        device=x.device,
                        generator=generator)
        return x + torch.sqrt(sigma**2 - self.sigma_min**2) * z

    def get_sigmas_sampling(self, indices: torch.Tensor):
        """Returns sigmas given a set of step indices
        Args:
            indices: (batch_size,)
        Returns:
            sigma: (batch_size,)
        """
        return self.sampling_sigmas[indices]
    
    def set_num_diffusion_steps_sampling(self, num_diffusion_steps_sampling):
        """Sets the number of diffusion steps for sampling."""
        self.num_diff_steps_sampling = num_diffusion_steps_sampling
        self.sampling_sigmas = self.calculate_karras_sigmas(self.num_diff_steps_sampling)


    def add_noise(self, traj, noise, indices):
        """Adds noise to the image during training.
        Args:
            traj: (batch_size, pred_horizon, action_dim)
            noise: (batch_size, pred_horizon, action_dim)
            indices: (batch_size,)
        Returns:
            traj: (batch_size, pred_horizon, action_dim)
        """
        return traj + noise * self.training_sigmas[indices].view(-1, 1, 1)
    

class EDMScheduler:
    """Karras diffusion scheduler as used in EDM
    Args:
    diffusion_steps: Number of diffusion steps during sampling
    sigma_min: Minimum noise level.
    sigma_max: Maximum noise level.
    sigma_data: Noise level of the data.
    rho: Exponent for the schedule.
    clip_sample: Whether to clip the samples during sampling.
    num_train_timesteps: Number of diffusion steps during training.
    prediction_type: Type of prediction. Can be "epsilon" or "sigma".
    variance_type: ? Can be "fixed_small", "fixed_large", or "learned".
    device: Device to use.
    dtype: Data type to use.
    """
    def __init__(self, 
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=0.5,
            rho=7,
            P_mean=-1.2, # From EDM paper
            P_std=1.2, # From EDM paper
            # These should be tuned through grid search, setting arbitrarily for now
            S_churn = 60,
            S_tmin = 0.03, 
            S_tmax = 25,
            S_noise = 1.005,
            clip_sample=True,
            prediction_type="epsilon",
            variance_type="fixed_small",
            device="cuda", 
            dtype=torch.float32, 
            ):
        # Karras noise level parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

        # Training parameters
        self.P_mean = P_mean
        self.P_std = P_std

        # Sampling parameters
        
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        # Other parameters
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.variance_type = variance_type
        self.device = device
        self.dtype = dtype

    def get_scaling_edm(self, sigma: torch.Tensor):
        """Returns scaling given a set of noise levels
        Args:
            sigmas: (B,) or (B, T)
        Returns:
            c_skip: same shape as sigmas
            c_out: same shape as sigmas
            c_in: same shape as sigmas
            c_noise: same shape as sigmas
        """
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = torch.log(sigma) / 4
        return c_skip, c_out, c_in, c_noise

    def get_loss_weights(self, sigmas):
        """Returns the loss weights for the EDM loss.
        sigmas: (B,) or (B, T_p)"""
        return (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data)**2

    def get_sigmas_training(self, sigma_shape, device, dtype):
        """Samples batch_size sigmas for EDM training from a log-normal distribution
        with mean P_mean and std. dev. P_std.
        Args:
            sigma_shape: Shape of the sigmas to sample (int or tuple)
            device: Device to use
            dtype: Data type to use
        Returns:
            sigmas: (sigma_shape,) Noise levels sampled from the log-normal distribution.
        """
        return torch.exp(torch.randn(sigma_shape, device=device, dtype=dtype) 
                                * self.P_std 
                                + self.P_mean)
    
    def get_sigmas_training_constant(self, shape, device, dtype):
        """
        Args:
            shape: Shape of the sigmas to sample (int or tuple). Typically (B, T) or (T,)
        Returns:
            sigmas: (B, T) Noise levels, which are constant for each trajectory in the batch."""
        if len(shape) == 1:
            T = shape
            return self.get_sigmas_training((1,), device, dtype).repeat(T)
        else:
            B, T = shape
            return self.get_sigmas_training((B, 1), device, dtype).repeat(1, T)
        

    def get_sigmas_training_linear_increase(self, shape, device, dtype):
        """
        Args:
            shape: typically (T,) or (B, T)
        Rerturns:
            sigmas (B, T) Noise levels, which rise linearly from sigma_min to sigma_max"""
        if len(shape) == 1:
            T = shape[0]
            return torch.linspace(self.sigma_min, self.sigma_max, T, device=device, dtype=dtype)
        else:
            B, T = shape
            return torch.linspace(self.sigma_min, self.sigma_max, T, device=device, dtype=dtype).repeat(B, 1)
    
    def get_sigmas_training_chunkwise(self, shape, To, Ta, num_sigmas, device, dtype):
        """
        Returns sigmas that increases every chunk. The first chunk spans over the first (To+Ta-1) elements,
        and the next is 
        Args:
            shape: typically (T,) or (B, T)
            To: number of steps of the first chunk (int)
            Ta: number of steps of the second chunk (int)
            num_sigmas: number of discretizations over the sigma-axis (int)
        Returns:
            sigmas: (B, T) Noise levels, which are constant for each element in the batch."""
        if len(shape) == 1:
            T = shape[0]
        else:
            B, T = shape
        
        indices = torch.arange(0,T, device=device)
        sequence_index = torch.div((indices - (To-1)), Ta, rounding_mode='floor')
        sequence_index = torch.where(indices < To, 0, sequence_index)
        fraction = Ta / (T-To+1) 
        sigma_indices = torch.round(num_sigmas * (1 - sequence_index * fraction)).long() - 1

        if len(shape) == 1:
            return self.get_sigmas_sampling(num_sigmas)[sigma_indices]
        else:
            return self.get_sigmas_sampling(num_sigmas)[sigma_indices].repeat(B, 1)

    
    def get_sigmas_sampling(self, num_sampling_sigmas):
        """Gets sigmas for EDM sampling with the discretization scheme described in the paper. Note that the last sigma is 0.
        sigma_max = sigma_0 > ... > sigma_(N-1) = sigma_min > sigma_N = 0
        Args:
            num_sampling_sigmas: Number of sigmas to sample. Same as N in the paper. (int)
        Returns:
            sigmas: (num_sampling_sigmas + 1,)"""
        
        # Special case for num_sampling_sigmas = 1, since we don't want to divide by 0
        if num_sampling_sigmas == 1:
            sigmas = torch.tensor([self.sigma_max], device=self.device, dtype=self.dtype)  
        else:
            indices = torch.arange(num_sampling_sigmas, device=self.device, dtype=self.dtype)
            sigmas = (self.sigma_max ** (1 / self.rho) + indices / (num_sampling_sigmas - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
            ))**self.rho
        # Append 0 to the end
        return torch.cat([sigmas, torch.zeros(1, device=self.device, dtype=self.dtype)])

    def get_gamma(self, sigma, num_sigmas_sampling):
        """Calculates gamma for EDM sampling
        Args:
            sigma: A tensor of shape (1,) or (batch_size,)
        Returns:
            gamma: A tensor of shape corresponding to sigma's shape"""
        
        # Calculate the constant value for comparison
        constant_value = min(self.S_churn / num_sigmas_sampling, np.sqrt(2) - 1)
        
        # Element-wise checks for sigma within range
        in_range = (sigma >= self.S_tmin) & (sigma <= self.S_tmax)
        
        # Calculate gamma for each element
        gamma_values = torch.where(
            in_range,
            torch.tensor(constant_value, device=self.device, dtype=self.dtype),
            torch.tensor(0, device=self.device, dtype=self.dtype)
        )
        
        return gamma_values


