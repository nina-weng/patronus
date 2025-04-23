import torch

class SimpleDiffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        img_shape=(3, 64, 64),
        device="cpu",
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device

        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta  = self.get_betas()
        self.alpha = 1 - self.beta
        
        self.sqrt_beta                       = torch.sqrt(self.beta)
        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)


        # TODO: there are repeated ones
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)  
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.zeros(1, device=self.device)])
        assert self.alphas_cumprod_prev.shape == (self.num_diffusion_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
        
    def forward_diffusion(self, x0: torch.Tensor, timesteps: torch.Tensor):
        eps = torch.randn_like(x0)  # Noise
        mean    = get(self.sqrt_alpha_cumulative, t=timesteps) * x0  # Image scaled
        std_dev = get(self.sqrt_one_minus_alpha_cumulative, t=timesteps) # Noise scaled
        sample  = mean + std_dev * eps # scaled inputs * scaled noise

        return sample, eps  # return ... , gt noise --> model predicts this)
    
    def model_err_pred_to_mean(self,
                               err: torch.Tensor, 
                               x_t: torch.Tensor, 
                               t: torch.Tensor) -> torch.Tensor:
        """
        Used to calculate the estimate for the mean in the reverse process using the predicted noise

        # formula 13 in improved DDPM paper
        mu_theta(x_t,t) = (1/sqrt(alpha_t) )* (x_t - (beta_t/sqrt(1-alpha_bar_t)) * err_theta(x_t,t))
        
        # formula 9 in improved DDPM paper
        x_0 = (x_t - sqrt(1-alpha_bar_t) * err) / sqrt(alpha_bar_t)


        Return:
        - mu_theta_xt_t:  mu_theta(x_t,t) in formula 13, the estismated mean of x_t given t and predicted err
        - x_0_blur: x_0 in formula 9, as it is reconstrated from the reversed noise implementation, 
                    the image will be a bit blur
        """
        c1 = get(self.one_by_sqrt_alpha, t)
        noise_coef = self.beta / self.sqrt_one_minus_alpha_cumulative
        c2 = get(noise_coef, t)

        mu_theta_xt_t = c1 * (x_t - c2 * err)  # mu_theta_xt_t.shape = [BS, C, H, W]
        x_0_blur = (x_t-err*get(self.sqrt_one_minus_alpha_cumulative,t))/get(self.sqrt_alpha_cumulative,t) # x_0_blur.shape = [BS, C, H, W]

        return mu_theta_xt_t, x_0_blur
        
    def p_mean_std(self,
                   model, 
                   x_t: torch.Tensor, 
                   t: torch.Tensor, 
                   model_kwargs=None,
                   ) -> dict[str, torch.Tensor]:
        """
        Calculate mean and std of p(x_{t-1} | x_t) using the reverse process and model
        Parameter:
        - x_t: shape = [BS, C, H, W]
        - t 
        

        Return:
        - out
            - out['mean']: shape = [BS, C, H, W]
            - out['std']: shape = [BS, 1,1,1] #for simple scheduled std
            - out['denoised']: shape = [BS, C, H, W]
        """
        out = dict()

        model_out = model(x_t,t,**model_kwargs) # get the noise ftom unet
        # model_out.shape = [BS, C, H, W]

        err = model_out

        # scheduled std
        assert len(x_t.shape) == 4
        out['std'] = get(self.sqrt_beta, t).repeat(x_t.shape[0],1,1,1)
        out['mean'], out['pred_xstart'] = self.model_err_pred_to_mean(err, x_t, t)
        return out
    
    def reverse_sample(
        self,
        model,
        x,
        t,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        NOTE: never used ? 
        """
        # print(type(eta))
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_std(
            model,
            x,
            t,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (_extract_into_tensor_from_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape)
               * x - out["pred_xstart"]) / _extract_into_tensor_from_tensor(
                   self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor_from_tensor(self.alphas_cumprod_next, t,
                                              x.shape)

        # Equation 12. reversed  (DDIM paper)  (th.sqrt == torch.sqrt)
        mean_pred = (out["pred_xstart"] * torch.sqrt(alpha_bar_next) +
                     torch.sqrt(1 - alpha_bar_next) * eps)

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}
    
    def reverse_sample_loop(
            self,
            model,
            x,
            model_kwargs=None,
            eta=0.0,
        ):
            
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(self.num_diffusion_timesteps))
        sample = x
        for i in indices:
            t = torch.tensor([i] * len(sample), device=self.device)
            with torch.no_grad():
                out = self.reverse_sample(model,
                                            sample,
                                            t=t,
                                            model_kwargs=model_kwargs,
                                            eta=eta)
                sample = out['sample']
                # [1, ..., T]
                sample_t.append(sample)
                # [0, ...., T-1]
                xstart_t.append(out['pred_xstart'])
                # [0, ..., T-1] ready to use
                T.append(t)

        return {
            #  xT "
            'sample': sample,
            # (1, ..., T)
            'sample_t': sample_t,
            # xstart here is a bit different from sampling from T = T-1 to T = 0
            # may not be exact
            'xstart_t': xstart_t,
            'T': T,
        }
    
    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_std(
            model,
            x,
            t,
            model_kwargs=model_kwargs,
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor_from_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor_from_tensor(self.alphas_cumprod_prev, t,
                                              x.shape)
        sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
                 torch.sqrt(1 - alpha_bar / alpha_bar_prev))
        
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (out["pred_xstart"] * torch.sqrt(alpha_bar_prev) +
                     torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def ddim_sample_loop_progressive(
        self,
        model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        num_samples=1,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
            if num_samples != 1 and noise.shape[0] == 1: # input is a single instance but we want to generate multiple samples
                img = img.repeat(num_samples,1,1,1)
        else:
            assert isinstance(shape, (tuple, list))
            
            if num_samples != 1:
                assert shape[0] == 1
                shape = [num_samples] + list(shape[1:])
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_diffusion_timesteps))[::-1]

        print(f'DEBUG:{img.shape=}')

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:

            if isinstance(model_kwargs, list):
                # index dependent model kwargs
                # (T-1, ..., 0)
                _kwargs = model_kwargs[i]
            else:
                _kwargs = model_kwargs

            t = torch.tensor([i] * len(img), device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=_kwargs,
                    eta=eta,
                )
                out['t'] = t
                yield out
                img = out["sample"]
    
    def ddim_sample_loop(
        self,
        model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        num_samples=1,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
                num_samples=num_samples,
        ):
            final = sample
        return final["sample"]
    

    def sample(self,
               model,
               shape=None,
               noise=None,
               cond=None,
               x_start=None,
               clip_denoised=True,
               model_kwargs=None,
               progress=False,
               num_samples=1,
               eta=0.0):
        """
        Args:
            x_start: given for the autoencoder
        """
        

        return self.ddim_sample_loop(model,
                                         shape=shape,
                                         noise=noise,
                                         clip_denoised=clip_denoised,
                                         model_kwargs=model_kwargs,
                                         progress=progress,
                                         num_samples=num_samples,
                                         eta=eta)
    

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (_extract_into_tensor_from_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                pred_xstart) / _extract_into_tensor_from_tensor(
                    self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    




def _extract_into_tensor_from_tensor(tensor, timesteps, broadcast_shape):
    """
    Extract values from a 1-D tensor for a batch of indices.

    :param tensor: A 1-D PyTorch tensor.
    :param timesteps: A tensor of indices into the tensor to extract.
    :param broadcast_shape: A target shape with the batch dimension 
                            matching the length of timesteps.
    :return: A tensor of shape broadcast_shape where the extracted 
             values are broadcasted to match.
    """
    # Ensure tensor and timesteps are on the same device
    tensor = tensor.to(device=timesteps.device)
    
    # Gather values based on timesteps
    res = tensor[timesteps].float()
    
    # Add singleton dimensions to match broadcast_shape
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    
    # Broadcast to the desired shape
    return res.expand(broadcast_shape)



def get(element: torch.Tensor, t: torch.Tensor):
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, t)
    return ele.reshape(-1, 1, 1, 1)