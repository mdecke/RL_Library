import os
from typing import Dict
import torch
import torch.nn as nn

from tqdm import tqdm
from skrl.resources.preprocessors.torch import RunningStandardScaler
from base_classes.models import MLE, GMM
from base_classes.utils import (gaussian_nll_loss, init_model_weights, 
                                print_model_summary, EarlyStopping,
                                gmm_nll_loss, get_activation)


class MLEExpert:
    def __init__(self, cfg:Dict):
        self.obs_dim = cfg['obs_dim']
        self.action_dim = cfg['action_dim']

        self.cfg = cfg
        self.expert_domain = cfg['expert_domain']
        self.device = cfg['device']
        self.validation_interval = cfg.get("validation_interval", 1)
        self.patience = cfg.get("early_stopping_patience", 10)
        self.min_delta = cfg.get("min_delta", 1e-4)
        self.grad_clipping = cfg.get("grad_clipping", None)

        self.preprocess_inputs = cfg.get("preprocess_inputs", True)
        if self.preprocess_inputs:
            self.obs_preprocessor = RunningStandardScaler(size=self.obs_dim, device=self.device)


        self.init_expert()

    def init_expert(self):
        self.lr = self.cfg["mle"]["learning_rate"]
        self.act_fct = self.cfg["mle"]["activation_fct"]
        self.hidden_sizes = self.cfg["mle"]["hidden_sizes"]
        self.dropout = self.cfg["mle"].get("dropout", 0.0)
        self.weight_decay = self.cfg["mle"].get("weight_decay", 0.0)

        self.model = MLE(self.obs_dim, self.action_dim, 1.0, self.hidden_sizes, 
                        self.lr, self.act_fct, dropout=self.dropout, 
                        weight_decay=self.weight_decay).to(self.device)
        init_model_weights(self.model)
        print("[INFO]: Models initialized")
        print("[INFO]: MLE Expert summary:")
        print_model_summary(self.model, input_size=(1, self.obs_dim))
      
    def train(self, train_data:torch.utils.data.DataLoader, val_data:torch.utils.data.DataLoader):
        self.model.train()
        num_epochs = self.cfg["mle"]["n_epochs"]

        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta, verbose=True)

        train_losses = []
        val_losses = []

        epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
        for epoch in epoch_pbar:
            epoch_loss = 0.0
            for obss, acts in train_data:
                if self.preprocess_inputs:
                    obss = self.obs_preprocessor(obss).to(self.device)
                else:
                    obss = obss.to(self.device)
                acts = acts.to(self.device)

                self.model.optimizer.zero_grad()
                mu, log_sigma = self.model(obss)
                batch_loss = gaussian_nll_loss(mu, log_sigma, acts)
                if self.grad_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
                batch_loss.backward()
                
                self.model.optimizer.step()

                epoch_loss += batch_loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_data)
            train_losses.append(avg_epoch_loss)
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_epoch_loss:.4f}',
                'lr': f'{self.model.optimizer.param_groups[0]["lr"]:.6f}',
            })

            if (epoch + 1) % self.cfg["val_interval"] == 0:
                val_loss = self.validate(val_data)
                val_losses.append(val_loss)
                # print(f"Validation Loss after Epoch {epoch+1}: {val_loss:.4f}")
                epoch_pbar.set_postfix({
                'train_loss': f'{avg_epoch_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'lr': f'{self.model.optimizer.param_groups[0]["lr"]:.6f}'
            })
                
                # Step scheduler based on validation loss
                self.model.scheduler.step(val_loss)
                
                if early_stopping(val_loss, self.model):
                    # print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
                    # print(f"[INFO] Best validation loss: {early_stopping.best_loss:.6f}")
                    epoch_pbar.write(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
                    epoch_pbar.write(f"[INFO] Best validation loss: {early_stopping.best_loss:.6f}")
                    break
        
        # Load best model from early stopping
        if early_stopping.best_model_state is not None:
            early_stopping.load_best_model(self.model)
            epoch_pbar.write("[INFO] Loaded best model from early stopping")
        
        epoch_pbar.close()

        return train_losses, val_losses

    def validate(self, val_data:torch.utils.data.DataLoader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obss, acts in val_data:
                if self.preprocess_inputs:
                    obss = self.obs_preprocessor(obss).to(self.device)
                else:
                    obss = obss.to(self.device)
                acts = acts.to(self.device)

                mu, log_sigma = self.model(obss)
                batch_loss = gaussian_nll_loss(mu, log_sigma, acts)
                val_loss += batch_loss.item()
        
        avg_val_loss = val_loss / len(val_data)
        return avg_val_loss
    
    def most_likely_component(self, inputs:torch.Tensor)->torch.Tensor:
        return self.model.forward(inputs)[0]
    
    def save(self, folder_path:str):
        model_path = os.path.join(folder_path, "mle_expert.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"[INFO]: MLE Expert model saved to {model_path}")
        if self.preprocess_inputs:
            experts_preprocessor_path = os.path.join(folder_path, "obs_preprocessor.pth")
            torch.save(self.obs_preprocessor.state_dict(), experts_preprocessor_path)
            print(f"[INFO]: Expert observation preprocessor saved to {experts_preprocessor_path}")


class GMMExpert:
    def __init__(self, cfg:Dict):
        self.obs_dim = cfg['obs_dim']
        self.action_dim = cfg['action_dim']

        self.cfg = cfg
        self.expert_domain = cfg['expert_domain']
        self.device = cfg['device']
        self.validation_interval = cfg.get("validation_interval", 1)
        self.patience = cfg.get("early_stopping_patience", 10)
        self.min_delta = cfg.get("min_delta", 1e-4)
        self.grad_clipping = cfg.get("grad_clipping", None)

        self.preprocess_inputs = cfg.get("preprocess_inputs", True)
        if self.preprocess_inputs:
            self.obs_preprocessor = RunningStandardScaler(size=self.obs_dim, device=self.device)


        self.init_expert()

    def init_expert(self):
        self.lr = self.cfg["gmm"]["learning_rate"]
        self.act_fct = self.cfg["gmm"]["activation_fct"]
        self.hidden_sizes = self.cfg["gmm"]["hidden_sizes"]
        self.dropout = self.cfg["gmm"].get("dropout", 0.0)
        self.weight_decay = self.cfg["gmm"].get("weight_decay", 0.0)
        self.gmm_components = self.cfg["gmm"].get("n_components", 5)

        self.model = GMM(self.obs_dim, self.action_dim, 1.0, self.gmm_components, self.hidden_sizes, 
                        self.lr, self.act_fct, dropout=self.dropout, 
                        weight_decay=self.weight_decay).to(self.device)
        init_model_weights(self.model)
        print("[INFO]: Models initialized")
        print("[INFO]: GMM MLE Expert summary:")
        print_model_summary(self.model, input_size=(1, self.obs_dim))


    def train(self, train_data:torch.utils.data.DataLoader, val_data:torch.utils.data.DataLoader):
        self.model.train()
        num_epochs = self.cfg["gmm"]["n_epochs"]

        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta, verbose=True)

        train_losses = []
        val_losses = []

        epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
        for epoch in epoch_pbar:
            epoch_loss = 0.0
            for obss, acts in train_data:
                if self.preprocess_inputs:
                    obss = self.obs_preprocessor(obss).to(self.device)
                else:
                    obss = obss.to(self.device)
                acts = acts.to(self.device)

                self.model.optimizer.zero_grad()
                mu, log_sigma, pi = self.model(obss)
                batch_loss = gmm_nll_loss(mu, log_sigma, pi, acts)
                if self.grad_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
                batch_loss.backward()
                
                self.model.optimizer.step()

                epoch_loss += batch_loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_data)
            train_losses.append(avg_epoch_loss)
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_epoch_loss:.4f}',
                'lr': f'{self.model.optimizer.param_groups[0]["lr"]:.6f}',
            })

            if (epoch + 1) % self.cfg["val_interval"] == 0:
                val_loss = self.validate(val_data)
                val_losses.append(val_loss)
                # print(f"Validation Loss after Epoch {epoch+1}: {val_loss:.4f}")
                epoch_pbar.set_postfix({
                'train_loss': f'{avg_epoch_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'lr': f'{self.model.optimizer.param_groups[0]["lr"]:.6f}'
            })
                
                # Step scheduler based on validation loss
                self.model.scheduler.step(val_loss)
                
                if early_stopping(val_loss, self.model):
                    # print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
                    # print(f"[INFO] Best validation loss: {early_stopping.best_loss:.6f }")
                    epoch_pbar.write(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
                    epoch_pbar.write(f"[INFO] Best validation loss: {early_stopping.best_loss:.6f}")
                    break
                
                if early_stopping.best_model_state is not None:
                    early_stopping.load_best_model(self.model)
                    epoch_pbar.write("[INFO] Loaded best model from early stopping")
        
        epoch_pbar.close()

        return train_losses, val_losses
    
    def validate(self, val_data:torch.utils.data.DataLoader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obss, acts in val_data:
                if self.preprocess_inputs:
                    obss = self.obs_preprocessor(obss).to(self.device)
                else:
                    obss = obss.to(self.device)
                acts = acts.to(self.device)

                mu, log_sigma, pi = self.model(obss)
                batch_loss = gmm_nll_loss(mu, log_sigma, pi, acts)
                val_loss += batch_loss.item()
        
        avg_val_loss = val_loss / len(val_data)
        return avg_val_loss
    
    def most_likely_component(self, inputs:torch.Tensor)->torch.Tensor:
        mu, _, pi = self.model.forward(inputs)
        batch_size = inputs.size(0)
        _, component_indices = torch.max(pi, dim=-1)

        most_likely_means = mu[torch.arange(batch_size), component_indices]
        return most_likely_means
    
    def save(self, folder_path:str):
        model_path = os.path.join(folder_path, "gmm_expert.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"[INFO]: GMM MLE Expert model saved to {model_path}")
        if self.preprocess_inputs:
            experts_preprocessor_path = os.path.join(folder_path, "obs_preprocessor.pth")
            torch.save(self.obs_preprocessor.state_dict(), experts_preprocessor_path)
            print(f"[INFO]: Expert observation preprocessor saved to {experts_preprocessor_path}")



class CNFExpert:
    def __init__(self, cfg:Dict):
        self.obs_dim = cfg['obs_dim']
        self.action_dim = cfg['action_dim']

        self.cfg = cfg
        self.expert_domain = cfg['expert_domain']
        self.device = cfg['device']
        self.validation_interval = cfg.get("validation_interval", 1)
        self.patience = cfg.get("early_stopping_patience", 10)
        self.min_delta = cfg.get("min_delta", 1e-4)
        self.grad_clipping = cfg.get("grad_clipping", None)

        self.preprocess_inputs = cfg.get("preprocess_inputs", True)
        if self.preprocess_inputs:
            self.obs_preprocessor = RunningStandardScaler(size=self.obs_dim, device=self.device)

        self.init_expert()

    def init_expert(self):
        self.lr = self.cfg["cnf"]["learning_rate"]
        self.act_fct = self.cfg["cnf"]["conditioner_activation_fct"]
        self.hidden_sizes = self.cfg["cnf"]["conditionner_hidden_sizes"]
        self.dropout = self.cfg["cnf"].get("dropout", 0.02)
        self.num_layers = self.cfg["cnf"]["n_layers"]
        self.split_dim = self.action_dim // 2
        
        # Use MLE model for conditional base distribution (obs -> mu, log_sigma)
        self.use_conditional_base = self.cfg["cnf"].get("use_conditional_base", True)
        
        if self.use_conditional_base:
            # MLE network: obs -> (mu, log_sigma) for base distribution
            mle_hidden_sizes = self.cfg["cnf"].get("base_hidden_sizes", [256, 256])
            mle_lr = self.cfg["cnf"].get("base_learning_rate", self.lr)
            mle_activation = self.cfg["cnf"].get("base_activation_fct", self.act_fct)
            mle_dropout = self.cfg["cnf"].get("base_dropout", 0.0)
            mle_weight_decay = self.cfg["cnf"].get("base_weight_decay", 0.0)
            
            self.base_model = MLE(
                input_dim=self.obs_dim,
                output_dim=self.action_dim,
                action_lim=1.0,
                hidden_dims=mle_hidden_sizes,
                lr=mle_lr,
                activation_fct=mle_activation,
                dropout=mle_dropout,
                weight_decay=mle_weight_decay,
                tanh_flag=False
            ).to(self.device)
            
            init_model_weights(self.base_model)
            print("[INFO]: Conditional base distribution (MLE) initialized")
            print("[INFO]: Base MLE summary:")
            print_model_summary(self.base_model, input_size=(1, self.obs_dim))
        else:
            self.base_model = torch.distributions.Normal(
                loc=torch.zeros(self.action_dim, device=self.device),
                scale=torch.ones(self.action_dim, device=self.device)
            )
            print("[INFO]: Using standard normal base distribution (unconditional)")
        
        # Build Real NVP flow coupling layers directly
        self.conditioners = nn.ModuleList()
        
        for i in range(self.num_layers):
            # Build conditioner network for this layer
            layers = []
            prev_dim = self.split_dim
            
            for j, hidden_dim in enumerate(self.hidden_sizes):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(get_activation(self.act_fct))
                if j < len(self.hidden_sizes) - 1:
                    layers.append(nn.Dropout(self.dropout))
                prev_dim = hidden_dim
            
            conditioner = nn.ModuleDict({
                'net': nn.Sequential(*layers),
                's_head': nn.Linear(prev_dim, self.action_dim - self.split_dim),
                't_head': nn.Linear(prev_dim, self.action_dim - self.split_dim)
            })
            self.conditioners.append(conditioner)
        
        for conditioner in self.conditioners:
            init_model_weights(conditioner['net'])
            init_model_weights(conditioner['s_head'])
            init_model_weights(conditioner['t_head'])
        
        # Optimizer includes both flow and base model parameters if conditional
        if self.use_conditional_base:
            flow_params = []
            for conditioner in self.conditioners:
                flow_params.extend(conditioner.parameters())
            params = flow_params + list(self.base_model.parameters())
        else:
            params = []
            for conditioner in self.conditioners:
                params.extend(conditioner.parameters())
            
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3,
            min_lr=1e-6
        )
        
        print(f"[INFO]: Real NVP Flow initialized with {self.num_layers} coupling layers")
        print(f"[INFO]: Split dimension: {self.split_dim}, Output dimension: {self.action_dim - self.split_dim}")
    
    def _affine_forward(self, z: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> tuple:
        z_lower = z[:, :self.split_dim]
        z_upper = z[:, self.split_dim:]
        
        z_prime_upper = z_upper * torch.exp(s) + t
        z_prime = torch.cat([z_lower, z_prime_upper], dim=1)
        
        log_det = s.sum(dim=1)
        
        return z_prime, log_det
    
    def _affine_inverse(self, z_prime: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> tuple:
        z_prime_lower = z_prime[:, :self.split_dim]
        z_prime_upper = z_prime[:, self.split_dim:]
        
        z_upper = (z_prime_upper - t) * torch.exp(-s)
        z = torch.cat([z_prime_lower, z_upper], dim=1)
        
        log_det = -s.sum(dim=1)
        
        return z, log_det
    
    def _flow_forward(self, z: torch.Tensor) -> tuple:
        log_det_total = torch.zeros(z.size(0), device=z.device)
        x = z
        
        for i, conditioner in enumerate(self.conditioners):
            # Alternate permutation for better mixing
            if i % 2 == 1:
                x = torch.flip(x, dims=[1])
            
            z_lower = x[:, :self.split_dim]
            logits = conditioner['net'](z_lower)
            s = conditioner['s_head'](logits)
            t = conditioner['t_head'](logits)
            
            x, log_det = self._affine_forward(x, s, t)
            log_det_total += log_det
            
            # Reverse permutation
            if i % 2 == 1:
                x = torch.flip(x, dims=[1])
        
        return x, log_det_total
    
    def _flow_inverse(self, x: torch.Tensor) -> tuple:
        log_det_total = torch.zeros(x.size(0), device=x.device)
        z = x
        
        # Reverse order for inverse
        for i in reversed(range(len(self.conditioners))):
            if i % 2 == 1:
                z = torch.flip(z, dims=[1])
            
            z_lower = z[:, :self.split_dim]
            logits = self.conditioners[i]['net'](z_lower)
            s = self.conditioners[i]['s_head'](logits)
            t = self.conditioners[i]['t_head'](logits)
            
            z, log_det = self._affine_inverse(z, s, t)
            log_det_total += log_det
            
            if i % 2 == 1:
                z = torch.flip(z, dims=[1])
        
        return z, log_det_total
    
    def _compute_log_prob(self, x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        z, log_det = self._flow_inverse(x)
        
        sigma = torch.exp(log_sigma) + 1e-5
        log_prob_z = -0.5 * torch.log(2 * torch.pi * sigma**2) - 0.5 * ((z - mu)**2 / sigma**2)
        log_prob_z = log_prob_z.sum(dim=-1)
        
        return log_prob_z + log_det
    
    def _sample_from_flow(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma) + 1e-5
        z = torch.normal(mu, sigma)

        x, _ = self._flow_forward(z)
        
        return x
    
    def train(self, train_data:torch.utils.data.DataLoader, val_data:torch.utils.data.DataLoader):
        if self.use_conditional_base:
            self.base_model.train()
        for conditioner in self.conditioners:
            conditioner.train()
            
        num_epochs = self.cfg["cnf"]["n_epochs"]
        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta, verbose=True)

        train_losses = []
        val_losses = []

        epoch_pbar = tqdm(range(num_epochs), desc="Training CNF", unit="epoch")
    
        for epoch in epoch_pbar:
            epoch_loss = 0.0
            for obss, acts in train_data:
                if self.preprocess_inputs:
                    obss = self.obs_preprocessor(obss.to(self.device), train=True)
                else:
                    obss = obss.to(self.device)
                acts = acts.to(self.device)

                self.optimizer.zero_grad()

                if self.use_conditional_base:
                    mu, log_sigma = self.base_model(obss)
                else:
                    mu = torch.zeros_like(acts)
                    log_sigma = torch.zeros_like(acts)
                
                # Compute negative log-likelihood
                log_prob = self._compute_log_prob(acts, mu, log_sigma) #TODO: check if better with regularization nll from utils.
                batch_loss = -log_prob.mean()
                
                batch_loss.backward()
                
                if self.grad_clipping is not None:
                    if self.use_conditional_base:
                        flow_params = []
                        for conditioner in self.conditioners:
                            flow_params.extend(conditioner.parameters())
                        torch.nn.utils.clip_grad_norm_(
                            flow_params + list(self.base_model.parameters()), 
                            self.grad_clipping
                        )
                    else:
                        flow_params = []
                        for conditioner in self.conditioners:
                            flow_params.extend(conditioner.parameters())
                        torch.nn.utils.clip_grad_norm_(flow_params, self.grad_clipping)
                
                self.optimizer.step()
                epoch_loss += batch_loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_data)
            train_losses.append(avg_epoch_loss)
            
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_epoch_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
            })

            if (epoch + 1) % self.validation_interval == 0:
                val_loss = self.validate(val_data)
                val_losses.append(val_loss)
                self.scheduler.step(val_loss)
                
                # Early stopping check (pass first conditioner as proxy for all flow params)
                if early_stopping(val_loss, self.conditioners[0]):
                    epoch_pbar.write(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model from early stopping
        if early_stopping.best_model_state is not None:
            early_stopping.load_best_model(self.conditioners[0])
            epoch_pbar.write("[INFO] Loaded best model from early stopping")
        
        epoch_pbar.close()

        return train_losses, val_losses

    def validate(self, val_data:torch.utils.data.DataLoader):
        if self.use_conditional_base:
            self.base_model.eval()
        for conditioner in self.conditioners:
            conditioner.eval()
            
        val_loss = 0.0
        with torch.no_grad():
            for obss, acts in val_data:
                if self.preprocess_inputs:
                    obss = self.obs_preprocessor(obss.to(self.device), train=False)
                else:
                    obss = obss.to(self.device)
                acts = acts.to(self.device)

                if self.use_conditional_base:
                    mu, log_sigma = self.base_model(obss)
                else:
                    mu = torch.zeros_like(acts)
                    log_sigma = torch.zeros_like(acts)
                
                # Compute negative log-likelihood TODO: match with loss function in train
                log_prob = self._compute_log_prob(acts, mu, log_sigma)
                batch_loss = -log_prob.mean()
                val_loss += batch_loss.item()
        
        avg_val_loss = val_loss / len(val_data)
        return avg_val_loss
    
    def sample(self, obss:torch.Tensor, num_samples:int=1, deterministic:bool=False)->torch.Tensor:
        if self.use_conditional_base:
            self.base_model.eval()
        for conditioner in self.conditioners:
            conditioner.eval()
            
        with torch.no_grad():
            if self.preprocess_inputs:
                obss = self.obs_preprocessor(obss.to(self.device), train=False)
            else:
                obss = obss.to(self.device)
                
            if self.use_conditional_base:
                mu, log_sigma = self.base_model(obss)
                
                if deterministic:
                    samples = self._sample_from_flow(mu, log_sigma)
                    samples = samples.unsqueeze(1)  # Add num_samples dimension
                else:
                    mu_expanded = mu.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, mu.size(-1))
                    log_sigma_expanded = log_sigma.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, log_sigma.size(-1))
                    samples = self._sample_from_flow(mu_expanded, log_sigma_expanded)
                    samples = samples.view(obss.size(0), num_samples, -1)
            else:
                # Use standard normal base
                mu = torch.zeros(obss.size(0), self.action_dim, device=self.device)
                log_sigma = torch.zeros(obss.size(0), self.action_dim, device=self.device)
                
                if deterministic:
                    samples = self._sample_from_flow(mu, log_sigma)
                    samples = samples.unsqueeze(1)  # Add num_samples dimension
                else:
                    mu_expanded = mu.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, mu.size(-1))
                    log_sigma_expanded = log_sigma.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, log_sigma.size(-1))
                    samples = self._sample_from_flow(mu_expanded, log_sigma_expanded)
                    samples = samples.view(obss.size(0), num_samples, -1)
                    
        return samples
    
    def most_likely_component(self, obss:torch.Tensor)->torch.Tensor:
        if self.use_conditional_base:
            self.base_model.eval()
        for conditioner in self.conditioners:
            conditioner.eval()
        with torch.no_grad():
            if self.preprocess_inputs:
                obss = self.obs_preprocessor(obss.to(self.device), train=False)
            else:
                obss = obss.to(self.device)
                
            if self.use_conditional_base:
                mu, log_sigma = self.base_model(obss)
            else:
                mu = torch.zeros(obss.size(0), self.action_dim, device=self.device)
                log_sigma = torch.zeros(obss.size(0), self.action_dim, device=self.device)

            mode = mu
            transformed_mode, _ = self._flow_forward(mode)
        return transformed_mode
    
    def save(self, folder_path:str):
        conditioners_state = [conditioner.state_dict() for conditioner in self.conditioners]
        model_path = os.path.join(folder_path, "cnf_expert.pth")
        torch.save({
            'conditioners': conditioners_state,
            'split_dim': self.split_dim,
            'num_layers': self.num_layers
        }, model_path)
        print(f"[INFO]: CNF Expert flow model saved to {model_path}")
        
        if self.use_conditional_base:
            base_model_path = os.path.join(folder_path, "cnf_base_model.pth")
            torch.save(self.base_model.state_dict(), base_model_path)
            print(f"[INFO]: CNF Base model saved to {base_model_path}")
        
        if self.preprocess_inputs:
            experts_preprocessor_path = os.path.join(folder_path, "obs_preprocessor.pth")
            torch.save(self.obs_preprocessor.state_dict(), experts_preprocessor_path)
            print(f"[INFO]: Expert observation preprocessor saved to {experts_preprocessor_path}")
