import os
from typing import Dict, Tuple, Optional
from networkx import sigma
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from skrl.resources.preprocessors.torch import RunningStandardScaler
from base_classes.models import MLE, GMM, AffineCoupling
from base_classes.utils import (gaussian_nll_loss, init_model_weights, 
                                print_model_summary, EarlyStopping,
                                gmm_nll_loss)


class MLEExpert:
    def __init__(self, cfg:Dict):
        self.obs_dim = cfg['obs_dim']
        self.action_dim = cfg['action_dim']

        self.cfg = cfg
        self.expert_domain = cfg['expert_domain']
        self.device = cfg['device']
        self.action_limit = cfg.get('action_limit', 1.0)
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

        self.model = MLE(self.obs_dim, self.action_dim, self.action_limit, self.hidden_sizes, 
                        self.lr, self.act_fct, dropout=self.dropout, 
                        weight_decay=self.weight_decay).to(self.device)
        init_model_weights(self.model)
        print("[INFO]: Models initialized")
        print("[INFO]: MLE Expert summary:")
        print_model_summary(self.model, input_size=(1, self.obs_dim))

    def fit_obs_preprocessor(self, obss:torch.Tensor):
        if self.preprocess_inputs:
            self.obs_preprocessor(obss, train=True)
            self.obs_preprocessor.eval()
        else:
            raise ValueError("Observation preprocessor must be enabled to fit.")
      
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
                    obss = self.obs_preprocessor(obss, train=False).to(self.device)
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
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_epoch_loss:.4f}',
                'lr': f'{self.model.optimizer.param_groups[0]["lr"]:.6f}',
            })

            if (epoch + 1) % self.cfg["val_interval"] == 0:
                val_loss = self.validate(val_data)
                val_losses.append(val_loss)
                epoch_pbar.set_postfix({
                'train_loss': f'{avg_epoch_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'lr': f'{self.model.optimizer.param_groups[0]["lr"]:.6f}'
            })
                
                # Step scheduler based on validation loss
                self.model.scheduler.step(val_loss)
                
                if early_stopping(val_loss, self.model):
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
                    obss = self.obs_preprocessor(obss, train=False).to(self.device)
                else:
                    obss = obss.to(self.device)
                acts = acts.to(self.device)

                mu, log_sigma = self.model(obss)
                batch_loss = gaussian_nll_loss(mu, log_sigma, acts)
                val_loss += batch_loss.item()
        
        avg_val_loss = val_loss / len(val_data)
        return avg_val_loss
    
    def most_likely_component(self, inputs:torch.Tensor)->torch.Tensor:
        if self.preprocess_inputs:
            inputs = self.obs_preprocessor(inputs, train=False).to(self.device)
        else:
            inputs = inputs.to(self.device)
        return self.model.forward(inputs)[0]
    
    def sample(self, inputs:torch.Tensor, generator:torch.Generator=None)->torch.Tensor:
        if self.preprocess_inputs:
            inputs = self.obs_preprocessor(inputs, train=False).to(self.device)
        else:
            inputs = inputs.to(self.device)
        mu, log_sigma = self.model.forward(inputs)
        sigma = torch.exp(log_sigma)
        normal_dist = torch.distributions.Normal(mu, sigma)
        sampled_actions = normal_dist.sample(generator=generator)
        return sampled_actions
    
    def eval(self):
        self.model.eval()
    
    def save(self, folder_path:str):
        model_path = os.path.join(folder_path, "mle_expert.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"[INFO]: MLE Expert model saved to {model_path}")
        if self.preprocess_inputs:
            experts_preprocessor_path = os.path.join(folder_path, "obs_preprocessor.pth")
            torch.save(self.obs_preprocessor.state_dict(), experts_preprocessor_path)
            print(f"[INFO]: Expert observation preprocessor saved to {experts_preprocessor_path}")
    
    def load(self, folder_path:str):
        model_path = os.path.join(folder_path, "mle_expert.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"[INFO]: MLE Expert model loaded from {model_path}")
        if self.preprocess_inputs:
            experts_preprocessor_path = os.path.join(folder_path, "obs_preprocessor.pth")
            self.obs_preprocessor.load_state_dict(torch.load(experts_preprocessor_path, map_location=self.device))
            print(f"[INFO]: Expert observation preprocessor loaded from {experts_preprocessor_path}")


class GMMExpert:
    def __init__(self, cfg:Dict):
        self.obs_dim = cfg['obs_dim']
        self.action_dim = cfg['action_dim']

        self.cfg = cfg
        self.expert_domain = cfg['expert_domain']
        self.device = cfg['device']
        self.action_limit = cfg.get('action_limit', 1.0)
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

        self.model = GMM(self.obs_dim, self.action_dim, self.action_limit, self.gmm_components, self.hidden_sizes, 
                        self.lr, self.act_fct, dropout=self.dropout, 
                        weight_decay=self.weight_decay).to(self.device)
        init_model_weights(self.model)
        print("[INFO]: Models initialized")
        print("[INFO]: GMM MLE Expert summary:")
        print_model_summary(self.model, input_size=(1, self.obs_dim))

    def fit_obs_preprocessor(self, obss:torch.Tensor):
        if self.preprocess_inputs:
            self.obs_preprocessor(obss, train=True)
            self.obs_preprocessor.eval()
        else: 
            raise ValueError("Observation preprocessor must be enabled to fit.")


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
                    obss = self.obs_preprocessor(obss, train=False).to(self.device)
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
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_epoch_loss:.4f}',
                'lr': f'{self.model.optimizer.param_groups[0]["lr"]:.6f}',
            })

            if (epoch + 1) % self.cfg["val_interval"] == 0:
                val_loss = self.validate(val_data)
                val_losses.append(val_loss)
                epoch_pbar.set_postfix({
                'train_loss': f'{avg_epoch_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'lr': f'{self.model.optimizer.param_groups[0]["lr"]:.6f}'
            })
                
                # Step scheduler based on validation loss
                self.model.scheduler.step(val_loss)
                
                if early_stopping(val_loss, self.model):
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
                    obss = self.obs_preprocessor(obss, train=False).to(self.device)
                else:
                    obss = obss.to(self.device)
                acts = acts.to(self.device)

                mu, log_sigma, pi = self.model(obss)
                batch_loss = gmm_nll_loss(mu, log_sigma, pi, acts)
                val_loss += batch_loss.item()
        
        avg_val_loss = val_loss / len(val_data)
        return avg_val_loss
    
    def most_likely_component(self, inputs:torch.Tensor)->torch.Tensor:
        if self.preprocess_inputs:
            inputs = self.obs_preprocessor(inputs, train=False).to(self.device)
        else:
            inputs = inputs.to(self.device)
        mu, _, pi = self.model.forward(inputs)
        batch_size = inputs.size(0)
        _, component_indices = torch.max(pi, dim=-1)

        most_likely_means = mu[torch.arange(batch_size), component_indices]
        return most_likely_means
    
    def sample(self, inputs:torch.Tensor, generator:torch.Generator=None)->torch.Tensor:
        if self.preprocess_inputs:
            inputs = self.obs_preprocessor(inputs, train=False).to(self.device)
        else:
            inputs = inputs.to(self.device)
        mu, log_sigma, pi = self.model.forward(inputs)
        batch_size = inputs.size(0)
        categorical = torch.distributions.Categorical(pi)
        component_indices = categorical.sample(generator=generator)

        means = mu[torch.arange(batch_size), component_indices]
        sigmas = torch.exp(log_sigma[torch.arange(batch_size), component_indices])
        normal_dist = torch.distributions.Normal(means, sigmas)
        sampled_actions = normal_dist.sample(generator=generator)
        return sampled_actions
    
    def eval(self):
        self.model.eval()
    
    def save(self, folder_path:str):
        model_path = os.path.join(folder_path, "gmm_expert.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"[INFO]: GMM MLE Expert model saved to {model_path}")
        if self.preprocess_inputs:
            experts_preprocessor_path = os.path.join(folder_path, "obs_preprocessor.pth")
            torch.save(self.obs_preprocessor.state_dict(), experts_preprocessor_path)
            print(f"[INFO]: Expert observation preprocessor saved to {experts_preprocessor_path}")
    
    def load(self, folder_path:str):
        model_path = os.path.join(folder_path, "gmm_expert.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"[INFO]: GMM MLE Expert model loaded from {model_path}")
        if self.preprocess_inputs:
            experts_preprocessor_path = os.path.join(folder_path, "obs_preprocessor.pth")
            self.obs_preprocessor.load_state_dict(torch.load(experts_preprocessor_path, map_location=self.device))
            print(f"[INFO]: Expert observation preprocessor loaded from {experts_preprocessor_path}")


class CNFExpert(nn.Module):
    def __init__(self, cfg:Dict):
        super().__init__()
        self.obs_dim = cfg['obs_dim']
        self.action_dim = cfg['action_dim']

        self.cfg = cfg
        self.expert_domain = cfg['expert_domain']
        self.device = cfg['device']
        self.action_limit = cfg.get('action_limit', 1.0)
        self.lr = cfg.get("learning_rate", 1e-3)
        self.validation_interval = cfg.get("validation_interval", 1)
        self.patience = cfg.get("early_stopping_patience", 10)
        self.min_delta = cfg.get("min_delta", 1e-4)
        self.grad_clipping = cfg.get("grad_clipping", None)
        self.n_flows = cfg.get("n_flows", 10)
        self.hidden_dims = cfg.get("hidden_dims", (512, 512, 256))
        self.epochs = cfg.get("n_epochs", 100)

        self.preprocess_inputs = cfg.get("preprocess_inputs", True)
        if self.preprocess_inputs:
            self.obs_preprocessor = RunningStandardScaler(size=self.obs_dim, device=self.device)
        
        masks = []
        for i in range(self.n_flows):
            mask = torch.tensor([((j + i) % 2) for j in range(self.action_dim)], dtype=torch.bool)
            masks.append(mask)

        self.couplings = nn.ModuleList([
            AffineCoupling(self.action_dim, self.obs_dim, self.hidden_dims, mask) for mask in masks
        ])
        self.register_buffer("base_mean", torch.zeros(self.action_dim))
        self.register_buffer("base_std", torch.ones(self.action_dim))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta, verbose=True)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                           factor=0.5, patience=5)

    def forward(self, z, state):
        '''Compute forward transformation from latent code z to action: decoding.'''
        log_det = torch.zeros(z.size(0), device=z.device)
        x = z
        for coupling in self.couplings:
            x, ld = coupling.forward(x, state)
            log_det = log_det + ld
        # Apply tanh to bound actions to [-action_limit, action_limit]
        x = torch.tanh(x) * self.action_limit
        return x, log_det

    def inverse(self, action, state):
        '''Compute inverse transformation from action -> z: encoding.'''
        # Apply inverse tanh (atanh) to unbounded actions
        action_unbounded = torch.atanh(torch.clamp(action / self.action_limit, -0.999, 0.999)) #TODO: action limit
        
        log_det = torch.zeros(action.size(0), device=action.device)
        z = action_unbounded
        for coupling in reversed(self.couplings):
            z, ld = coupling.inverse(z, state)
            log_det = log_det + ld
        return z, log_det

    def log_prob(self, action, state):
        '''Compute log probability of action given state.'''
        z, log_det = self.inverse(action, state)
        # base log prob
        log_pz = -0.5 * torch.log(2 * torch.pi * (self.base_std ** 2)) - 0.5 * ((z - self.base_mean) ** 2) / (self.base_std ** 2)
        log_pz = log_pz.sum(dim=1)
        return log_pz + log_det
    
    def regularized_loss(self, action, state, reg_weight=0.01):
        """Compute negative log likelihood loss with L2 regularization on latent codes."""
        lp = self.log_prob(action, state)
        nll_loss = -lp.mean()
        z, _ = self.inverse(action, state)
        l2_reg = (z ** 2).mean()
        total_loss = nll_loss + reg_weight * l2_reg
        
        return total_loss
    
    def sample(self, state, n_samples=1):
        """Sample actions conditioned on state."""
        self.eval()
        with torch.no_grad():
            if self.preprocess_inputs:
                state = self.obs_preprocessor(state, train=False)
            
            B = state.size(0)
            z = torch.randn(B * n_samples, self.action_dim, device=state.device)
            state_exp = state.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, -1)
            actions, _ = self.forward(z, state_exp)
            actions = actions.view(B, n_samples, self.action_dim)
        return actions
    
    def train(self, train_data:torch.utils.data.DataLoader, val_data:torch.utils.data.DataLoader):
        self.couplings.train()
        train_losses = []
        val_losses = []

        epoch_pbar = tqdm(range(self.epochs), desc="Training", unit="epoch")

        for epoch in epoch_pbar:
            epoch_loss = 0.0
            for state_batch, action_batch in train_data:

                state_batch = state_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                
                if self.preprocess_inputs:
                    state_batch = self.obs_preprocessor(state_batch, train=False).to(self.device)

                self.optimizer.zero_grad()
                loss = self.regularized_loss(action_batch, state_batch, reg_weight=1e-4)
                if self.grad_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clipping)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * action_batch.size(0)
            
            avg_epoch_loss = epoch_loss / len(train_data.dataset)
            train_losses.append(avg_epoch_loss)
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_epoch_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
            })

            if (epoch + 1) % self.cfg["val_interval"] == 0:
                val_loss = self.validate(val_data)
                val_losses.append(val_loss)
                epoch_pbar.set_postfix({
                    'train_loss': f'{avg_epoch_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Step scheduler based on validation loss
                self.scheduler.step(val_loss)
                
                if self.early_stopping(val_loss, self):
                    epoch_pbar.write(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
                    epoch_pbar.write(f"[INFO] Best validation loss: {self.early_stopping.best_loss:.6f}")
                    break
        
        # Load best model from early stopping
        if self.early_stopping.best_model_state is not None:
            self.early_stopping.load_best_model(self)
            epoch_pbar.write("[INFO] Loaded best model from early stopping")
        
        epoch_pbar.close()

        return train_losses, val_losses
    
    def validate(self, val_data:torch.utils.data.DataLoader):
        self.couplings.eval()
        loss = 0.0
        with torch.no_grad():
            for state_batch, action_batch in val_data:
                state_batch = state_batch.to(self.device)
                action_batch = action_batch.to(self.device)

                if self.preprocess_inputs:
                    state_batch = self.obs_preprocessor(state_batch, train=False).to(self.device)
                
                batch_loss = self.regularized_loss(action_batch, state_batch, reg_weight=1e-4)
                
                loss += batch_loss.item() * action_batch.size(0)
                
            avg_loss = loss / len(val_data.dataset)
        return avg_loss
    
    def fit_obs_preprocessor(self, obss:torch.Tensor):
        if self.preprocess_inputs:
            self.obs_preprocessor(obss, train=True)
            self.obs_preprocessor.eval()
        else:
            raise ValueError("Observation preprocessor must be enabled to fit.")

    def eval(self):
        self.couplings.eval()
    
    def save(self, folder_path:str):
        model_path = os.path.join(folder_path, "cnf_expert.pth")
        torch.save(self.state_dict(), model_path)
        print(f"[INFO]: CNF Expert model saved to {model_path}")
        if self.preprocess_inputs:
            experts_preprocessor_path = os.path.join(folder_path, "obs_preprocessor.pth")
            torch.save(self.obs_preprocessor.state_dict(), experts_preprocessor_path)
            print(f"[INFO]: Expert observation preprocessor saved to {experts_preprocessor_path}")
    
    def load(self, folder_path:str):
        model_path = os.path.join(folder_path, "cnf_expert.pth")
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"[INFO]: CNF Expert model loaded from {model_path}")
        if self.preprocess_inputs:
            experts_preprocessor_path = os.path.join(folder_path, "obs_preprocessor.pth")
            self.obs_preprocessor.load_state_dict(torch.load(experts_preprocessor_path, map_location=self.device))
            print(f"[INFO]: Expert observation preprocessor loaded from {experts_preprocessor_path}")

    def most_likely_component(self, inputs:torch.Tensor)->torch.Tensor:
        self.eval()
        with torch.no_grad():
            if self.preprocess_inputs:
                inputs = self.obs_preprocessor(inputs, train=False)
            
            batch_size = inputs.size(0)
        
            z_mode = torch.zeros(batch_size, self.action_dim, device=inputs.device)
            best_actions, _ = self.forward(z_mode, inputs)
            
            return best_actions


    


        
