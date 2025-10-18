import os
from typing import Dict
import torch

from tqdm import tqdm
from skrl.resources.preprocessors.torch import RunningStandardScaler
from base_classes.models import MLE, GMM #, CNF
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
    
    
    def save(self, folder_path:str):
        model_path = os.path.join(folder_path, "mle_expert.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"[INFO]: MLE Expert model saved to {model_path}")
        if self.preprocess_inputs:
            experts_preprocessor_path = os.path.join(folder_path, "obs_preprocessor.pth")
            torch.save(self.obs_preprocessor.state_dict(), experts_preprocessor_path)
            print(f"[INFO]: Expert observation preprocessor saved to {experts_preprocessor_path}")


class GMM_MLEExpert:
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
    
    def save(self, folder_path:str):
        model_path = os.path.join(folder_path, "gmm_expert.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"[INFO]: GMM MLE Expert model saved to {model_path}")
        if self.preprocess_inputs:
            experts_preprocessor_path = os.path.join(folder_path, "obs_preprocessor.pth")
            torch.save(self.obs_preprocessor.state_dict(), experts_preprocessor_path)
            print(f"[INFO]: Expert observation preprocessor saved to {experts_preprocessor_path}")

