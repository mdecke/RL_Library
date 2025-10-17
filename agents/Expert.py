from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from skrl.resources.preprocessors.torch import RunningStandardScaler
from base_classes.models import MLE #, CNF
from base_classes.utils import gaussian_nll_loss, EarlyStopping




class Expert:
    def __init__(self, env, cfg:Dict):
        self.env = env
        self.obs_dim = env.single_observation_space.shape[0]
        self.action_dim = env.single_action_space.shape[0]

        self.cfg = cfg
        self.expert_type = cfg['expert_type']
        self.expert_domain = cfg['expert_domain']
        self.device = cfg['device']
        self.validation_interval = cfg.get("validation_interval", 1)
        self.patience = cfg.get("early_stopping_patience", 10)
        self.min_delta = cfg.get("min_delta", 1e-4)

        self.preprocess_inputs = cfg.get("preprocess_inputs", True)
        if self.preprocess_inputs:
            self.obs_preprocessor = RunningStandardScaler(size=self.obs_dim, device=self.device)


        self.init_expert()

    def init_expert(self):
        self.lr = self.cfg[f"{self.expert_type}"]["learning_rate"]
        self.act_fct = self.cfg[f"{self.expert_type}"]["activation_fct"]

        if self.expert_type == 'mle':
            self.model = MLE(self.action_dim, self.obs_dim, self.cfg["hidden_sizes"], self.lr, self.act_fct).to(self.device)
        # elif self.expert_type == 'cnf':
        #     Raise NotImplementedError("CNF expert not yet implemented")
        else:
            raise ValueError(f"Expert type '{self.expert_type}' is not recognized.")

    def train_mle(self, train_data:torch.utils.data.DataLoader, val_data:torch.utils.data.DataLoader):
        self.model.train()
        num_epochs = self.cfg["num_epochs"]

        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta, verbose=True)

        losses = []

        for epoch in range(num_epochs):
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
                batch_loss.backward()
                self.model.optimizer.step()

                epoch_loss += batch_loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_data)
            losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

            if (epoch + 1) % self.cfg["val_interval"] == 0:
                val_loss = self.validate_mle(val_data)
                print(f"Validation Loss after Epoch {epoch+1}: {val_loss:.4f}")
                if early_stopping(val_loss, self.model):
                    print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
                    print(f"[INFO] Best validation loss: {early_stopping.best_loss:.6f}")
                    break
            
            self.model.scheduler.step(avg_epoch_loss)
        return losses
    
    def validate_mle(self, val_data:torch.utils.data.DataLoader):
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
    
    def test_mle(self, test_data:torch.utils.data.DataLoader):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for obss, acts in test_data:
                if self.preprocess_inputs:
                    obss = self.obs_preprocessor(obss).to(self.device)
                else:
                    obss = obss.to(self.device)
                acts = acts.to(self.device)

                mu, log_sigma = self.model(obss)
                batch_loss = gaussian_nll_loss(mu, log_sigma, acts)
                test_loss += batch_loss.item()
        
        avg_test_loss = test_loss / len(test_data)
        print(f"Test Loss: {avg_test_loss:.4f}")
        return avg_test_loss