from typing import Dict
import torch

from tqdm import tqdm
from skrl.resources.preprocessors.torch import RunningStandardScaler
from base_classes.models import MLE #, CNF
from base_classes.utils import gaussian_nll_loss, init_model_weights, print_model_summary, EarlyStopping




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

        self.preprocess_inputs = cfg.get("preprocess_inputs", True)
        if self.preprocess_inputs:
            self.obs_preprocessor = RunningStandardScaler(size=self.obs_dim, device=self.device)


        self.init_expert()

    def init_expert(self):
        self.lr = self.cfg["mle"]["learning_rate"]
        self.act_fct = self.cfg["mle"]["activation_fct"]
        self.hidden_sizes = self.cfg["mle"]["hidden_sizes"]

        self.model = MLE(self.obs_dim, self.action_dim, self.hidden_sizes, self.lr, self.act_fct).to(self.device)
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
                if early_stopping(val_loss, self.model):
                    # print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
                    # print(f"[INFO] Best validation loss: {early_stopping.best_loss:.6f}")
                    epoch_pbar.write(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
                    epoch_pbar.write(f"[INFO] Best validation loss: {early_stopping.best_loss:.6f}")
                    break
            
            self.model.scheduler.step(avg_epoch_loss)
        
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
    
    def test(self, test_data:torch.utils.data.DataLoader):
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