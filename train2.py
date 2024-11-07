from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib
import os
import ssl
import time
from PIL import Image
from torchvision import transforms
import warnings
from aitypes import NormLayerType

os.environ["WANDB_SILENT"] = "True"
warnings.filterwarnings("ignore")



@dataclass
class TrainerConfig:
    """Configuration for training process"""
    # Paths
    project_dir: Path = "/media/oem/12TB/EMOPortraits"
    experiment_name: str = "volumetric_avatar"
    model_checkpoint: Optional[str] = None
    trainer_checkpoint: Optional[str] = None
    
    # Training settings
    num_gpus: int = 1
    max_epochs: int = 200
    random_seed: int = 0
    use_amp: bool = False
    normalize_losses: bool = False
    
    # Logging
    checkpoint_freq: int = 10
    latest_checkpoint_freq: int = 1
    logging_freq: int = 50
    visuals_freq: int = 500
    output_visuals: bool = True
    
    # Dataset settings
    dataset_name: str = "voxceleb2hq_pairs"
    dataset_name_test: str = "voxceleb2hq_pairs"
    use_sec_dataset: bool = False
    sec_dataset_every: int = 2
    
    # Model settings
    model_type: str = "volumetric_avatar"
    model_name: str = "va2"
    image_size: int = 512
    enc_channel_mult: int = 4
    norm_layer_type:NormLayerType= NormLayerType.GROUP_NORM
    dec_max_channels: int = 512
    gen_latent_texture_channels: int = 96

    use_mix_mask: bool = True

class Trainer:
    """Main trainer class for volumetric avatar model"""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.setup_environment()
        self.initialize_directories()
        self.setup_model()
        self.setup_dataloaders()
        self.setup_logging()

    def setup_environment(self):
        """Setup training environment"""
        ssl._create_default_https_context = ssl._create_unverified_context
        warnings.filterwarnings("ignore")
        os.environ["WANDB_SILENT"] = "True"
        
        # Set random seeds
        torch.manual_seed(self.config.random_seed)
        if self.config.num_gpus > 0:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed_all(self.config.random_seed)
            
        # Setup distributed training
        self.setup_distributed()

    def setup_distributed(self):
        """Setup distributed training if using multiple GPUs"""
        if self.config.num_gpus <= 1:
            self.rank = 0
        elif self.config.num_gpus <= 8:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
            self.rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.rank)
        else:
            raise ValueError("More than 8 GPUs not supported")

    def initialize_directories(self):
        """Setup experiment directories"""
        self.project_dir = Path(self.config.project_dir)
        self.experiment_dir = self.project_dir / 'logs' / self.config.experiment_name
        self.checkpoints_dir = self.experiment_dir / 'checkpoints'
        self.exp_dir = self.experiment_dir / 'expression_vectors'  
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        if self.exp_dir:
            os.makedirs(self.exp_dir, exist_ok=True)


    def setup_model(self):
        """Initialize and setup the model"""
        model_module = importlib.import_module(
            f'models.stage_1.{self.config.model_type}.{self.config.model_name}'
        )
        self.model = model_module.Model(self.config, rank=self.rank, exp_dir=self.exp_dir)
        
        if self.config.num_gpus > 0:
            self.model.cuda()
            
        self.load_checkpoints()
        
        # Initialize optimizers and schedulers
        self.opts = self.model.configure_optimizers()
        self.shds, self.shd_max_iters = self.model.configure_schedulers(
            self.opts,
            epochs=self.config.max_epochs,
            steps_per_epoch=len(self.train_dataloader)
        )

    def setup_dataloaders(self):
        """Initialize training and testing dataloaders"""
        # Main dataset
        data_module = importlib.import_module(f'datasets.{self.config.dataset_name}').DataModule(self.config)
        self.train_dataloader = data_module.train_dataloader()
        
        test_module = importlib.import_module(f'datasets.{self.config.dataset_name_test}').DataModule(self.config)
        self.test_dataloader = test_module.test_dataloader()
        
        # Secondary dataset if needed
        if self.config.use_sec_dataset:
            self.setup_secondary_dataloaders()

    def setup_logging(self):
        """Initialize logging"""
        self.logger = Logger(
            self.config,
            self.experiment_dir,
            self.rank,
            self.model,
            project_name=self.config.project_name,
            entity=self.config.entity
        )
        self.to_tensor = transforms.ToTensor()

    def train(self):
        """Main training loop"""
        for epoch in range(self.logger.epoch, self.config.max_epochs):
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            self.save_checkpoints(epoch)

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        if self.rank == 0:
            train_iterator = tqdm(self.train_dataloader)
        else:
            train_iterator = self.train_dataloader
            
        if self.config.use_sec_dataset:
            self.prepare_secondary_iterators(epoch)
            
        for i, data_dict in enumerate(train_iterator):
            if self.config.use_sec_dataset:
                data_dict = self.maybe_use_secondary_data(data_dict, i)
                
            losses_dict, visuals = self.training_step(data_dict, epoch=epoch, iteration=i)
            self.log_training_step(losses_dict, visuals, i)

    def training_step(
        self, 
        data_dict: Dict[str, torch.Tensor], 
        epoch: int,
        iteration: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Perform single training step"""
        output_visuals = self.logger.output_train_visuals and self.config.output_visuals
        losses_dict = {}
        visuals = torch.empty(0)

        for opt_idx, optimizer in enumerate(self.opts):
            optimizer.zero_grad()
            
            loss, step_losses, step_visuals, updated_data = self.model(
                data_dict,
                phase='train',
                optimizer_idx=opt_idx,
                visualize=output_visuals and opt_idx == 0,
                iteration=iteration,
                rank=self.rank,
                epoch=epoch
            )
            
            losses_dict.update(step_losses)
            if opt_idx == 0 and step_visuals is not None:
                visuals = step_visuals

            if self.config.use_amp:
                with amp.scale_loss(loss, optimizer, loss_id=opt_idx) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
            optimizer.step()

        self.step_schedulers()
        
        return losses_dict, visuals if len(visuals) else None

    def test_epoch(self, epoch: int):
        """Test for one epoch"""
        self.model.eval()
        
        if self.rank == 0:
            test_iterator = tqdm(self.test_dataloader) 
        else:
            test_iterator = self.test_dataloader
            
        test_visuals = None
        
        for i, data_dict in enumerate(test_iterator):
            with torch.no_grad():
                test_visuals = self.test_step(data_dict, i==0, i, epoch)
                
        self.logger.log(
            'test',
            visuals=test_visuals,
            epoch_end=True,
            explaining_var=getattr(self, 'expl_var', None)
        )

    def save_checkpoints(self, epoch: int):
        """Save model and trainer checkpoints"""
        if self.rank == 0 and (epoch % self.config.checkpoint_freq == 0 or 
                              epoch % self.config.latest_checkpoint_freq == 0):
            
            self.save_model_checkpoint(epoch)
            self.save_trainer_checkpoint(epoch)
            self.cleanup_old_checkpoints(epoch)

    def save_model_checkpoint(self, epoch: int):
        """Save model state"""
        model = self.model.module if self.config.num_gpus > 1 else self.model
        torch.save(
            model.state_dict(),
            self.checkpoints_dir / f'{epoch:03d}_model.pth'
        )

    def save_trainer_checkpoint(self, epoch: int):
        """Save training state"""
        trainer_state = {
            'logger': self.logger.state_dict()
        }
        
        # Save optimizer states
        for i, opt in enumerate(self.opts):
            trainer_state[f'opt_{i}'] = opt.state_dict()
            
        # Save scheduler states
        if self.shds:
            for i, shd in enumerate(self.shds):
                trainer_state[f'shd_{i}'] = shd.state_dict()
                
        torch.save(
            trainer_state,
            self.checkpoints_dir / f'{epoch:03d}_trainer.pth'
        )

    def cleanup_old_checkpoints(self, epoch: int):
        """Remove old checkpoints"""
        if epoch > 1:
            prev_epoch = epoch - 1
            if prev_epoch % self.config.checkpoint_freq != 0:
                try:
                    os.remove(self.checkpoints_dir / f'{prev_epoch:03d}_model.pth')
                    os.remove(self.checkpoints_dir / f'{prev_epoch:03d}_trainer.pth')
                except:
                    print('Previous checkpoints not found')




if __name__ == "__main__":
    cfg = TrainerConfig()
    trainer = Trainer(cfg)
    trainer.train()