import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

@dataclass
class LossWeights:
    """Configuration for loss weights"""
    adversarial: float = 0.0
    feature_matching: float = 0.0
    vgg19: float = 0.0
    gaze: float = 0.0
    l1_weight: float = 0.0
    vgg19_face: float = 0.0
    landmarks: float = 0.0
    warping_reg: float = 0.0
    cycle_idn: float = 0.0
    cycle_exp: float = 0.0
    l1_vol_rgb: float = 0.0
    barlow: float = 0.0
    
class LossCalculator:
    """Handles calculation of all model losses"""
    
    def __init__(self, weights: LossWeights, args):
        self.weights = weights
        self.args = args
        self.initialize_loss_functions()
        self.prev_targets = None
        self.expansion_factor = 1
        
    def initialize_loss_functions(self):
        """Initialize all loss function objects"""
        self.adversarial_loss = losses.AdversarialLoss() if self.weights.adversarial else None
        self.feature_matching_loss = losses.FeatureMatchingLoss() if self.weights.feature_matching else None
        self.vgg19_loss = losses.PerceptualLoss(
            num_scales=self.args.vgg19_num_scales, 
            use_fp16=False
        ) if self.weights.vgg19 else None
        
        # Basic losses
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.cosin_sim = torch.nn.CosineEmbeddingLoss(margin=0.3)
        self.cosin_sim_2 = torch.nn.CosineEmbeddingLoss(margin=0.5, reduce=False)
        self.cosin_dis = torch.nn.CosineSimilarity()

    def calc_discriminator_losses(
        self, 
        data_dict: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """Calculate discriminator losses"""
        losses = {}
        
        if self.weights.adversarial:
            losses['dis_adversarial'] = (
                self.weights.adversarial * 
                self.adversarial_loss(
                    real_scores=data_dict['real_score_dis'],
                    fake_scores=data_dict['fake_score_dis'],
                    mode='dis'
                )
            )

            if self.args.use_mix_dis and epoch >= self.args.dis2_train_start:
                losses['dis_adversarial_mix'] = (
                    self.weights.adversarial *
                    self.adversarial_loss(
                        real_scores=data_dict['real_score_dis_mix'],
                        fake_scores=data_dict['fake_score_dis_mix'],
                        mode='dis'
                    )
                )
                
        return losses

    def calc_generator_reconstruction_losses(
        self, 
        data_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate generator reconstruction losses"""
        losses = {}
        
        if self.weights.l1_weight:
            losses['L1'] = self.weights.l1_weight * self.l1_loss(
                data_dict['pred_target_img'],
                data_dict['target_img']
            )

        if self.weights.vgg19:
            losses['vgg19'], _ = self.vgg19_loss(
                data_dict['pred_target_img'], 
                data_dict['target_img'],
                None
            )
            losses['vgg19'] *= self.weights.vgg19
            
        return losses

    def calc_feature_losses(
        self, 
        data_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate feature-based losses"""
        losses = {}
        
        if self.weights.feature_matching:
            losses['feature_matching'] = (
                self.weights.feature_matching *
                self.feature_matching_loss(
                    real_features=data_dict['real_feats_gen'],
                    fake_features=data_dict['fake_feats_gen']
                )
            )
            
        return losses

    def calc_cycle_consistency_losses(
        self, 
        data_dict: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """Calculate cycle consistency losses"""
        losses = {}
        
        if self.weights.cycle_idn and data_dict['target_img'].shape[0] > 1:
            losses['vgg19_cycle_idn'], _ = self.vgg19_loss(
                data_dict['target_img'].detach(),
                data_dict['pred_identical_cycle'], 
                None
            )
            losses['vgg19_cycle_idn'] *= self.weights.cycle_idn
            
        return losses

    def calc_train_losses(
        self,
        data_dict: Dict[str, torch.Tensor], 
        mode: str = 'gen',
        epoch: int = 0,
        ffhq_per_b: int = 0,
        iteration: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate all training losses"""
        if mode == 'dis':
            losses = self.calc_discriminator_losses(data_dict, epoch)
        else:
            losses = {
                **self.calc_generator_reconstruction_losses(data_dict),
                **self.calc_feature_losses(data_dict),
                **self.calc_cycle_consistency_losses(data_dict, epoch)
            }
            
        total_loss = sum(losses.values())
        
        return total_loss, losses

    def calc_test_losses(
        self,
        data_dict: Dict[str, torch.Tensor],
        iteration: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Calculate test losses"""
        with torch.no_grad():
            losses = {
                'ssim': self.ssim(data_dict['pred_target_img'], data_dict['target_img']).mean(),
                'psnr': self.psnr(data_dict['pred_target_img'], data_dict['target_img']),
                'lpips': self.lpips(data_dict['pred_target_img'], data_dict['target_img'])
            }

            if self.args.image_size > 160:
                losses['ms_ssim'] = self.ms_ssim(
                    data_dict['pred_target_img'], 
                    data_dict['target_img']
                ).mean()
                
        return losses, None, None

def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Returns off-diagonal elements of a square matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def contrastive_loss(
    pos_dot: torch.Tensor,
    neg_dot: torch.Tensor,
    t: float = 0.35,
    m: float = 0.0,
    N: int = 1
) -> torch.Tensor:
    """Calculate contrastive loss between positive and negative samples"""
    a = torch.exp((pos_dot - m) / t) 
    b = torch.exp(neg_dot / t)
    loss = -torch.log(a / (a + torch.sum(b, dim=0))) / N
    return torch.sum(loss, dim=0)
