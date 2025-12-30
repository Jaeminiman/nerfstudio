# [CUSTOM] Semantic Splatfacto Model for NeRFStudio
# Extends SplatfactoModel to add per-Gaussian semantic features and semantic loss

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from torch.nn import Parameter

from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.utils.rich_utils import CONSOLE


def resize_image(image: torch.Tensor, d: int):
    """Downscale images using the same 'area' method in opencv"""
    import torch.nn.functional as tf
    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@dataclass
class SemanticSplatfactoModelConfig(SplatfactoModelConfig):
    """Semantic Splatfacto Model Config - extends base splatfacto with semantic learning"""
    
    _target: Type = field(default_factory=lambda: SemanticSplatfactoModel)
    semantic_loss_weight: float = 0.1
    """Weight for semantic cross-entropy loss"""
    semantic_dim: int = 16
    """Dimension of semantic feature embedding per Gaussian (before projection to num_classes)"""
    pass_semantic_gradients: bool = False
    """Whether to pass gradients through semantic rendering to Gaussian parameters"""


class SemanticSplatfactoModel(SplatfactoModel):
    """Semantic 3D Gaussian Splatting Model
    
    Extends SplatfactoModel to learn per-Gaussian semantic features.
    Each Gaussian has additional learnable semantic logits that are rendered
    via alpha-blending, similar to RGB colors.
    
    Args:
        config: SemanticSplatfactoModelConfig configuration
        metadata: Dictionary containing 'semantics' with class info
    """
    
    config: SemanticSplatfactoModelConfig
    
    def __init__(
        self,
        config: SemanticSplatfactoModelConfig,
        metadata: Dict,
        **kwargs,
    ) -> None:
        # Extract semantic information from metadata
        if "semantics" in metadata and isinstance(metadata["semantics"], Semantics):
            self.semantics = metadata["semantics"]
            self.num_semantic_classes = len(self.semantics.classes)
            self.colormap = self.semantics.colors.clone().detach()
            CONSOLE.print(f"[green]Semantic classes: {self.semantics.classes}[/green]")
        else:
            CONSOLE.print("[yellow]Warning: No semantic metadata found. Using fallback.[/yellow]")
            self.semantics = None
            self.num_semantic_classes = 2  # Fallback: background + foreground
            self.colormap = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float32)
        
        # Get semantic weights for loss balancing
        if "semantics_weights" in metadata:
            self.semantic_weights = metadata["semantics_weights"].clone().detach()
        else:
            self.semantic_weights = torch.ones(self.num_semantic_classes, dtype=torch.float32)
            
        super().__init__(config=config, **kwargs)
    
    def populate_modules(self):
        """Initialize modules including semantic features"""
        # Call parent to initialize base Gaussian parameters
        super().populate_modules()
        
        # Add semantic features to each Gaussian
        # Shape: [N_gaussians, num_semantic_classes] - direct logits
        num_points = self.means.shape[0]
        semantic_features = torch.nn.Parameter(
            torch.zeros(num_points, self.num_semantic_classes)
        )
        # Initialize with small random values to break symmetry
        torch.nn.init.normal_(semantic_features, mean=0.0, std=0.01)
        
        # Add to gauss_params
        self.gauss_params["semantic_features"] = semantic_features
        
        # Semantic loss - weights will be moved to device when loss is first called
        # Don't use self.device here as it may not be available yet
        self.semantic_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self._semantic_weights_initialized = False
        
        CONSOLE.print(f"[green]Initialized {num_points} Gaussians with {self.num_semantic_classes} semantic classes[/green]")
    
    @property
    def semantic_features(self):
        """Get semantic features for all Gaussians"""
        return self.gauss_params["semantic_features"]
    
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Override to add semantic features to optimizer groups"""
        groups = super().get_gaussian_param_groups()
        groups["semantic_features"] = [self.gauss_params["semantic_features"]]
        return groups
    
    def load_state_dict(self, dict, **kwargs):
        """Override to handle semantic features in checkpoint"""
        # Handle semantic features resize similar to other params
        if "gauss_params.semantic_features" in dict:
            newp = dict["gauss_params.means"].shape[0]
            old_semantic_shape = self.gauss_params["semantic_features"].shape
            new_semantic_shape = (newp, self.num_semantic_classes)
            self.gauss_params["semantic_features"] = torch.nn.Parameter(
                torch.zeros(new_semantic_shape, device=self.device)
            )
        super().load_state_dict(dict, **kwargs)
    
    def _render_semantics(
        self,
        semantic_features_crop: torch.Tensor,
        opacities_crop: torch.Tensor,
        means_crop: torch.Tensor,
        quats_crop: torch.Tensor,
        scales_crop: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        W: int,
        H: int,
    ) -> torch.Tensor:
        """Render semantic features using gsplat rasterization
        
        Uses the same rasterization as RGB but with semantic features as colors.
        Returns semantic logits per pixel.
        """
        try:
            from gsplat.rendering import rasterization
        except ImportError:
            raise ImportError("Please install gsplat>=1.0.0")
        
        # Semantic features as "colors" - shape [N, num_classes]
        # When sh_degree=None, gsplat expects colors in shape [N, C] directly
        semantic_colors = semantic_features_crop  # [N, num_classes]
        
        render_semantics, alpha, _ = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=semantic_colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",  # Use RGB mode for semantic features
            sh_degree=None,  # No SH for semantics
            sparse_grad=False,
            rasterize_mode=self.config.rasterize_mode,
        )
        
        # render_semantics shape: [1, H, W, num_classes]
        return render_semantics.squeeze(0)  # [H, W, num_classes]
    
    def get_outputs(self, camera) -> Dict[str, Union[torch.Tensor, List]]:
        """Override get_outputs to add semantic rendering"""
        # Get base outputs (RGB, depth, etc.)
        outputs = super().get_outputs(camera)
        
        if outputs.get("rgb") is None:
            # Empty output case
            return outputs
        
        # Get cropped Gaussian parameters (same as parent does)
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                # Add empty semantics
                H, W = outputs["rgb"].shape[:2]
                outputs["semantics"] = torch.zeros(H, W, self.num_semantic_classes, device=self.device)
                outputs["semantics_colormap"] = torch.zeros(H, W, 3, device=self.device)
                return outputs
        else:
            crop_ids = None
        
        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            semantic_features_crop = self.semantic_features[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            scales_crop = self.scales
            quats_crop = self.quats
            semantic_features_crop = self.semantic_features
        
        # Get camera matrices (repeat of parent logic, but needed for semantic render)
        if self.training:
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds
        
        from nerfstudio.models.splatfacto import get_viewmat
        
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        camera.rescale_output_resolution(camera_scale_fac)
        
        # Render semantics
        semantic_logits = self._render_semantics(
            semantic_features_crop,
            opacities_crop,
            means_crop,
            quats_crop,
            scales_crop,
            viewmat,
            K,
            W,
            H,
        )
        
        # Apply gradient control
        if not self.config.pass_semantic_gradients:
            semantic_logits = semantic_logits.detach()
        
        outputs["semantics"] = semantic_logits  # [H, W, num_classes]
        
        # Lazy init colormap to device if needed
        if not self._semantic_weights_initialized:
            self.colormap = self.colormap.to(self.device)
            weight = self.semantic_weights.to(self.device)
            self.semantic_loss = torch.nn.CrossEntropyLoss(weight=weight, reduction="mean")
            self._semantic_weights_initialized = True
        
        # Generate semantic colormap for visualization
        semantic_labels = torch.argmax(torch.softmax(semantic_logits, dim=-1), dim=-1)
        outputs["semantics_colormap"] = self.colormap[semantic_labels]  # [H, W, 3]
        
        return outputs
    
    def _get_gt_semantics(self, batch: Dict) -> torch.Tensor:
        """Get ground truth semantic labels from batch and resize if needed"""
        gt_semantics = batch["semantics"].to(self.device)  # [H, W, 1] containing class indices
        
        # Downscale if training with resolution schedule
        d = self._get_downscale_factor()
        if d > 1:
            # For semantic labels, use nearest neighbor to avoid interpolating class indices
            H, W = gt_semantics.shape[:2]
            new_H, new_W = H // d, W // d
            gt_semantics = gt_semantics.permute(2, 0, 1).unsqueeze(0).float()  # [1, 1, H, W]
            gt_semantics = torch.nn.functional.interpolate(
                gt_semantics, size=(new_H, new_W), mode='nearest'
            )
            gt_semantics = gt_semantics.squeeze(0).permute(1, 2, 0).long()  # [H', W', 1]
        
        return gt_semantics.squeeze(-1).long()  # [H, W]
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Override to add semantic loss"""
        # Get base losses
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        
        # Lazy initialization of semantic loss weights and colormap
        if not self._semantic_weights_initialized:
            weight = self.semantic_weights.to(self.device)
            self.semantic_loss = torch.nn.CrossEntropyLoss(weight=weight, reduction="mean")
            self.colormap = self.colormap.to(self.device)
            self._semantic_weights_initialized = True
        
        # Add semantic loss if semantics available
        if "semantics" in outputs and "semantics" in batch:
            pred_semantics = outputs["semantics"]  # [H, W, num_classes]
            gt_semantics = self._get_gt_semantics(batch)  # [H, W]
            
            # Resize gt to match pred if sizes differ (due to rounding in downscale)
            pred_H, pred_W = pred_semantics.shape[:2]
            gt_H, gt_W = gt_semantics.shape[:2]
            if pred_H != gt_H or pred_W != gt_W:
                gt_semantics = torch.nn.functional.interpolate(
                    gt_semantics.unsqueeze(0).unsqueeze(0).float(),
                    size=(pred_H, pred_W),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
            
            # Flatten for cross-entropy: pred [H*W, num_classes], gt [H*W]
            H, W, C = pred_semantics.shape
            pred_flat = pred_semantics.reshape(-1, C)
            gt_flat = gt_semantics.reshape(-1)
            
            semantic_loss = self.semantic_loss(pred_flat, gt_flat)
            loss_dict["semantic_loss"] = self.config.semantic_loss_weight * semantic_loss
        
        return loss_dict
    
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Override to add semantic accuracy metric"""
        metrics_dict = super().get_metrics_dict(outputs, batch)
        
        if "semantics" in outputs and "semantics" in batch:
            pred_semantics = outputs["semantics"]  # [H, W, num_classes]
            gt_semantics = self._get_gt_semantics(batch)  # [H', W']
            
            # Resize gt to match pred if sizes differ (due to rounding in downscale)
            pred_H, pred_W = pred_semantics.shape[:2]
            gt_H, gt_W = gt_semantics.shape[:2]
            if pred_H != gt_H or pred_W != gt_W:
                gt_semantics = torch.nn.functional.interpolate(
                    gt_semantics.unsqueeze(0).unsqueeze(0).float(),
                    size=(pred_H, pred_W),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
            
            # Compute accuracy
            pred_labels = torch.argmax(torch.softmax(pred_semantics, dim=-1), dim=-1)
            accuracy = (pred_labels == gt_semantics).float().mean()
            metrics_dict["semantic_accuracy"] = accuracy
        
        return metrics_dict
    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Override to add semantic visualization"""
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        
        # Add semantic colormap to images
        if "semantics_colormap" in outputs:
            images_dict["semantics"] = outputs["semantics_colormap"]
        
        # Add ground truth semantics visualization
        if "semantics" in batch:
            gt_semantics = self._get_gt_semantics(batch)
            gt_colormap = self.colormap[gt_semantics]
            images_dict["gt_semantics"] = gt_colormap
        
        # Add semantic accuracy to metrics
        if "semantics" in outputs and "semantics" in batch:
            pred_semantics = outputs["semantics"]
            gt_semantics = self._get_gt_semantics(batch)

            # Resize gt to match pred if sizes differ (due to rounding in downscale)
            pred_H, pred_W = pred_semantics.shape[:2]
            gt_H, gt_W = gt_semantics.shape[:2]
            if pred_H != gt_H or pred_W != gt_W:
                gt_semantics = torch.nn.functional.interpolate(
                    gt_semantics.unsqueeze(0).unsqueeze(0).float(),
                    size=(pred_H, pred_W),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
                
            pred_labels = torch.argmax(torch.softmax(pred_semantics, dim=-1), dim=-1)
            accuracy = (pred_labels == gt_semantics).float().mean()
            metrics_dict["semantic_accuracy"] = float(accuracy.item())
        
        return metrics_dict, images_dict
