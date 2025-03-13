import math
import time
from typing import Optional, Tuple, Union
import warnings
from einops import rearrange
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers.models.depth_anything.modeling_depth_anything import DepthEstimatorOutput
import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
from math import ceil


class DepthAnythingWrapper(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")


    def da_forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions

        outputs = self.model.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.model.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        hidden_states = self.model.neck(hidden_states, patch_height, patch_width)

        predicted_depth = self.model.head(hidden_states, patch_height, patch_width)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        ), hidden_states[self.model.config.head_in_index]


    def forward(self, x, return_features=False):
        n, c, h, w = x.shape

        x = x - torch.tensor([[0.485, 0.456, 0.406]], device=x.device).view(1, 3, 1, 1)
        x = x / torch.tensor([[0.229, 0.224, 0.225]], device=x.device).view(1, 3, 1, 1)

        if h < w:
            size = (518, math.ceil(w * 518 / h / 14) * 14)
        else:
            size = (math.ceil(h * 518 / w / 14) * 14, 518)

        x = F.interpolate(x, size=size, mode="bicubic", align_corners=True)

        inputs = {
            "pixel_values": x,
        }

        outputs, features = self.da_forward(**inputs)

        predicted_depth = outputs.predicted_depth

        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="nearest",
            # align_corners=True,
        )

        if not return_features:
            return [prediction]
        else:
            return [prediction], features

    @classmethod
    def from_conf(cls, conf):
        return cls()


RESOLUTION_LEVELS = 10


def get_paddings(original_shape, aspect_ratio_range):
    # Original dimensions
    H_ori, W_ori = original_shape
    orig_aspect_ratio = W_ori / H_ori

    # Determine the closest aspect ratio within the range
    min_ratio, max_ratio = aspect_ratio_range
    target_aspect_ratio = min(max_ratio, max(min_ratio, orig_aspect_ratio))

    if orig_aspect_ratio > target_aspect_ratio:  # Too wide
        W_new = W_ori
        H_new = int(W_ori / target_aspect_ratio)
        pad_top = (H_new - H_ori) // 2
        pad_bottom = H_new - H_ori - pad_top
        pad_left, pad_right = 0, 0
    else:  # Too tall
        H_new = H_ori
        W_new = int(H_ori * target_aspect_ratio)
        pad_left = (W_new - W_ori) // 2
        pad_right = W_new - W_ori - pad_left
        pad_top, pad_bottom = 0, 0

    return (pad_left, pad_right, pad_top, pad_bottom), (H_new, W_new)


def get_resize_factor(original_shape, pixels_range, shape_multiplier=14):
    # Original dimensions
    H_ori, W_ori = original_shape
    n_pixels_ori = W_ori * H_ori

    # Determine the closest number of pixels within the range
    min_pixels, max_pixels = pixels_range
    target_pixels = min(max_pixels, max(min_pixels, n_pixels_ori))

    # Calculate the resize factor
    resize_factor = (target_pixels / n_pixels_ori) ** 0.5
    new_width = int(W_ori * resize_factor)
    new_height = int(H_ori * resize_factor)
    new_height = ceil(new_height / shape_multiplier) * shape_multiplier
    new_width = ceil(new_width / shape_multiplier) * shape_multiplier

    return resize_factor, (new_height, new_width)


def _postprocess(tensor, shapes, paddings, interpolation_mode="bilinear"):
    # interpolate to original size
    tensor = F.interpolate(
        tensor, size=shapes, mode=interpolation_mode, align_corners=False
    )

    # remove paddings
    pad1_l, pad1_r, pad1_t, pad1_b = paddings
    tensor = tensor[..., pad1_t : shapes[0] - pad1_b, pad1_l : shapes[1] - pad1_r]
    return tensor


def _postprocess_intrinsics(K, resize_factors, paddings):
    batch_size = K.shape[0]
    K_new = K.clone()

    for i in range(batch_size):
        scale = resize_factors[i]
        pad_l, _, pad_t, _ = paddings[i]

        K_new[i, 0, 0] /= scale  # fx
        K_new[i, 1, 1] /= scale  # fy
        K_new[i, 0, 2] /= scale  # cx
        K_new[i, 1, 2] /= scale  # cy

        K_new[i, 0, 2] -= pad_l  # cx
        K_new[i, 1, 2] -= pad_t  # cy

    return K_new


class UniDepthV2Wrapper(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        conf,
    ):
        super().__init__()

        version = conf.get("version", "v2")
        backbone = conf.get("backbone", "vits14")
        self.scaling = conf.get("scaling", 0.1)

        self.model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=version, backbone=backbone, pretrained=True, trust_repo=True)

    def forward(self, rgbs, return_features=False):
        n, c, h, w = rgbs.shape

        start_time = time.time()

        rgbs = rgbs - torch.tensor([[0.485, 0.456, 0.406]], device=rgbs.device).view(1, 3, 1, 1)
        rgbs = rgbs / torch.tensor([[0.229, 0.224, 0.225]], device=rgbs.device).view(1, 3, 1, 1)

        n, c, H, W = rgbs.shape


        ratio_bounds = self.model.shape_constraints["ratio_bounds"]
        pixels_bounds = [
            self.model.shape_constraints["pixels_min"],
            self.model.shape_constraints["pixels_max"],
        ]

        paddings, (padded_H, padded_W) = get_paddings((H, W), ratio_bounds)
        (pad_left, pad_right, pad_top, pad_bottom) = paddings
        resize_factor, (new_H, new_W) = get_resize_factor(
            (padded_H, padded_W), pixels_bounds
        )

        rgbs = F.pad(rgbs, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        rgbs = F.interpolate(
            rgbs, size=(new_H, new_W), mode="bilinear", align_corners=False
        )

        inputs={"image": rgbs, "camera": None}

        # run encoder
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            features, tokens = self.model.pixel_encoder(inputs["image"])
            inputs["features"] = [
                self.model.stacking_fn(features[i:j]).contiguous()
                for i, j in self.model.slices_encoder_range
            ]
            inputs["tokens"] = [
                self.model.stacking_fn(tokens[i:j]).contiguous()
                for i, j in self.model.slices_encoder_range
            ]

            outputs = self.model.pixel_decoder(inputs, [])
            outputs["rays"] = rearrange(outputs["rays"], "b (h w) c -> b c h w", h=new_H, w=new_W)
            pts_3d = outputs["rays"] * outputs["radius"]
            outputs.update({"points": pts_3d, "depth": pts_3d[:, -1:]})

        points = _postprocess(
            outputs["points"],
            (padded_H, padded_W),
            paddings=paddings,
            interpolation_mode=self.model.interpolation_mode,
        )

        depth = points[:, -1:]

        depth = depth * self.scaling

        depth = 1 / depth

        if not return_features:
            return [depth]
        else:
            return [depth], torch.zeros_like(depth)

    @classmethod
    def from_conf(cls, conf):
        return cls(conf)


class Metric3DV2Wrapper(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        conf,
    ):
        super().__init__()

        self.variant = conf.get("variant", "metric3d_vit_large")
        self.scaling = conf.get("scaling", 0.1)

        
        self.model = torch.hub.load('yvanyin/metric3d', self.variant, pretrain=True)

    def forward(self, rgbs, return_features=False):
        n, c, h, w = rgbs.shape

        start_time = time.time()

        rgbs = rgbs - torch.tensor([[0.485, 0.456, 0.406]], device=rgbs.device).view(1, 3, 1, 1)
        rgbs = rgbs / torch.tensor([[0.229, 0.224, 0.225]], device=rgbs.device).view(1, 3, 1, 1)

        n, c, H, W = rgbs.shape

        depth, confidence, output_dict = self.model.inference({'input': rgbs})

        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)

        depth = depth * self.scaling

        depth = 1 / depth

        if not return_features:
            return [depth]
        else:
            return [depth], torch.zeros_like(depth)

    @classmethod
    def from_conf(cls, conf):
        return cls(conf)
