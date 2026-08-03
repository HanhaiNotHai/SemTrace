from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from semtrace.backbones.dinov3 import DINOv3Backbone
from semtrace.data.collate import ImageBatch, collate_image_samples
from semtrace.data.forensynths import ForenSynthsDataset
from semtrace.data.genimage import GenImageDataset
from semtrace.data.manifest import ManifestImageDataset
from semtrace.data.self_synthesis import SelfSynthesisDataset
from semtrace.data.transforms import ProtocolTransform
from semtrace.models.baselines import BaselineMode, FrozenFeatureLinearBaseline
from semtrace.models.normal_predictor import (
    MultiScaleNormalPredictors,
    NormalFeaturePredictor,
)
from semtrace.models.semantic_anchor import FrozenSemanticAnchor
from semtrace.models.semtrace import SemTrace


def read_selected_layers(path: str | Path) -> tuple[int, int, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = tuple(int(layer) for layer in payload["selected_layers"])
    if len(layers) != 3 or len(set(layers)) != 3:
        raise ValueError("selected_layers.json must contain three distinct layer indices")
    return layers[0], layers[1], layers[2]


def build_backbone(config: DictConfig, selected_layers: tuple[int, ...]) -> DINOv3Backbone:
    return DINOv3Backbone.from_pretrained(
        str(config.model.model_path or config.model.model_id),
        selected_layers,
        model_id=str(config.model.model_id),
        revision=str(config.model.revision),
        local_files_only=bool(config.model.local_files_only),
    )


def build_semantic_anchor(config: DictConfig, backbone_dim: int) -> FrozenSemanticAnchor:
    return FrozenSemanticAnchor(
        backbone_dim=backbone_dim,
        semantic_dim=int(config.model.semantic_dim),
        seed=int(config.model.semantic_projection.seed),
    )


def load_semantic_anchor(config: DictConfig, backbone_dim: int) -> FrozenSemanticAnchor:
    anchor = build_semantic_anchor(config, backbone_dim)
    checkpoint_path = Path(str(config.probe.semantic_anchor_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"semantic anchor checkpoint does not exist: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    anchor.load_state_dict(state)
    return anchor


def build_normal_predictors(
    config: DictConfig,
    selected_layers: tuple[int, ...],
    feature_dim: int,
) -> nn.ModuleDict:
    predictor_config = config.normal_predictor
    return nn.ModuleDict(
        {
            str(layer): NormalFeaturePredictor(
                input_dim=feature_dim,
                semantic_dim=int(config.model.semantic_dim),
                hidden_dim=int(predictor_config.hidden_dim),
                num_heads=int(predictor_config.num_heads),
                num_layers=int(predictor_config.num_layers),
                neighborhood_size=int(predictor_config.neighborhood_size),
                dropout=float(predictor_config.dropout),
            )
            for layer in selected_layers
        }
    )


def build_normal_predictor_collection(
    config: DictConfig,
    selected_layers: tuple[int, ...],
    feature_dim: int,
) -> MultiScaleNormalPredictors:
    return MultiScaleNormalPredictors(
        build_normal_predictors(config, selected_layers, feature_dim)
    )


def build_detection_model(
    config: DictConfig,
    *,
    backbone: DINOv3Backbone,
    semantic_anchor: FrozenSemanticAnchor,
    selected_layers: tuple[int, ...],
    normal_predictors: nn.ModuleDict | None,
) -> nn.Module:
    baseline = str(config.model_options.baseline)
    if baseline == "semtrace":
        if bool(config.model_options.semantic_direct_classifier):
            raise ValueError("semantic_direct_classifier is prohibited in SemTrace")
        return SemTrace(
            backbone=backbone,
            semantic_anchor=semantic_anchor,
            selected_layers=selected_layers,
            feature_dim=backbone.hidden_size,
            semantic_dim=int(config.model.semantic_dim),
            trace_dim=int(config.trace_adapter.hidden_dim),
            normal_predictors=normal_predictors,
            use_normal_predictor=bool(config.model_options.use_normal_predictor),
            use_cross_attention=bool(config.model_options.use_cross_attention),
            cross_attention_heads=int(config.cross_attention.num_heads),
            cross_attention_dropout=float(config.cross_attention.dropout),
            max_semantic_gate=float(config.cross_attention.max_semantic_gate),
        )
    if baseline not in {"final_cls", "intermediate_patch_mean"}:
        raise ValueError(f"unsupported model_options.baseline: {baseline}")
    if not bool(config.model_options.semantic_direct_classifier):
        raise ValueError(
            "diagnostic linear baselines require semantic_direct_classifier=true"
        )
    mode = cast(BaselineMode, baseline)
    return FrozenFeatureLinearBaseline(
        backbone,
        feature_dim=backbone.hidden_size,
        mode=mode,
        layer=selected_layers[0] if mode == "intermediate_patch_mean" else None,
    )


def build_dataset(
    config: DictConfig,
    *,
    split: str | None,
    training: bool,
    manifest_path: str | Path | None = None,
    real_only: bool = False,
) -> Dataset[Any]:
    configured_manifest = (
        manifest_path
        if manifest_path is not None
        else config.data.train_manifest
        if split == "train"
        else config.data.validation_manifest
    )
    if configured_manifest is None:
        raise ValueError(f"data manifest for split '{split}' must be configured")
    small_image_policy = str(config.preprocessing.small_image_policy)
    if small_image_policy not in {"skip", "reflect"}:
        raise ValueError("small_image_policy must be 'skip' or 'reflect'")
    transform = ProtocolTransform(
        crop_size=int(config.preprocessing.crop_size),
        training=training,
        small_image_policy=cast(Literal["skip", "reflect"], small_image_policy),
    )
    adapter: type[ManifestImageDataset]
    if str(config.protocol.name) == "forensynths_progan4":
        adapter = ForenSynthsDataset
    elif str(config.protocol.name) == "genimage_sdv14":
        adapter = GenImageDataset
    elif str(config.protocol.name) == "self_synthesis":
        adapter = SelfSynthesisDataset
    else:
        adapter = ManifestImageDataset
    dataset = adapter(
        configured_manifest,
        transform,
        split=split,
        data_root=config.data.root,
    )
    if not real_only:
        return dataset
    real_indices = [
        index for index, record in enumerate(dataset.records) if record.label == 0
    ]
    if not real_indices:
        raise ValueError("stage 2 manifest contains no real training samples")
    return Subset(dataset, real_indices)


def build_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    num_workers: int,
    training: bool,
    world_size: int,
    rank: int,
) -> tuple[DataLoader[ImageBatch], DistributedSampler[Any] | None]:
    sampler: DistributedSampler[Any] | None = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
            drop_last=training,
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        collate_fn=collate_image_samples,
    )
    return loader, sampler
