"""Run script for InstaGeo model training, evaluation, and inference."""

import os
from functools import partial
from typing import Any, List, Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from instageo.data.datamodule import InstaGeoDataset, eval_collate_fn
from instageo.data.preprocessing import process_and_augment, process_test
from instageo.model.prithvi_seg import PrithviSeg

# Constants
BANDS = [1, 2, 3, 8, 11, 12]  # Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2
MEAN = [0.14245495, 0.13921481, 0.12434631, 0.31420089, 0.20743526, 0.12046503]
STD = [0.04036231, 0.04186983, 0.05267646, 0.0822221, 0.06834774, 0.05294205]
IM_SIZE = 224
TEMPORAL_SIZE = 1

log = pl._logger


def get_device() -> str:
    """Determine the device to use for training.

    Returns:
        str: The device to use ('gpu' or 'cpu').
    """
    return "gpu" if torch.cuda.is_available() else "cpu"


def check_required_flags(required_flags: List[str], cfg: DictConfig) -> None:
    """Check if required configuration flags are provided.

    Args:
        required_flags (List[str]): List of required configuration flags.
        cfg (DictConfig): Configuration object.

    Raises:
        SystemExit: If any required flag is missing.
    """
    missing_flags = [flag for flag in required_flags if getattr(cfg, flag) is None]
    if missing_flags:
        log.error(f"Missing required flags: {missing_flags}")
        exit(1)


def create_dataloader(
    dataset: InstaGeoDataset,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn: Optional[Any] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader.

    Args:
        dataset (InstaGeoDataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses to use for data loading.
        collate_fn (Optional[Callable]): Merges a list of samples to form a mini-batch.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned
            memory.

    Returns:
        DataLoader: An instance of the PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


class PrithviSegmentationModule(pl.LightningModule):
    """Prithvi Segmentation PyTorch Lightning Module."""

    def __init__(
        self,
        image_size: int = 224,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
        num_classes: int = 2,
        temporal_step: int = 1,
        class_weights: List[float] = [1, 2],
        ignore_index: int = -100,
        weight_decay: float = 1e-2,
    ) -> None:
        """Initialization.

        Initialize the PrithviSegmentationModule, a PyTorch Lightning module for image
        segmentation.

        Args:
            image_size (int): Size of input image.
            num_classes (int): Number of classes for segmentation.
            temporal_step (int): Number of temporal steps for multi-temporal input.
            learning_rate (float): Learning rate for the optimizer.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            class_weights (List[float]): Class weights for mitigating class imbalance.
            ignore_index (int): Class index to ignore during loss computation.
            weight_decay (float): Weight decay for L2 regularization.
        """
        super().__init__()
        self.net = PrithviSeg(
            image_size=image_size,
            num_classes=num_classes,
            temporal_step=temporal_step,
            freeze_backbone=freeze_backbone,
        )
        weight_tensor = torch.tensor(class_weights).float() if class_weights else None
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=weight_tensor
        )
        self.learning_rate = learning_rate
        self.ignore_index = ignore_index
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor for the model.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a training step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a validation step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure the optimizer for training.

        Returns:
            Optimizer: The optimizer for training.
        """
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run the model based on the configuration.

    Args:
        cfg (DictConfig): Configuration object containing all parameters.
    """
    seed_everything(42, workers=True)

    # Extract configuration values
    root_dir = cfg.root_dir
    train_filepath = cfg.train_filepath
    valid_filepath = cfg.valid_filepath
    test_filepath = cfg.test_filepath
    checkpoint_path = cfg.checkpoint_path
    batch_size = cfg.train.batch_size

    if cfg.mode == "train":
        check_required_flags(["root_dir", "train_filepath", "valid_filepath"], cfg)

        train_dataset = InstaGeoDataset(
            filename=train_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )

        valid_dataset = InstaGeoDataset(
            filename=valid_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )
        train_loader = create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        valid_loader = create_dataloader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )
        model = PrithviSegmentationModule(
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=cfg.dataloader.temporal_dim,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,
        )
        hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        checkpoint_callback = ModelCheckpoint(
            monitor="val_mIoU",
            dirpath=hydra_out_dir,
            filename="instageo_epoch-{epoch:02d}-val_iou-{val_mIoU:.2f}",
            auto_insert_metric_name=False,
            mode="max",
            save_top_k=3,
        )

        logger = TensorBoardLogger(hydra_out_dir, name="instageo")

        trainer = pl.Trainer(
            accelerator=get_device(),
            max_epochs=cfg.train.num_epochs,
            callbacks=[checkpoint_callback],
            logger=logger,
        )

        # run training and validation
        trainer.fit(model, train_loader, valid_loader)

    elif cfg.mode == "resume":
        check_required_flags(["root_dir", "train_filepath", "valid_filepath", "checkpoint_path"], cfg)

        train_dataset = InstaGeoDataset(
            filename=train_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )

        valid_dataset = InstaGeoDataset(
            filename=valid_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )

        train_loader = create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        valid_loader = create_dataloader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )

        # Load model from checkpoint with NEW hyperparameters
        model = PrithviSegmentationModule.load_from_checkpoint(
            checkpoint_path,
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,  # NEW learning rate
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=cfg.dataloader.temporal_dim,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,  # NEW weight decay
        )

        hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        checkpoint_callback = ModelCheckpoint(
            monitor="val_mIoU",
            dirpath=hydra_out_dir,
            filename="instageo_resumed_epoch-{epoch:02d}-val_iou-{val_mIoU:.2f}",
            auto_insert_metric_name=False,
            mode="max",
            save_top_k=3,
        )

        logger = TensorBoardLogger(hydra_out_dir, name="instageo_resumed")

        trainer = pl.Trainer(
            accelerator=get_device(),
            max_epochs=1,  # Only 1 epoch
            callbacks=[checkpoint_callback],
            logger=logger,
        )

        log.info(f"Resuming training from checkpoint: {checkpoint_path}")
        log.info(f"New learning rate: {cfg.train.learning_rate}")
        log.info(f"New weight decay: {cfg.train.weight_decay}")

        # Resume training
        trainer.fit(model, train_loader, valid_loader)

    elif cfg.mode == "eval":
        check_required_flags(["root_dir", "test_filepath", "checkpoint_path"], cfg)
        test_dataset = InstaGeoDataset(
            filename=test_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_test,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                img_size=cfg.test.img_size,
                crop_size=cfg.test.crop_size,
                stride=cfg.test.stride,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
            include_filenames=True,
        )
        test_loader = create_dataloader(
            test_dataset, batch_size=batch_size, collate_fn=eval_collate_fn
        )
        model = PrithviSegmentationModule.load_from_checkpoint(
            checkpoint_path,
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=cfg.dataloader.temporal_dim,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,
        )
        trainer = pl.Trainer(accelerator=get_device())
        result = trainer.test(model, dataloaders=test_loader)
        log.info(f"Evaluation results:\n{result}")

    elif cfg.mode == "sliding_inference":
        model = PrithviSegmentationModule.load_from_checkpoint(
            cfg.checkpoint_path,
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=cfg.dataloader.temporal_dim,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,
        )
        model.eval()
        # Add your sliding inference implementation here


if __name__ == "__main__":
    main()
