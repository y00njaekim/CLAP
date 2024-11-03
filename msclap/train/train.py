import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import wandb

from msclap.CLAPWrapper import CLAPWrapper
from custom_data_loader import CustomAudioTextDataset, collate_fn

from config import *
from trainer import CLAPTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLAP model")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay"
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of epochs"
    )
    parser.add_argument(
        "--is_augmented",
        type=bool,
        default=DEFAULT_IS_AUGMENTED,
        help="Use data augmentation",
    )
    parser.add_argument(
        "--config_version",
        type=str,
        default=DEFAULT_CONFIG_VERSION,
        help="Configuration version",
    )
    return parser.parse_args()


def get_config(args):
    return {
        "learning_rate": args.learning_rate,
        "architecture": "CLAP",
        "dataset": "CustomAudioTextDataset",
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "pad_token": args.pad_token,
        "is_augmented": args.is_augmented,
        "config_version": args.config_version,
    }


def setup_training(args, use_cuda):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    clap_wrapper = CLAPWrapper(
        version=args.config_version, use_cuda=use_cuda
    )
    clap_model = clap_wrapper.clap.to(device)

    train_dataset = CustomAudioTextDataset(AUDIO_FILES_DIR, TRAIN_CSV)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    validation_dataset = CustomAudioTextDataset(AUDIO_FILES_DIR, VALIDATION_CSV)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    optimizer = AdamW(
        clap_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler()

    return (
        device,
        clap_wrapper,
        clap_model,
        train_dataloader,
        validation_dataloader,
        optimizer,
        scaler,
    )


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    
    training_components = setup_training(args, use_cuda)
    device, clap_wrapper, clap_model, train_dataloader, validation_dataloader, optimizer, scaler = training_components
    
    trainer = CLAPTrainer(clap_model, optimizer, scaler, device)
    
    with wandb.init(project=PROJECT_NAME, config=get_config(args)):
        trainer.train(
            clap_wrapper,
            train_dataloader,
            validation_dataloader,
            num_epochs=args.num_epochs,
            args=args
        )
    wandb.finish()

if __name__ == "__main__":
    main()