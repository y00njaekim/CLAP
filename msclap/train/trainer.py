import torch
import wandb
from utils import process_batch
import torch.nn.functional as F
from config import *

class CLAPTrainer:
    def __init__(self, model, optimizer, scaler, device):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device

    def train_epoch(self, clap_wrapper, dataloader):
        total_loss = 0
        torch.cuda.empty_cache()

        for batch in dataloader:
            loss, _, _ = process_batch(
                clap_wrapper,
                self.device,
                batch,
            )

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def validate(self, clap_wrapper, dataloader):
        torch.cuda.empty_cache()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                loss, audio_embeddings, text_embeddings = process_batch(
                    clap_wrapper,
                    self.device,
                    batch,
                )
                total_loss += loss.item()

                # Accuracy calculation
                batch_size = audio_embeddings.shape[0]
                for i in range(batch_size):
                    sample_audio_embedding = audio_embeddings[i].unsqueeze(0)
                    similarities = F.cosine_similarity(
                        sample_audio_embedding, text_embeddings
                    )
                    predicted_index = similarities.argmax().item()
                    if predicted_index == i:
                        total_correct += 1
                total_samples += batch_size

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def train(self, clap_wrapper, train_dataloader, validation_dataloader, 
              num_epochs):

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(
                clap_wrapper,
                train_dataloader,
            )

            val_loss, val_accuracy = self.validate(
                clap_wrapper,
                validation_dataloader
            )

            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy:.4f}"
            )

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )

            if epoch == num_epochs - 1:
                torch.save(
                    self.model.state_dict(),
                    f"{DATA_DIR}/params/model_epoch_{epoch+1}_{wandb.run.name}.pth",
                )

        return self.model