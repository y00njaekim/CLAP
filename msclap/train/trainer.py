import torch
import wandb
from utils import process_batch
import torch.nn.functional as F
from config import *
import gc

class CLAPTrainer:
    def __init__(self, model, optimizer, scaler, device, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.scheduler = scheduler

    def train_epoch(self, clap_wrapper, dataloader):
        # 학습 모드로 설정
        self.model.train()
        total_loss = 0
        torch.cuda.empty_cache()
        gc.collect()
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                loss, _, _ = process_batch(
                    clap_wrapper,
                    self.device,
                    batch,
                )
                
                self.scaler.scale(loss).backward()
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                
            except RuntimeError as e:
                print(f"배치 {batch_idx}에서 오류 발생: {str(e)}")
                continue
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def validate(self, clap_wrapper, dataloader):
        self.model.eval()
        torch.cuda.empty_cache()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # base_classes = ["개", "수탉", "돼지", "소", "개구리", "고양이", "암탉", "곤충", "양", 
        #                "까마귀", "비", "바다 파도", "타오르는 불", "귀뚜라미", "새 지저귐", 
        #                "물방울", "바람", "물을 붓는 소리", "변기 물 내림", "뇌우", "우는 아기", 
        #                "재채기", "박수", "숨소리", "기침", "발소리", "웃음소리", "양치질", 
        #                "코골이", "마시기, 홀짝임", "문 두드림", "마우스 클릭", "키보드 타이핑", 
        #                "문, 나무 삐걱임", "캔 열기", "세탁기", "청소기", "알람 시계", 
        #                "시계 초침 소리", "유리 깨짐", "헬리콥터", "전기톱", "사이렌", 
        #                "자동차 경적", "엔진", "기차", "교회 종소리", "비행기", "불꽃놀이", "톱질"]
        # sound_prompts = [f"이것은 {cls} 소리입니다" for cls in base_classes]
        
        base_classes = ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow", "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm", "crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking", "helicopter", "chainsaw", "siren", "car_horn", "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw"]
        sound_prompts = [f"This is sound of {cls}" for cls in base_classes]
        fixed_text_embeddings = clap_wrapper.get_text_embeddings(sound_prompts)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                loss, audio_embeddings, _ = process_batch(
                    clap_wrapper,
                    self.device,
                    batch,
                )
                total_loss += loss.item()
                    
                similarities = clap_wrapper.compute_similarity(
                    audio_embeddings,
                    fixed_text_embeddings
                )
                predicted_indexes = similarities.argmax(axis=1)
                predicted_classes = [base_classes[idx] for idx in predicted_indexes]
                true_labels = batch["transcript"]
                
                for true_label, predicted_class in zip(true_labels, predicted_classes):
                    if true_label == predicted_class:
                        total_correct += 1
                    
                total_samples += len(true_labels)
                
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print(f"모델이 {model_path}에서 로드되었습니다.")

    def train(self, clap_wrapper, train_dataloader, validation_dataloader, 
              num_epochs, model_path=None):
        # 모델 경로가 제공되면 로드
        if model_path:
            self.load_model(model_path)


        for epoch in range(num_epochs):
            train_loss = self.train_epoch(
                clap_wrapper,
                train_dataloader,
            )

            val_loss, val_accuracy = self.validate(
                clap_wrapper,
                validation_dataloader
            )

            self.scheduler.step(val_loss)

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

            if epoch == num_epochs - 1 or epoch % 3 == 0:
                torch.save(
                    self.model.state_dict(),
                    f"{DATA_DIR}/params/model_epoch_{epoch+1}_{wandb.run.name}.pth",
                )

        return self.model