import json
import torch
import torch.nn.functional as F
import re

def process_batch(clap_wrapper, device, batch):
    try:
        waveforms = batch["waveform"].to(device)
        audio_files = batch["audio_path"]
        transcripts = batch["transcript"]

        with torch.cuda.amp.autocast():
            audio_embeddings = clap_wrapper.get_audio_embeddings(audio_files, resample=True)
            text_embeddings = clap_wrapper.get_text_embeddings(transcripts)

            similarity = clap_wrapper.compute_similarity(audio_embeddings, text_embeddings)
            loss = calculate_contrastive_loss(similarity)

        # 계산 완료 후
        del waveforms
        torch.cuda.empty_cache()
        
        return loss, audio_embeddings, text_embeddings
    finally:
        # 모든 중간 텐서들 명시적 해제
        torch.cuda.empty_cache()

def calculate_contrastive_loss(similarity_matrix):
    similarity = similarity_matrix.float()
    labels = torch.arange(similarity.shape[0], device=similarity.device)
    
    l_text = F.cross_entropy(similarity, labels)
    
    l_audio = F.cross_entropy(similarity.t(), labels)
    
    return 0.5 * (l_text + l_audio)