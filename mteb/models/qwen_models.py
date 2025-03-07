from functools import partial
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType, AudioEncoder
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, Qwen2AudioForConditionalGeneration, AutoProcessor
from mteb.model_meta import ModelMeta
from datasets import Audio

from functools import partial
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType, AudioEncoder
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, Qwen2AudioForConditionalGeneration, AutoProcessor
from mteb.model_meta import ModelMeta
from datasets import Audio
import os
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from tqdm import tqdm

class Qwen2AudioWrapper:
    def __init__(self, model_name: str, device: str | None = None, **kwargs):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")

        self.audio_encoder = self.model.audio_tower
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.audio_encoder = self.audio_encoder.to(self.device)

        if hasattr(self.model.config.audio_config, "d_model"):
            self.embed_dim = self.model.config.audio_config.d_model
        elif hasattr(self.model.config.audio_config, "hidden_size"):
            self.embed_dim = self.model.config.audio_config.hidden_size
        else:
            self.embed_dim = None

        print("Qwen2-Audio initialized. Hidden dim:", self.embed_dim)

    def get_audio_embeddings(
            self,
            audio_files: list[dict],
            labels: list[int],
            batch_size: int = 32,
            save_dir: str = "new_emotion_embeddings",
            **kwargs
    ) -> None:

        hidden_layer_percentages = [0.25, 0.5, 1]  # Extract these layers
        num_files = len(audio_files)

        # Initialize storage for embeddings and labels
        all_embeddings = {perc: [] for perc in hidden_layer_percentages}
        all_labels = {perc: [] for perc in hidden_layer_percentages}

        print(f"Processing {num_files} audio files with {len(labels)} labels...")
        # print(labels)

        for i in tqdm(range(0, num_files, batch_size), desc="Processing batches"):

            batch = audio_files[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            audio_data = [file['array'] for file in batch]
            sampling_rates = [file['sampling_rate'] for file in batch]

            prompt = " ".join(["<|AUDIO|>"] * len(batch))
            inputs = self.processor(
                text=prompt,
                audios=audio_data,
                sampling_rate=sampling_rates[0],
                return_tensors="pt",
                padding=True
            )

            input_features = inputs.input_features.to(self.device)

            with torch.no_grad():
                outputs = self.audio_encoder(input_features=input_features, output_hidden_states=True)

            num_hidden_states = len(outputs.hidden_states)

            for percentage in hidden_layer_percentages:
                layer_index = int(percentage * num_hidden_states) - 1  # Get correct layer index
                hidden_states = outputs.hidden_states[layer_index]

                batch_embeddings = hidden_states.mean(dim=1).cpu().numpy()
                all_embeddings[percentage].append(batch_embeddings)
                all_labels[percentage].extend(batch_labels)  # Store labels

        # Concatenate all batches and save embeddings
        for percentage, embeddings_list in all_embeddings.items():
            full_embeddings = np.vstack(embeddings_list)  # Stack all batches into (num_files, embed_dim)
            full_labels = np.array(all_labels[percentage])

            layer_folder = os.path.join(save_dir, "Qwen/Qwen2-Audio-7B", str(percentage))
            os.makedirs(layer_folder, exist_ok=True)

            # Save as .npz for easy loading
            save_path = os.path.join(layer_folder, "embeddings.npz")
            np.savez(save_path, embeddings=full_embeddings, labels=full_labels)
            print(f"Saved embeddings at {save_path} with shape {full_embeddings.shape} and labels shape {full_labels.shape}")

    def encode(self, audio_files: list[dict], *, task_name: str, prompt_type: str | None = None, **kwargs) -> None:
        self.get_audio_embeddings(audio_files, **kwargs)


qwen2_audio_meta = ModelMeta(
    loader=partial(Qwen2AudioWrapper, model_name="Qwen/Qwen2-Audio-7B"),
    name="Qwen/Qwen2-Audio-7B",
    languages=["multilingual"],
    open_weights=True,
    revision=None,
    release_date="2024-08-09",
    max_tokens=float("inf"),
    n_parameters=7_000_000_000,
    memory_usage_mb=None,
    embed_dim=1280,
    license="Unknown",
    reference="https://huggingface.co/Qwen/Qwen2-Audio-7B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)
