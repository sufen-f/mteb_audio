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
            batch_size: int = 32,
            save_dir: str = "embeddings",
            **kwargs
    ) -> None:
        
        hidden_layer_percentages = [0.25, 0.5, 1]  # Extract these layers
        num_files = len(audio_files)

        # Initialize dictionaries to store embeddings
        all_embeddings = {perc: [] for perc in hidden_layer_percentages}

        print(f"Processing {num_files} audio files...")

        from tqdm import tqdm

        for i in tqdm(range(0, num_files, batch_size), desc="Processing batches"):

            batch = audio_files[i:i + batch_size]
            audios = [file['array'] for file in batch]
            sr = batch[0]['sampling_rate']

            prompt = " ".join(["<|AUDIO|>"] * len(batch))
            inputs = self.processor(
                text=prompt,
                audios=audios,
                sampling_rate=sr,
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

        # Concatenate all batches and save embeddings
        for percentage, embeddings_list in all_embeddings.items():
            full_embeddings = np.vstack(embeddings_list)  # Stack all batches into (2048, embed_dim)
            layer_folder = os.path.join(save_dir, "Qwen/Qwen2-Audio-7B", str(percentage))
            os.makedirs(layer_folder, exist_ok=True)

            save_path = os.path.join(layer_folder, "embeddings.npy")
            np.save(save_path, full_embeddings)
            print(f"Saved embeddings at {save_path} with shape {full_embeddings.shape}")

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
