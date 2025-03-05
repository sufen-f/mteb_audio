from functools import partial
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType, AudioEncoder
import numpy as np
import torch
from transformers import WhisperModel, WhisperProcessor
from mteb.model_meta import ModelMeta
from datasets import Audio

import os
import numpy as np
import torch
from transformers import WhisperModel, WhisperProcessor

class WhisperWrapper:
    def __init__(self,
                 model_name: str,
                 revision: str = "main",
                 device: str | None = None,
                 **kwargs):
        self.model_name = model_name
        self.model_revision = revision
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.model = WhisperModel.from_pretrained(self.model_name, revision=self.model_revision).to(self.device)
        self.feature_extractor = WhisperProcessor.from_pretrained(self.model_name, revision=self.model_revision)
        self.embed_dim = self.model.config.d_model

        print("Whisper model initialized.")

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
            audio_data = [file['array'] for file in batch]
            sampling_rates = [file['sampling_rate'] for file in batch]

            # Converts raw waveform to log-Mel spectrograms
            inputs = self.feature_extractor(
                audio_data,
                sampling_rate=sampling_rates[0],
                return_tensors="pt",
                padding="max_length",
                max_length=480000  # 30 sec * 16000 Hz = 480000 samples
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.encoder(
                    inputs['input_features'],
                    output_hidden_states=True
                )

            num_hidden_states = len(outputs.hidden_states)

            for percentage in hidden_layer_percentages:
                layer_index = int(percentage * num_hidden_states) - 1  # Get correct layer index
                hidden_states = outputs.hidden_states[layer_index]

                batch_embeddings = hidden_states.mean(dim=1).cpu().numpy()
                all_embeddings[percentage].append(batch_embeddings)

        # Concatenate all batches and save embeddings
        for percentage, embeddings_list in all_embeddings.items():
            full_embeddings = np.vstack(embeddings_list)  # Stack all batches into (2048, embed_dim)
            layer_folder = os.path.join(save_dir, self.model_name, str(percentage))
            os.makedirs(layer_folder, exist_ok=True)

            save_path = os.path.join(layer_folder, "embeddings.npy")
            np.save(save_path, full_embeddings)
            print(f"Saved embeddings at {save_path} with shape {full_embeddings.shape}")

    def encode(self,
               audio_files: list[dict],
               *,
               task_name: str,
               prompt_type: str | None = None,
               **kwargs) -> None:
        self.get_audio_embeddings(audio_files, **kwargs)


whisper_tiny = ModelMeta(
    loader=partial(WhisperWrapper, model_name="openai/whisper-tiny"),
    name="openai/whisper-tiny",
    languages=["eng", "multilingual"],
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=39_000_000,      
    memory_usage_mb=144,         
    embed_dim=512,              
    license="MIT",
    reference="https://huggingface.co/openai/whisper-tiny",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)

whisper_base = ModelMeta(
    loader=partial(WhisperWrapper, model_name="openai/whisper-base"),
    name="openai/whisper-base",
    languages=["eng", "multilingual"],
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=74_000_000,      
    memory_usage_mb=277,          
    embed_dim=512,  
    license="MIT",
    reference="https://huggingface.co/openai/whisper-base",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)

whisper_small = ModelMeta(
    loader=partial(WhisperWrapper, model_name="openai/whisper-small"),
    name="openai/whisper-small",
    languages=["eng", "multilingual"],
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=244_000_000,    
    memory_usage_mb=922,        
    embed_dim=768,         
    license="MIT",
    reference="https://huggingface.co/openai/whisper-small",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)

whisper_medium = ModelMeta(
    loader=partial(WhisperWrapper, model_name="openai/whisper-medium"),
    name="openai/whisper-medium",
    languages=["eng", "multilingual"],
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=769_000_000,     
    memory_usage_mb=2914,     
    embed_dim=1024,      
    license="MIT",
    reference="https://huggingface.co/openai/whisper-medium",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)

whisper_large_v3 = ModelMeta(
    loader=partial(WhisperWrapper, model_name="openai/whisper-large-v3"),
    name="openai/whisper-large-v3",
    languages=["multilingual"],
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=1550_000_000,  
    memory_usage_mb=5887,   
    embed_dim=1280,           
    license="MIT",
    reference="https://huggingface.co/openai/whisper-large-v3",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)

# print(f"whisper_tiny: {whisper_tiny.calculate_memory_usage_mb()}")
# print(f"whisper_base: {whisper_base.calculate_memory_usage_mb()}")
# print(f"whisper_small: {whisper_small.calculate_memory_usage_mb()}")
# print(f"whisper_medium: {whisper_medium.calculate_memory_usage_mb()}")
# print(f"whisper_large_v3: {whisper_large.calculate_memory_usage_mb()}")
