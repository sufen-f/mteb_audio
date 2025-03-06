from functools import partial
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType, AudioEncoder
import numpy as np
import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from mteb.model_meta import ModelMeta
from datasets import Audio
import os
import numpy as np
import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

class WavlmWrapper:
    def __init__(self,
        model_name: str, 
        revision: str = "main", 
        device: str | None = None, 
        **kwargs
    ):
        self.model_name = model_name
        self.model_revision = revision
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.model = WavLMModel.from_pretrained(
            self.model_name, 
            revision=self.model_revision
        ).to(self.device)
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, 
            revision=self.model_revision
        )
        
        self.embed_dim = self.model.config.hidden_size
        
        print("WavLM initialized.")


    def get_audio_embeddings(
            self,
            audio_files: list[dict],
            labels: list[int],
            batch_size: int = 32,
            save_dir: str = "new_gender_embeddings",
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

            # Preprocess batch
            inputs = self.feature_extractor(
                audio_data,
                sampling_rate=sampling_rates[0],
                padding=True,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_values=inputs["input_values"],
                    output_hidden_states=True,
                    return_dict=True
                )

            num_hidden_states = len(outputs.hidden_states)

            for percentage in hidden_layer_percentages:
                layer_index = int(percentage * num_hidden_states) - 1  # Get correct layer index
                hidden_states = outputs.hidden_states[layer_index]

                batch_embeddings = hidden_states.mean(dim=1).cpu().numpy()
                all_embeddings[percentage].append(batch_embeddings)
                all_labels[percentage].extend(batch_labels)  # Store labels

        # Save embeddings and labels
        for percentage, embeddings_list in all_embeddings.items():
            full_embeddings = np.vstack(embeddings_list)  # Stack all batches into (num_files, embed_dim)
            full_labels = np.array(all_labels[percentage])

            layer_folder = os.path.join(save_dir, self.model_name, str(percentage))
            os.makedirs(layer_folder, exist_ok=True)

            # Save as .npz for easy loading
            save_path = os.path.join(layer_folder, "embeddings.npz")
            np.savez(save_path, embeddings=full_embeddings, labels=full_labels)
            print(f"Saved embeddings at {save_path} with shape {full_embeddings.shape} and labels shape {full_labels.shape}")

    def encode(
        self,
        audio_files: list[dict],
        *,
        task_name: str,
        prompt_type: str | None = None,
        **kwargs
    ) -> None:
        self.get_audio_embeddings(audio_files, **kwargs)



wavlm_base = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base"),
    name="microsoft/wavlm-base",
    languages=["eng"],
    open_weights=True,
    revision="main",         
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,    
    modalities=["audio"]
)

wavlm_base_sd = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base-sd"),
    name="microsoft/wavlm-base-sd",
    languages=["eng"],
    open_weights=True,
    revision="main",         
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base-sd",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,    
    modalities=["audio"]
)
# print(f"wavlm_base: {wavlm_base.calculate_memory_usage_mb()}")

wavlm_base_plus = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base-plus"),
    name="microsoft/wavlm-base-plus",
    languages=["eng"],
    open_weights=True,
    revision="main",         
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base-plus",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,    
    modalities=["audio"]
)

# print(f"wavlm_base_plus: {wavlm_base_plus.calculate_memory_usage_mb()}")

wavlm_base_plus_sv = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base-plus-sv"),
    name="microsoft/wavlm-base-plus-sv",
    languages=["eng"],
    open_weights=True,
    revision="main",         
    release_date="2022-07-19", # estimate
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base-plus-sv",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,    
    modalities=["audio"]
)

wavlm_base_plus_sd = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base-plus-sd"),
    name="microsoft/wavlm-base-plus-sd",
    languages=["eng"],
    open_weights=True,
    revision="main",         
    release_date="2022-07-19", # estimate
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base-plus-sd",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,    
    modalities=["audio"]
)

# print(f"wavlm_base_plus_sv: {wavlm_base_plus_sv.calculate_memory_usage_mb()}")

wavlm_base_sv = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base-sv"),
    name="microsoft/wavlm-base-sv",
    languages=["eng"],
    open_weights=True,
    revision="main",         
    release_date="2022-07-19", # estimate
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base-sv",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,    
    modalities=["audio"]
)

# print(f"wavlm_base_sv: {wavlm_base_sv.calculate_memory_usage_mb()}")

wavlm_large = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-large"),
    name="microsoft/wavlm-large",
    languages=["eng"],
    open_weights=True,
    revision="main",         
    release_date="2022-07-19", # estimate
    max_tokens=float("inf"),
    n_parameters=316_620_000,
    memory_usage_mb=1208,
    embed_dim=1024,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-large",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,    
    modalities=["audio"]
)

# print(f"wavlm_large: {wavlm_large.calculate_memory_usage_mb()}")