

from functools import partial
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType, AudioEncoder
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
from mteb.model_meta import ModelMeta
from datasets import Audio
import torchaudio
import tempfile
import soundfile as sf


class TitanetWrapper(AudioEncoder):
    def __init__(
            self,
            model_name: str,
            device: str | None = None,
            **kwargs
    ):
        super().__init__(device=device, **kwargs)
        self.model_name = model_name
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(self.model_name)
        self.embed_dim = 192  # Titanet Large embedding size

        if device:
            self.model = self.model.to(device)
        print("Titanet initialized.")

    def get_audio_embeddings(
            self,
            audio_files: list[Audio] | Audio,
            batch_size: int = 32,
            **kwargs
    ) -> np.ndarray:

        if not isinstance(audio_files, list):
            audio_files = [audio_files]

        all_embeddings = []

        for file in audio_files:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                sf.write(temp_wav.name, file['array'], file['sampling_rate'])
                embedding = self.model.get_embedding(temp_wav.name).cpu().numpy()
                all_embeddings.append(embedding)

        return np.vstack(all_embeddings)

    def encode(
            self,
            audio_files: list[Audio],
            *,
            task_name: str,
            prompt_type: PromptType | None = None,
            **kwargs
    ) -> np.ndarray:
        return self.get_audio_embeddings(audio_files, **kwargs)


titanet_large = ModelMeta(
    loader=partial(TitanetWrapper, model_name="nvidia/speakerverification_en_titanet_large"),
    name="nvidia/speakerverification_en_titanet_large",
    languages=["multilingual"],
    open_weights=True,
    revision="main",
    release_date="2022-06-01",
    max_tokens=float("inf"),
    n_parameters=95_000_000,
    memory_usage_mb=512,
    embed_dim=192,
    license="Apache-2.0",
    reference="https://huggingface.co/nvidia/speakerverification_en_titanet_large",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)
