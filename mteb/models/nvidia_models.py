from __future__ import annotations

import logging
from functools import partial

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper

logger = logging.getLogger(__name__)


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return f"Instruct: {instruction}\nQuery: " if instruction else ""


nvidia_training_datasets = {
    # source: https://arxiv.org/pdf/2405.17428
    "ArguAna": ["train"],
    "ArguAna-PL": ["train"],
    "ArguAna-NL": ["train"],  # translation not trained on
    "NanoArguAnaRetrieval": ["train"],
    "HotpotQA": ["train"],
    "HotpotQA-PL": ["train"],  # translation not trained on
    "HotpotQA-NL": ["train"],  # translation not trained on
    "HotpotQAHardNegatives": ["train"],
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "MSMARCO-PL": ["train"],  # translation not trained on
    "mMARCO-NL": ["train"],  # translation not trained on
    "NQ": ["train"],
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    "NQ-PL": ["train"],  # translation not trained on
    "NQ-NL": ["train"],  # translation not trained on
    "FEVER": ["train"],
    "FEVER-NL": ["train"],  # translation not trained on
    "FEVERHardNegatives": ["train"],
    "NanoFEVERRetrieval": ["train"],
    "FiQA2018": ["train"],
    "FiQA2018-PL": ["train"],  # translation not trained on
    "FiQA2018-NL": ["train"],  # translation not trained on
    "STS12": ["train"],
    "STS22": ["train"],
    "AmazonReviewsClassification": ["train"],
    "AmazonCounterfactualClassification": ["train"],
    "Banking77Classification": ["train"],
    "EmotionClassification": ["train"],
    "ImdbClassification": ["train"],
    "MTOPIntentClassification": ["train"],
    "ToxicConversationsClassification": ["train"],
    "TweetSentimentExtractionClassification": ["train"],
    "ArxivClusteringP2P": ["train"],
    "ArxivClusteringP2P.v2": ["train"],
    "ArxivClusteringS2S": ["train"],
    "ArxivClusteringS2S.v2": ["train"],
    "BiorxivClusteringP2P": ["train"],
    "BiorxivClusteringP2P.v2": ["train"],
    "BiorxivClusteringS2S": ["train"],
    "BiorxivClusteringS2S.v2": ["train"],
    "MedrxivClusteringP2P": ["train"],
    "MedrxivClusteringP2P.v2": ["train"],
    "MedrxivClusteringS2S": ["train"],
    "MedrxivClusteringS2S.v2": ["train"],
    "TwentyNewsgroupsClustering": ["train"],
    "TwentyNewsgroupsClustering.v2": ["train"],
    "STSBenchmark": ["train"],
    "STSBenchmarkMultilingualSTS": ["train"],  # translated, not trained on
}

NV_embed_v2 = ModelMeta(
    loader=partial(  # type: ignore
        InstructSentenceTransformerWrapper,
        model="nvidia/NV-Embed-v2",
        revision="7604d305b621f14095a1aa23d351674c2859553a",
        instruction_template=instruction_template,
        trust_remote_code=True,
        max_seq_length=32768,
        padding_side="right",
        # for nv-embed, we add eos token to each input example
        add_eos_token=True,
    ),
    name="nvidia/NV-Embed-v2",
    languages=["eng_Latn"],
    open_weights=True,
    revision="7604d305b621f14095a1aa23d351674c2859553a",
    release_date="2024-09-09",  # initial commit of hf model.
    n_parameters=7_850_000_000,
    memory_usage_mb=14975,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/nvidia/NV-Embed-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=nvidia_training_datasets,
    public_training_code=None,
    public_training_data=None,
)

NV_embed_v1 = ModelMeta(
    loader=partial(  # type: ignore
        InstructSentenceTransformerWrapper,
        model="nvidia/NV-Embed-v1",
        revision="7604d305b621f14095a1aa23d351674c2859553a",
        instruction_template=instruction_template,
        trust_remote_code=True,
        max_seq_length=32768,
        padding_side="right",
        # for nv-embed, we add eos token to each input example
        add_eos_token=True,
    ),
    name="nvidia/NV-Embed-v1",
    languages=["eng_Latn"],
    open_weights=True,
    revision="570834afd5fef5bf3a3c2311a2b6e0a66f6f4f2c",
    release_date="2024-09-13",  # initial commit of hf model.
    n_parameters=7_850_000_000,
    memory_usage_mb=29945,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/nvidia/NV-Embed-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=nvidia_training_datasets,
    public_training_code=None,
    public_training_data=None,
)

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
