from mteb.abstasks.Audio.AbsTaskAudioClassification import AbsTaskAudioClassification
from mteb.abstasks.TaskMetadata import TaskMetadata
import random
import datasets
import mteb
from mteb import MTEB
from sklearn.model_selection import train_test_split


class CREMADEmotionClassification(AbsTaskAudioClassification):
    label_column_name: str = "label"

    metadata = TaskMetadata(
        name="CREMADEmotionClassification",
        description="Classifying audio recordings based on expressed emotions from the CREMA-D dataset.",
        reference="https://huggingface.co/datasets/AbstractTTS/CREMA-D",
        dataset={
            "path": "AbstractTTS/CREMA-D",
            "revision": "main",
        },
        type="AudioClassification",
        category="a2a",
        eval_splits=["train", "test"], 
        eval_langs=["eng-Latn"],
        main_score="accuracy", 
        date=("2014-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Voice Emotion Classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
    )

def dataset_transform(self):
    EMOTION_MAP = {
        "anger": 0, "happy": 1, "neutral": 2, "sad": 3, "fear": 4, "disgust": 5
    }
    ds_split = self.dataset["train"]

    audio = ds_split["audio"]
    labels = ds_split["major_emotion"]
    
    audio = [{"array": item["array"], "sampling_rate": item["sampling_rate"]} for item in audio]
    labels = [EMOTION_MAP.get(str(label).lower().strip(), -1) for label in labels]

    audio_train, audio_test, labels_train, labels_test = train_test_split(audio, labels, test_size=0.2, random_state=42, stratify=labels)
    self.dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"audio": audio_train, "label": labels_train}),
        "test": datasets.Dataset.from_dict({"audio": audio_test, "label": labels_test}),
    })
        
if __name__ == "__main__":
    model_name = "facebook/wav2vec2-base"
    model = mteb.get_model(model_name)
    print(f"Loaded model type: {type(model)}")
    evaluation = mteb.MTEB(tasks=[CREMADEmotionClassification()])
    classification_method = "logReg" #CHANGE TO K_NN IF NEEDED
    encode_kwarg = {"hidden_layer": 6}
    dataset_size = 224

    results = evaluation.run(
        model,
        output_folder=f"results_Emotions/{classification_method}/{dataset_size}/{model_name}",
        overwrite_results=True,
        classification_method=classification_method, 
        limit=dataset_size,
        encode_kwargs=encode_kwarg
    )
    print(results)
