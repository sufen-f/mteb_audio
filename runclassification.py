import mteb
from mteb.tasks.Audio.Classification.eng.VoiceGender import VoiceGenderClassification
from mteb.tasks.Audio.Classification.eng.VoiceEmotions import CREMADEmotionClassification
import itertools
from tqdm import tqdm
import multiprocessing
import torch

model_names = [
    "facebook/wav2vec2-base",
    # "facebook/wav2vec2-base-960h",
    # "facebook/wav2vec2-large",
    # "facebook/wav2vec2-large-xlsr-53",
    # "facebook/wav2vec2-lv-60-espeak-cv-ft",
    # "microsoft/wavlm-base-plus-sd",
    # "microsoft/wavlm-base-plus-sv",
    # "microsoft/wavlm-base-sd",
    # "microsoft/wavlm-base-sv",
    # "microsoft/wavlm-base-plus",
    # "microsoft/wavlm-base",
    # "microsoft/wavlm-large",
    # "openai/whisper-large-v3",
    # "openai/whisper-medium",
    #"openai/whisper-tiny",
    # "openai/whisper-base",
    # "openai/whisper-small",
    # "Qwen/Qwen2-Audio-7B"
]

encode_hidden_layers = [0.5]
dataset_sizes = [1024]
tasks = [[VoiceGenderClassification()],[CREMADEmotionClassification()]]
tasks_name = ['gender','emotion']
class_algos = ["logReg"]


for i in range(len(tasks)):
    task = tasks[i]
    task_name = tasks_name[i]
    for model_name in model_names:
        model = mteb.get_model(model_name)
        print(f"Loaded model: {model_name} (Type: {type(model)})")

        evaluation = mteb.MTEB(tasks=task)

        for class_algo, hidden_layer_percentage, dataset_size in tqdm(
                itertools.product(class_algos, encode_hidden_layers, dataset_sizes), 
                total=len(class_algos)  * len(encode_hidden_layers) * len(dataset_sizes)):
    
            
            print(f"results for Model={model_name}, Classification={class_algo}, Hidden Layer={hidden_layer_percentage}, Dataset Size={dataset_size}:")
            encode_kwarg = {"file_path": f"new_{task_name}_embeddings/{model_name}/{hidden_layer_percentage}/embeddings.npz", "embed_limit": dataset_size, "test_split": .8,} 
            try:
                results = evaluation.run(
                    model,
                    output_folder=f"Class_results_{tasks_name[i]}/{model_name}/{class_algo}/{dataset_size}/{hidden_layer_percentage}",
                    overwrite_results=True,
                    encode_kwargs=encode_kwarg
                )
                print("saved in",f"Class_results_{tasks_name[i]}/{model_name}/{class_algo}/{dataset_size}/{hidden_layer_percentage}")
            
            except RuntimeError as e:
                print("ERROR")
                continue

            
            print(results)
        del model
        torch.cuda.empty_cache()