import mteb
from mteb.tasks.Audio.Clustering.eng.VoiceGender import VoiceGenderClustering
from mteb.tasks.Audio.Clustering.eng.VoiceEmotions import CREMADEmotionClustering
import itertools
from tqdm import tqdm

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
    # "openai/whisper-tiny",
    # "openai/whisper-base",
    # "openai/whisper-small",
    # "Qwen/Qwen2-Audio-7B"
]


# cluster_algos = ["Kmeans", "DBSCAN", "Agg"]
cluster_algos = ["Kmeans", "Agg"]

pca_n_components_values = [200, None]
encode_hidden_layers = [0.25, 0.5, 1]
dataset_sizes = [512, 1024, 2048]
tasks = [[VoiceGenderClustering()],[CREMADEmotionClustering()]]
tasks_name = ['gender','emotion']


for i in range(len(tasks)-1):
    task = tasks[i]
    task_name = tasks_name[i]
    for model_name in model_names:
        model = mteb.get_model(model_name,device='cpu')
        print(f"Loaded model: {model_name} (Type: {type(model)})")

        evaluation = mteb.MTEB(tasks=task)

        for cluster_algo, pca_n_components, hidden_layer_percentage, dataset_size in tqdm(
                itertools.product(cluster_algos, pca_n_components_values, encode_hidden_layers, dataset_sizes), 
                total=len(cluster_algos) * len(pca_n_components_values) * len(encode_hidden_layers) * len(dataset_sizes)):
    
            
            print(f"results for Model={model_name}, Cluster={cluster_algo}, PCA={pca_n_components}, Hidden Layer={hidden_layer_percentage}, Dataset Size={dataset_size}:")

            encode_kwarg = {"file_path": f"new_{task_name}_embeddings/{model_name}/{hidden_layer_percentage}/embeddings.npz", "embed_limit": dataset_size} 
            try:
                results = evaluation.run(
                    model,
                    output_folder=f"results_{tasks_name[i]}/{model_name}/{cluster_algo}/{dataset_size}/{pca_n_components}/{hidden_layer_percentage}",
                    cluster_algo=cluster_algo,
                    limit=dataset_size,
                    pca_n_components=pca_n_components,
                    encode_kwargs=encode_kwarg
                )
                print("saved in",f"results_{tasks_name[i]}/{model_name}/{cluster_algo}/{dataset_size}/{pca_n_components}/{hidden_layer_percentage}")
            
            except RuntimeError as e:
                print("ERROR")
                continue

            
            print(results)