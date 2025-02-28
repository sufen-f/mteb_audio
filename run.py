import mteb
from mteb.tasks.Audio.Clustering.eng.VoiceGender import VoiceGenderClustering
from mteb.tasks.Audio.Clustering.eng.VoiceEmotions import CREMADEmotionClustering
import itertools

model_names = ["facebook/wav2vec2-base"]

cluster_algos = ["Kmeans"]
pca_n_components_values = [5]
encode_hidden_layers = [-1]
dataset_sizes = [10]

for model_name in model_names:
    model = mteb.get_model(model_name)
    print(f"Loaded model: {model_name} (Type: {type(model)})")

    evaluation = mteb.MTEB(tasks=[CREMADEmotionClustering()])

    for cluster_algo, pca_n_components, hidden_layer, dataset_size in itertools.product(
            cluster_algos, pca_n_components_values, encode_hidden_layers, dataset_sizes):
        
        encode_kwarg = {"hidden_layer": hidden_layer}

        results = evaluation.run(
            model,
            output_folder=f"results_gender/{model_name}/{cluster_algo}/{dataset_size}/{pca_n_components}/{hidden_layer}",
            overwrite_results=True,
            cluster_algo=cluster_algo,
            limit=dataset_size,
            pca_n_components=pca_n_components,
            encode_kwargs=encode_kwarg
        )
        
        print(f"results for Model={model_name}, Cluster={cluster_algo}, PCA={pca_n_components}, Hidden Layer={hidden_layer}, Dataset Size={dataset_size}:")
        print(results)
