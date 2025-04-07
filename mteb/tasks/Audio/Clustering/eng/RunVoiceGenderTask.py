from __future__ import annotations

import mteb
from mteb.tasks.Audio.Clustering.eng.VoiceGender import *

model_name = "facebook/wav2vec2-base"
model = mteb.get_model(model_name, revision="0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8")
print(f"Loaded model type: {type(model)}")

cluster_algo = "Kmeans"
pca_n_components = 50
h = 6
encode_kwarg = {"hidden_layer": 6}
dataset_size = 224
evaluation = mteb.MTEB(tasks=[VoiceGenderClustering()])
results = evaluation.run(
    model,
    output_folder=f"results_Voice/{cluster_algo}/{dataset_size}/{pca_n_components}/{h}/{model_name}",
    overwrite_results=True,
    cluster_algo=cluster_algo,
    limit=dataset_size,
    pca_n_components=pca_n_components,
    encode_kwargs=encode_kwarg,
)
print(results)
