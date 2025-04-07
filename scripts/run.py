import mteb

model_name = "microsoft/wavlm-base"
model = mteb.get_model(
    model_name, model_revision="efa81aae7ff777e464159e0f877d54eac5b84f81"
)
tasks = mteb.get_tasks(tasks=["VoiceGenderClustering"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")