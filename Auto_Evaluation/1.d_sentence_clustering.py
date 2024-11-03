from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import spacy
import json
import os

embedder = SentenceTransformer("all-mpnet-base-v2")

models = ['full', 'gpt','org']

for model in models:
    data_filename = f"{model}_result.json"
    print("Start working on " + data_filename)
    result_filename = f"{model}_clustered.json"

    with open(data_filename, "r") as f:
        data = json.load(f)

    result = []

    for i in range(len(data)):
    # for i in range(5):
        task_sample = data[i]
        id = task_sample['id']
        current_corpus = task_sample["generated_follow_up"]
        clustered_list = []

        corpus_embeddings = embedder.encode(current_corpus)

        # Perform agglomerative clustering
        clustering_model = AgglomerativeClustering(
            # affinity='cosine', linkage='average', distance_threshold=0.4
            n_clusters=None, distance_threshold=1.0
        ) 
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []

            clustered_sentences[cluster_id].append(current_corpus[sentence_id])

        for i, cluster in clustered_sentences.items():
            clustered_list.append(cluster[0])
        
        task_sample["generated_follow_up"] = clustered_list
        result.append(task_sample)

        print("Task", id, "Completed")

        json_data = json.dumps(result, indent=2)
        
        with open(result_filename, "w") as f:
            f.write(json_data)
