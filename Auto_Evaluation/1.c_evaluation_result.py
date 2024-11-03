import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

models = ['full', 'gpt', 'org']
data = []  # Collect data for the combined grouped histogram

for model in models:
    current_filename = f"Auto_Evaluation/{model}_clustered_evaluation.json"
    
    with open(current_filename, 'r') as json_file:
        json_data = json.load(json_file)
        
        max_score = {}
        scores = {metric: [] for metric in [
            'bert', 'nlp_similarity', 'bleu', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge', 'meteor'
        ]}

        # Calculate max scores per case and store in scores dictionary
        for case in json_data:
            max_score = {metric: case[0][metric] for metric in scores}
            
            for task in case[1:]:
                for metric in scores:
                    max_score[metric] = max(max_score[metric], task[metric])
                    
            for metric, value in max_score.items():
                scores[metric].append(value)
                data.append({'Model': model, 'Metric': metric, 'Score': value})  # Add for combined figure

    # Plot each model's metrics as a separate bar plot with Seaborn
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.barplot(data=pd.DataFrame(
        [{'Metric': k, 'Score': v} for k, values in scores.items() for v in values]),
        x='Metric', y='Score', ci='sd', palette='pastel'
    )
    plt.xticks(rotation=25)
    plt.ylim(0.0, 1.0)
    plt.title(f'{model.capitalize()} Model - Scores by Metric')
    plt.tight_layout()
    plt.savefig(f'Auto_Evaluation/{model}_scores_enhanced.png')
    plt.close()  # Close figure to avoid overlap in loops

# Create DataFrame for combined plot
df = pd.DataFrame(data)

# Combined grouped bar plot using Seaborn for consistency
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
sns.barplot(data=df, x='Metric', y='Score', hue='Model', ci='sd', palette='pastel')
plt.xticks(rotation=25)
plt.ylim(0.0, 1.0)
plt.title('Model Performance Comparison by Metric')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('Auto_Evaluation/all_models_comparison_enhanced.png')
plt.show()