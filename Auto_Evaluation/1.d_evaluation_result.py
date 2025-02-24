import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

models = ['full', 'gpt', 'org']
data = []  # Collect data for the combined grouped histogram

for model in models:
    current_filename = f"Auto_Evaluation/{model}_clustered_evaluation.json"

    with open(current_filename, 'r') as json_file:
        json_data = json.load(json_file)

        scores = {metric: [] for metric in [
            'bert', 'nlp_similarity', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge', 'meteor'
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

    # Print out the results in text format
    print(f"\n=== {model.capitalize()} Model - Score Summary ===")
    for metric, values in scores.items():
        avg_score = sum(values) / len(values)
        print(f"{metric}: {avg_score:.4f}")
    print("=======================================\n")
    
    # Plot each model's metrics as a separate bar plot with Seaborn
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    ax = sns.barplot(
        data=pd.DataFrame(
            [{'Metric': k, 'Score': v} for k, values in scores.items() for v in values]
        ),
        x='Metric', y='Score', errorbar='sd', palette='muted'
    )
    
    # Add score labels on bars
    for bar in ax.patches:
        ax.annotate(
            format(bar.get_height(), ".2f"),
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
            ha='center', va='bottom', fontsize=10
        )

    plt.xticks(rotation=25, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.0, 1.1)
    plt.title(f'{model.capitalize()} Model - Scores by Metric', fontsize=14, weight='bold')
    plt.xlabel('Metric', fontsize=12, weight='bold')
    plt.ylabel('Score', fontsize=12, weight='bold')
    plt.tight_layout()
    plt.savefig(f'Auto_Evaluation/{model}_scores_enhanced.png')
    plt.close()

# Create DataFrame for combined plot
df = pd.DataFrame(data)

# Combined grouped bar plot using Seaborn for consistency
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
ax = sns.barplot(data=df, x='Metric', y='Score', hue='Model', palette='muted')

# Add score labels on bars for combined plot
for bar in ax.patches:
    ax.annotate(
        format(bar.get_height(), ".2f"),
        (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
        ha='center', va='bottom', fontsize=10
    )

plt.xticks(rotation=25, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0.0, 1.1)
plt.title('Model Performance Comparison by Metric', fontsize=16, weight='bold')
plt.xlabel('Metric', fontsize=12, weight='bold')
plt.ylabel('Score', fontsize=12, weight='bold')
plt.legend(title='Model', title_fontsize=12, fontsize=10)
plt.tight_layout()
plt.savefig('Auto_Evaluation/all_models_comparison_enhanced.png')
plt.show()