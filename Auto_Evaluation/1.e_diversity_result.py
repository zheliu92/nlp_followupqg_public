import json
import numpy as np

models = ['full', 'gpt', 'org']

def calculate_statistics(lengths):
    if not lengths:
        return 0, 0, 0, 0  # mean, min, max, std_dev
    return (
        np.mean(lengths),
        np.min(lengths),
        np.max(lengths),
        np.std(lengths)
    )

def word_count(text):
    return len(text.split())

for model in models:
    current_filename = f"Human_Evaluation/Helpers/diversity_output/{model}_valid_fq_only.json"
    
    with open(current_filename, 'r') as json_file:
        json_data = json.load(json_file)
    
    vanilla_follow_up_lengths = []
    follow_up_data = []
    
    for entry in json_data:
        generated_follow_ups = entry.get("generated_follow_up", [])
        for follow_up in generated_follow_ups:
            length = word_count(follow_up)
            vanilla_follow_up_lengths.append(length)
            follow_up_data.append((entry["id"], follow_up, length))
    
    # Calculate statistics
    vanilla_mean, vanilla_min, vanilla_max, vanilla_std = calculate_statistics(vanilla_follow_up_lengths)
    
    # Find shortest and longest follow-ups
    if follow_up_data:
        shortest_entry = min(follow_up_data, key=lambda x: x[2])
        longest_entry = max(follow_up_data, key=lambda x: x[2])
    else:
        shortest_entry = (None, "", 0)
        longest_entry = (None, "", 0)
    
    # Print results
    print(f"Model: {model}")
    print("Vanilla Condition:")
    print(f"Average length of generated follow-ups (words): {vanilla_mean:.2f}")
    print(f"Shortest follow-up length (words): {vanilla_min}")
    print(f"Longest follow-up length (words): {vanilla_max}")
    print(f"Standard deviation of follow-up lengths: {vanilla_std:.2f}")
    print(f"\nShortest Follow-Up: [ID: {shortest_entry[0]}] {shortest_entry[1]}")
    print(f"Longest Follow-Up: [ID: {longest_entry[0]}] {longest_entry[1]}")
    print("\n")
