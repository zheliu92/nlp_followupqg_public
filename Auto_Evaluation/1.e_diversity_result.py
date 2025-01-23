import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import string
import numpy as np

models = ['full', 'gpt', 'org']
data = []  # Collect data for the combined grouped histogram

def calculate_statistics(lengths):
    if not lengths:
        return 0, 0, 0, 0  # mean, min, max, std_dev
    return (
        np.mean(lengths),
        np.min(lengths),
        np.max(lengths),
        np.std(lengths)
    )

for model in models:
    current_filename = f"Auto_Evaluation/{model}_clustered.json"

    with open(current_filename, 'r') as json_file:
        json_data = json.load(json_file)

    # Initialize variables for calculations
    vanilla_follow_up_lengths = []
    cleaned_follow_up_lengths = []

    # Helper function to calculate word count
    def word_count(text):
        return len(text.split())

    # Process each dictionary in the data
    for entry in json_data:
        generated_follow_ups = entry.get("generated_follow_up", [])

        # Vanilla calculations
        vanilla_follow_up_lengths.extend([word_count(follow_up) for follow_up in generated_follow_ups])

        # Cleaned calculations (filter follow-ups that end with '?')
        cleaned_follow_ups = [follow_up for follow_up in generated_follow_ups if follow_up.strip().endswith('?')]
        cleaned_follow_up_lengths.extend([word_count(follow_up) for follow_up in cleaned_follow_ups])

    # Calculate statistics for vanilla
    vanilla_mean, vanilla_min, vanilla_max, vanilla_std = calculate_statistics(vanilla_follow_up_lengths)

    # Calculate statistics for cleaned
    cleaned_mean, cleaned_min, cleaned_max, cleaned_std = calculate_statistics(cleaned_follow_up_lengths)

    # Print results
    print(f"Model: {model}")

    print("Vanilla Condition:")
    print(f"Average length of generated follow-ups (words): {vanilla_mean:.2f}")
    print(f"Shortest follow-up length (words): {vanilla_min}")
    print(f"Longest follow-up length (words): {vanilla_max}")
    print(f"Standard deviation of follow-up lengths: {vanilla_std:.2f}")

    print("\nCleaned Condition:")
    print(f"Average length of generated follow-ups (words): {cleaned_mean:.2f}")
    print(f"Shortest follow-up length (words): {cleaned_min}")
    print(f"Longest follow-up length (words): {cleaned_max}")
    print(f"Standard deviation of follow-up lengths: {cleaned_std:.2f}")
    print("\n")
