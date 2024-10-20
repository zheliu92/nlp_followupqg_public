import json
from typing import List
from pydantic import parse_obj_as, BaseModel, Field
import matplotlib.pyplot as plt
import os

# class Result(BaseModel):
#     id: int = Field(..., example="123456")
#     # follow_up: str = Field(..., example="123456")
#     bert: float = Field(..., example="123456")
#     nlp_similarity: float = Field(..., example="123456")
#     bleu: float = Field(..., example="123456")
#     bleu1: float = Field(..., example="123456")
#     bleu2: float = Field(..., example="123456")
#     bleu3: float = Field(..., example="123456")
#     bleu4: float = Field(..., example="123456")
#     rouge: float = Field(..., example="123456")
#     meteor: float = Field(..., example="123456")

models = ['full', 'gpt','org','small']
modes = ['topkp'] # 'beam'

# print(os.listdir())

for model in models:
    for mode in modes:
        f = open("result.txt", "a")
        
        current_filename = f"Auto_Evaluation/{model}_evaluation_repeat_{mode}.json"

        # Open the JSON file and load its contents
        with open(current_filename, 'r') as json_file:
            json_data = json.load(json_file)

            max_score = {}

            scores = {
                'bert': [],
                'nlp_similarity': [],
                'bleu': [],
                'bleu1': [],
                'bleu2': [],
                'bleu3': [],
                'bleu4': [],
                'rouge': [],
                'meteor': []
            }

            for case in json_data:
                if max_score:
                    scores['bert'].append(max_score['bert'])
                    scores['nlp_similarity'].append(max_score['nlp_similarity'])
                    scores['bleu'].append(max_score['bleu'])
                    scores['bleu1'].append(max_score['bleu1'])
                    scores['bleu2'].append(max_score['bleu2'])
                    scores['bleu3'].append(max_score['bleu3'])
                    scores['bleu4'].append(max_score['bleu4'])
                    scores['rouge'].append(max_score['rouge'])
                    scores['meteor'].append(max_score['meteor'])

                max_score['bert'] = case[0]['bert']
                max_score['nlp_similarity'] = case[0]['nlp_similarity']
                max_score['bleu'] = case[0]['bleu']
                max_score['bleu1'] = case[0]['bleu1']
                max_score['bleu2'] = case[0]['bleu2']
                max_score['bleu3'] = case[0]['bleu3']
                max_score['bleu4'] = case[0]['bleu4']
                max_score['rouge'] = case[0]['rouge']
                max_score['meteor'] = case[0]['meteor']
                
                for task in case[1:]:
                    max_score['bert'] = max(max_score['bert'], task['bert'])
                    max_score['nlp_similarity'] = max(max_score['nlp_similarity'], task['nlp_similarity'])
                    max_score['bleu'] = max(max_score['bleu'], task['bleu'])
                    max_score['bleu1'] = max(max_score['bleu1'], task['bleu1'])
                    max_score['bleu2'] = max(max_score['bleu2'], task['bleu2'])
                    max_score['bleu3'] = max(max_score['bleu3'], task['bleu3'])
                    max_score['bleu4'] = max(max_score['bleu4'], task['bleu4'])
                    max_score['rouge'] = max(max_score['rouge'], task['rouge'])
                    max_score['meteor'] = max(max_score['meteor'], task['meteor'])
              
                scores['bert'].append(max_score['bert'])
                scores['nlp_similarity'].append(max_score['nlp_similarity'])
                scores['bleu'].append(max_score['bleu'])
                scores['bleu1'].append(max_score['bleu1'])
                scores['bleu2'].append(max_score['bleu2'])
                scores['bleu3'].append(max_score['bleu3'])
                scores['bleu4'].append(max_score['bleu4'])
                scores['rouge'].append(max_score['rouge'])
                scores['meteor'].append(max_score['meteor'])
            plt.clf()
            fig, ax = plt.subplots()
            ax.boxplot(scores.values())
            ax.set_xticklabels(scores.keys())
            # rotate x-axis labels
            plt.xticks(rotation=25)
            ax.set_title(f'Bart - Scores by metric_{model}_{mode}')
            plt.savefig(f'Auto_Evaluation/{model}_scores_{mode}.png')

            f.write(f'Auto_Evaluation: {model}_scores_{mode} \n')
            for key, value in scores.items():
                # print(f'{key}: {sum(value) / len(value)}')       
                f.write(f'{key}: {"%.4f" % (sum(value) / len(value))} \n')
        f.close()