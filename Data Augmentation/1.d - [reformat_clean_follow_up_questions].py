import json
import random

with open("filtered_follow_up_questions.json", "r") as f:
    data = json.load(f)

print(len(data))

res = []

id = 0

for task_sample in data:
    # remove the first follow-up question for only gpt-gerenated follow-up questions
    for follow_up in task_sample["follow-ups"][1:]:
        res.append({
            "id": id,
            "question": task_sample["question"],
            "answer": task_sample["answer"],
            "follow-up": follow_up
        })
        id += 1

print(len(res))

json_data = json.dumps(res, indent=2)

# with open("train_full.json", "w") as f:
#     f.write(json_data)

original_size = len(res)
small_set_size = 2789

assert small_set_size < original_size

small_set = random.sample(res, small_set_size)

json_data = json.dumps(small_set, indent=2)

with open("train_gpt.json", "w") as f:
    f.write(json_data)

manual_set = random.sample(small_set, 100)

json_data = json.dumps(manual_set, indent=2)

with open("train_manual.json", "w") as f:
    f.write(json_data)
