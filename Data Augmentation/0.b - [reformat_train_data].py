import json

with open("../data/train.json", "r") as f:
    data = json.load(f)

appeared_questions = {}

for i in range(len(data)):
    task_sample = data[i]
    question = data[i]["question"]
    if question in appeared_questions:
        # duplicate question
        if task_sample["answer"] not in appeared_questions[question]:
            # new answer
            appeared_questions[question][task_sample["answer"]] = [task_sample["follow-up"]]
        else:
            # duplicate answer
            if task_sample["follow-up"] not in appeared_questions[question][task_sample["answer"]]:
                appeared_questions[question][task_sample["answer"]].append(task_sample["follow-up"])
    else:
        appeared_questions[question] = {}
        appeared_questions[question][task_sample["answer"]] = [task_sample["follow-up"]]

res = []
num_answers = 0
for i in range(len(appeared_questions)):
    question = list(appeared_questions.keys())[i]
    task_sample = {"id": i, "question": question, "answers": []}

    for answer in appeared_questions[question]:
        num_answers += 1
        task_sample["answers"].append({"answer": answer, "follow-ups": appeared_questions[question][answer]})

    res.append(task_sample)

json_data = json.dumps(res, indent=2)

with open('cleaned_train.json', 'w') as json_file:
    json_file.write(json_data)

print(num_answers)
print(len(res))
