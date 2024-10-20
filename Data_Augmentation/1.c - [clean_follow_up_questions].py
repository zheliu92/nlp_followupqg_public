import json

with open("follow_up_questions.json", "r") as f:
    data = json.load(f)

for i in range(len(data)):
    task_sample = data[i]
    for j in range(len(data[i]["follow-ups"])):
        q = data[i]["follow-ups"][j]

        front_pointer = 0
        if q:
            while not q[front_pointer].isalpha():
                front_pointer += 1

        back_pointer = len(q) - 1
        if q:
            while not q[back_pointer].isalpha():
                back_pointer -= 1

        if back_pointer < len(q) - 1:
            back_pointer += 1

        data[i]["follow-ups"][j] = q[front_pointer:back_pointer + 1]

json_data = json.dumps(data, indent=2)

with open("clean_follow_up_questions.json", "w") as f:
    f.write(json_data)
