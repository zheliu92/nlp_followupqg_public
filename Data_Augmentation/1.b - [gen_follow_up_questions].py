import json
from llm import llm_response

with open("cleaned_train.json", "r") as f:
    data = json.load(f)

with open("complete_answers.json", "r") as f:
    complete_answers = json.load(f)


def follow_up_questions_list(q, a, ca):
    instruction = (
        """generate all possible follow-up questions as candidates. These follow-up questions must be related to the original question, but must not be rephrases of the original question. These follow-up questions should be answerable by the complete answer. These follow-up questions should not be answered, covered or detailed by the original answer, but must be targeting on terminologies mentioned in the original answer. separate each follow-up question with <sep>"""
    )

    prompt = f"""Original Question: {q}
            Original Answer: {a}
            Comprehensive Answer: {ca}"""

    follow_up_questions = llm_response(instruction, prompt, is_claude=False).strip().split('<sep>')

    res = []
    for i in range(len(follow_up_questions)):
        follow_up_questions[i] = follow_up_questions[i].strip()
        # remove \n
        follow_up_questions[i] = follow_up_questions[i].replace('\n', '')
        res.append(follow_up_questions[i])

    return res


res = []
c_a_index = 0
for i in range(len(data)):
    task_sample = data[i]
    question = data[i]["question"]
    answers = data[i]["answers"]
    id = data[i]["id"]

    for answer in answers:
        c_a = complete_answers[c_a_index]["complete_answer"]
        old_follow_up_questions = answer["follow-ups"]
        new_follow_up_questions = follow_up_questions_list(question, answer["answer"], c_a)
        old_follow_up_questions.extend(new_follow_up_questions)
        res.append({"id": c_a_index, "question": question, "answer": answer["answer"], "follow-ups": old_follow_up_questions})
        c_a_index += 1

    print("Task", id, "completed")
    json_data = json.dumps(res, indent=2)
    with open("follow_up_questions.json", "w") as f:
        f.write(json_data)
