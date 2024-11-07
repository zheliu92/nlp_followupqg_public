import json
from llm import llm_response

with open("Data_Augmentation/test.json", "r") as f:
# Data_Augmentation/test.json 
# with open("cleaned_train.json", "r") as f:
    data = json.load(f)


def complete_answer(question, is_claude=False):
    def generate_list_of_answers_non_repeat(q, num_answers, is_claude=False):
        if num_answers < 2:
            raise ValueError("Number of answers must be at least 2")
        num_answers -= 1

        answer_list = []

        # Generate the first answer
        first_answer = llm_response(
            "Generate a answer focused on a single perspective only, without any conversational fillers. "
            "Do not repeat the question in the answer.",
            q,
            is_claude=is_claude
        )

        answer_list.append(first_answer.strip())

        for i in range(num_answers):
            previous_answers = "\n".join([f"{i + 1}. {answer}" for i, answer in enumerate(answer_list)])
            refined_q = (
                f"{q}\n"
                f"Previous answers:\n{previous_answers}\n\n"
                "Please provide a new answer focused on a different perspective, ensuring no overlap with previous answers. "
                "Focus on unique aspects or insights not covered earlier, and provide the answer only without any conversational fillers. "
                "Do not repeat the question in the answer."
            )
            next_answer = llm_response(
                "Generate a answer focused on a single perspective only, without any conversational fillers.",
                refined_q,
                is_claude=is_claude
            )
            answer_list.append(next_answer.strip())

        return answer_list

    def summarized_answer(question, answer_list, is_claude=False):
        # Combine the list of answers into a single string
        combined_answers = "\n".join([f"{i + 1}. {answer}" for i, answer in enumerate(answer_list)])

        # Create the instruction and prompt for the LLM
        instruction = "Synthesize the following answers into a single, comprehensive response. Integrate the key points and insights from each answer, ensuring a cohesive and well-rounded explanation. The final answer should be thorough and address multiple aspects of the question without unnecessary repetition."

        prompt = f"""Question: {question}
        
        Answers:
        {combined_answers}"""

        # Generate the comprehensive answer using the LLM
        comprehensive_answer = llm_response(instruction, prompt, is_claude=is_claude)

        return comprehensive_answer.strip()

    return summarized_answer(question, generate_list_of_answers_non_repeat(question, 4, is_claude=is_claude),
                             is_claude=is_claude)


res = []
for i in range(len(data)):
    task_sample = data[i]
    id= data[i]["id"]
    question = data[i]["question"]
    # answers = data[i]["answers"]
    answer = data[i]["answer"]

    c_a = complete_answer(question)
    # for answer in answers:
    res.append({"question": question, "answer": answer, "complete_answer": c_a})

    print("Task", id, "completed")
    json_data = json.dumps(res, indent=2)
    with open("Data_Augmentation/test_complete_answers.json", "w") as f:
        f.write(json_data)
