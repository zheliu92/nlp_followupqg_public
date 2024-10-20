import re
import spacy
import json
import evaluate

nlp = spacy.load('en_core_web_md')


def preprocess_spacy(
        doc,
        min_token_len=2,
        irrelevant_pos=["ADV", "CCONJ", "PUNCT", "PART", "DET", "ADP", "SPACE"],
):
    clean_text = []

    for token in doc:
        if token.like_email:  # Check if the token is an not like email
            clean_text.append("EMAIL")
        elif token.like_url:  # Check if the token is an not like email
            clean_text.append("URL")
        elif token.like_num:  # Check if the token is an not like email
            clean_text.append("NUM")
        elif (
                token.is_stop == False  # Check if it's not a stopword
                and len(token) > min_token_len  # Check if the word meets minimum threshold
                and token.pos_ not in irrelevant_pos
        ):  # Check if the POS is in the acceptable POS tags
            clean_text.append(token.lemma_.lower())
    return " ".join(clean_text)


def preprocess(text):
    # Replace a sequence of whitespaces by a single whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove other strange characters
    text = re.sub(r"""[\n\r]+""", "", text)

    # Remove other strange characters
    text = re.sub(r"""[\*\~]+""", "", text)

    # Replace slashes with spaces
    text = re.sub(r"""[\/]+""", " ", text)

    return text


def clean_text(text):
    text = preprocess(text)
    text = preprocess_spacy(nlp(text))
    return text


def bert(prediction, reference):
    return evaluate.load("bertscore").compute(predictions=[prediction], references=[reference], lang="en")['f1'][0]


def bleu(prediction, reference, geometric_mean=True):
    res = evaluate.load("bleu").compute(predictions=[prediction], references=[reference])
    if geometric_mean:
        return res['bleu']
    return res['precisions']


def rouge(prediction, reference):
    return evaluate.load("rouge").compute(predictions=[prediction], references=[reference])['rougeL']


def meteor(prediction, reference):
    return evaluate.load("meteor").compute(predictions=[prediction], references=[reference])['meteor']


def nlp_similarity(text1, text2):
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)


def get_auto_eval_result(task_sample, generate_id):
    res = task_sample.copy()
    original_follow_up = task_sample["follow-up"]
    generated_follow_up = task_sample["generated_follow_up"][generate_id]

    res["bert"] = bert(generated_follow_up, original_follow_up)
    res["bleu"] = bleu(generated_follow_up, original_follow_up)
    bleu_1234 = bleu(generated_follow_up, original_follow_up, geometric_mean=False)
    res["bleu1"] = bleu_1234[0]
    res["bleu2"] = bleu_1234[1]
    res["bleu3"] = bleu_1234[2]
    res["bleu4"] = bleu_1234[3]
    res["rouge"] = rouge(generated_follow_up, original_follow_up)
    res["meteor"] = meteor(generated_follow_up, original_follow_up)
    res["nlp_similarity"] = nlp_similarity(generated_follow_up, original_follow_up)
    return res

# model_version = "org" # "org" "small" "full" "full_slow"
# result_filename = model_version + "_result.json"
# evaluation_filename = model_version + "_evaluation.json"

for model_version in ["org", "small", "full", "gpt"]: # 
    for mode in ['topkp']: # 'beam'
        result_filename = model_version + f"_result_repeat_{mode}.json"
        evaluation_filename = model_version + f"_evaluation_repeat_{mode}.json"

        with open(result_filename, "r") as f:
            data = json.load(f)

        result = []

        for i in range(len(data)):
        # for i in range(5):
            id = data[i]["id"]
            temp_res = []
            task_sample = data[i]
            for j in range(len(task_sample["generated_follow_up"])):
                temp_res.append(get_auto_eval_result(task_sample, j))
            result.append(temp_res)

            print("Task", id, "Completed")

            json_data = json.dumps(result, indent=2)
            with open(evaluation_filename, "w") as f:
                f.write(json_data)

# with open("test_set_with_generated_followup_questions_small.json", "r") as f:
#     data = json.load(f)

# result = []

# for i in range(len(data)):
#     result.append(get_auto_eval_result(data[i]))

#     json_data = json.dumps(result, indent=2)
#     with open("auto_evaluation_small.json", "w") as f:
#         f.write(json_data)