from openai import OpenAI, AzureOpenAI
import anthropic


# call this function to get a response from the selected model given an instruction and a prompt
def llm_response(instruction, prompt, is_claude=False):
    # if is_claude:
    #     return claude_response(instruction, prompt)
    # return openai_response(instruction, prompt)
    return azure_response(instruction, prompt)


def azure_response(instruction, prompt):
    try:
        azure_client = AzureOpenAI(
            azure_endpoint='', # copy the endpoint from the azure portal
            api_key='', # copy the key from the azure portal
            api_version=""
        )

        response = azure_client.chat.completions.create(
            model="gpt4o",  # change this to the model you want to use (4o)
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt}
            ]
        )
    except Exception as e:
        print(e)
        return "LLM failed to generate a response"

    if response.choices[0].message.content is None:
        return "LLM failed to generate a response"
    return response.choices[0].message.content

#
# def openai_response(instruction, prompt, model="gpt-4o"):
#     client = OpenAI(api_key="sk-proj-8tSVsjs8qOaPbvut4XS3T3BlbkFJNDhTNRiiaH5nIebF9xbW")
#     completion = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": instruction},
#             {"role": "user", "content": prompt}
#         ])
#
#     return completion.choices[0].message.content
#
#
# def claude_response(instruction, prompt, model="claude-3-5-sonnet-20240620"):
#     client = anthropic.Anthropic(
#         api_key="")
#
#     message = client.messages.create(
#         model=model,
#         max_tokens=1000,
#         temperature=0,
#         system=instruction,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": prompt
#                     }
#                 ]
#             }
#         ]
#     )
#     return message.content[0].text
