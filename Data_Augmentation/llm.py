from openai import OpenAI, AzureOpenAI
import anthropic

# call this function to get a response from the selected model given an instruction and a prompt
def llm_response(instruction, prompt):
    return azure_response(instruction, prompt)

def azure_response(instruction, prompt):
    try:
        azure_client = AzureOpenAI(
            azure_endpoint='https://[REPLACE_WITH_AZURE_ENDPOINT]', # copy the endpoint from the azure portal
            api_key='[REPLACE_WITH_API_KEY]', # copy the key from the azure portal
            api_version="2024-02-15-preview"
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
