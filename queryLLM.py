import boto3
import json

# Define the API endpoint for the chatbot
endpoint_name = "jumpstart-dft-meta-textgeneration-llama-2-7b-f"
client = boto3.client("runtime.sagemaker", region_name="ap-southeast-2")

def query_llm(user_input):
    payload = {
        "inputs" : [[
                {"role": "system", "content": "You are a super smart chat bot who provides succint and insightful answers."},
                {"role": "user", "content": user_input}]],
                
        "parameters": {"max_new_tokens":256, "top_p":0.9, "temperature":0.6}
    }

    payload_json = json.dumps(payload)

    b = bytes(payload_json, encoding="utf-8")
    response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=b,
            ContentType="application/json",
            CustomAttributes="accept_eula=true"

    )

    return response