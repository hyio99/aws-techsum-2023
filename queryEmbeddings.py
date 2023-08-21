import json
import boto3

def inference_embeddings(query):
    client = boto3.client("runtime.sagemaker", region_name="ap-southeast-2")
    endpoint_name_embed = "jumpstart-dft-hf-textembedding-gpt-j-6b-fp16"
    payload = {"text_inputs": query}
    encoded_json = json.dumps(payload).encode("utf-8")

    response = client.invoke_endpoint(
        EndpointName=endpoint_name_embed, 
        ContentType="application/json", 
        Body=encoded_json
    )
        
    model_predictions = json.loads(response["Body"].read())
    embeddings = model_predictions["embedding"]
    
    return embeddings
