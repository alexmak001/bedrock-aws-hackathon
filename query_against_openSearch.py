import boto3
import json
from dotenv import load_dotenv
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.bedrock import BedrockChat
from botocore.client import Config
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import RetrievalQA

# loading in variables from .env file
load_dotenv()

# instantiating the Bedrock client, and passing in the CLI profile
boto3.setup_default_session(profile_name=os.getenv('profile_name'))
bedrock = boto3.client('bedrock-runtime', 'us-west-2',
                    aws_access_key_id="ASIA4J2KKC27ITX6GYJR", aws_secret_access_key="KHk28QPe0mgHg5su+7dn03KG5iIZXt5pX3lXP8IZ",
                    aws_session_token="IQoJb3JpZ2luX2VjEO3//////////wEaCXVzLWVhc3QtMSJGMEQCIApLKm6tPIWCp9b3xIlA7mo5bbf65lWFnB6RENYem//NAiATdCbfSBTF22jb05kj9xSv4ljSzTfM4NyjIYkeGxf0DSqZAghlEAAaDDg0NTcyNzUzNjgzMCIMM7FkP6zkVMNBQL7rKvYB8wg9Z0hSwvrA1peXXbQuzBB1Rpbu8avi+AG9If5ZvgV/nzQbobXkbHGvfIvJuyqTIOx3GVnNcUuIOjxQB7Wp9KDvbgfsoryncdx3bhRcNHduBOs0buORut6KrqZJd9tI/BUfpWAmbIXHD7bT3gy4zjSzMQsP0DuBDXtUMDydeSMhIUu+Aei+BezrJz1ACwnV7/0I+t76n1QZzQmWUk3QwkkTBa+iGdII52PXX/fgStc5bLlOHSK/prc3ttH9yLmclCsjj/zxpNq2ujtciFAZY2hhYOAHiaezwM6pjJfHMeM6ZINhfpR1VuWAAGBt1RkUffJJa8JaMMekubIGOp4B37mHhbjioeTxaDEjiRoA93gQ+jPvQMr0Cg0Wld6zNlen4NhNyKbN7iyBa1tFBZQXHdKns3vY1YcQKxe0aMXflzc70VTwaS9sllA+V2DCVs+c9WjMopBGQeGhoJ1EKaTCQjuZYPftYq+eTW2ftr44gc794F0TqedJkiWrUXN8Xp3lIXvnymn2KnLknskzSgDPQ3igSeMGI7xChm6p0cE=")

# instantiating the OpenSearch client, and passing in the CLI profile
opensearch = boto3.client("opensearchserverless")
host = "yggodhs7a5uc99foq2of.us-west-2.aoss.amazonaws.com"  # cluster endpoint, for example: my-test-domain.us-east-1.aoss.amazonaws.com
region = 'us-west-2'
service = 'aoss'
credentials = boto3.Session(aws_access_key_id="ASIA4J2KKC27ITX6GYJR", aws_secret_access_key="KHk28QPe0mgHg5su+7dn03KG5iIZXt5pX3lXP8IZ",
                    aws_session_token="IQoJb3JpZ2luX2VjEO3//////////wEaCXVzLWVhc3QtMSJGMEQCIApLKm6tPIWCp9b3xIlA7mo5bbf65lWFnB6RENYem//NAiATdCbfSBTF22jb05kj9xSv4ljSzTfM4NyjIYkeGxf0DSqZAghlEAAaDDg0NTcyNzUzNjgzMCIMM7FkP6zkVMNBQL7rKvYB8wg9Z0hSwvrA1peXXbQuzBB1Rpbu8avi+AG9If5ZvgV/nzQbobXkbHGvfIvJuyqTIOx3GVnNcUuIOjxQB7Wp9KDvbgfsoryncdx3bhRcNHduBOs0buORut6KrqZJd9tI/BUfpWAmbIXHD7bT3gy4zjSzMQsP0DuBDXtUMDydeSMhIUu+Aei+BezrJz1ACwnV7/0I+t76n1QZzQmWUk3QwkkTBa+iGdII52PXX/fgStc5bLlOHSK/prc3ttH9yLmclCsjj/zxpNq2ujtciFAZY2hhYOAHiaezwM6pjJfHMeM6ZINhfpR1VuWAAGBt1RkUffJJa8JaMMekubIGOp4B37mHhbjioeTxaDEjiRoA93gQ+jPvQMr0Cg0Wld6zNlen4NhNyKbN7iyBa1tFBZQXHdKns3vY1YcQKxe0aMXflzc70VTwaS9sllA+V2DCVs+c9WjMopBGQeGhoJ1EKaTCQjuZYPftYq+eTW2ftr44gc794F0TqedJkiWrUXN8Xp3lIXvnymn2KnLknskzSgDPQ3igSeMGI7xChm6p0cE=").get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],

    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    pool_maxsize=20
)

def get_embedding(body):
    """
    This function is used to generate the embeddings for each question the user submits.
    :param body: This is the question that is passed in to generate an embedding
    :return: A vector containing the embeddings of the passed in content
    """
    # defining the embeddings model
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType = 'application/json'
    # invoking the embedding model
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    # reading in the specific embedding
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding

def answer_query(user_input):
    """
    This function takes the user question, creates an embedding of that question,
    and performs a KNN search on your Amazon OpenSearch Index. Using the most similar results it feeds that into the Prompt
    and LLM as context to generate an answer.
    :param user_input: This is the natural language question that is passed in through the app.py file.
    :return: The answer to your question from the LLM based on the context that was provided by the KNN search of OpenSearch.
    """
    # Setting primary variables, of the user input
    userQuery = user_input

    bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
    bedrock_client = boto3.client('bedrock-runtime','us-west-2',aws_access_key_id="ASIA4J2KKC27ITX6GYJR", aws_secret_access_key="KHk28QPe0mgHg5su+7dn03KG5iIZXt5pX3lXP8IZ",
                    aws_session_token="IQoJb3JpZ2luX2VjEO3//////////wEaCXVzLWVhc3QtMSJGMEQCIApLKm6tPIWCp9b3xIlA7mo5bbf65lWFnB6RENYem//NAiATdCbfSBTF22jb05kj9xSv4ljSzTfM4NyjIYkeGxf0DSqZAghlEAAaDDg0NTcyNzUzNjgzMCIMM7FkP6zkVMNBQL7rKvYB8wg9Z0hSwvrA1peXXbQuzBB1Rpbu8avi+AG9If5ZvgV/nzQbobXkbHGvfIvJuyqTIOx3GVnNcUuIOjxQB7Wp9KDvbgfsoryncdx3bhRcNHduBOs0buORut6KrqZJd9tI/BUfpWAmbIXHD7bT3gy4zjSzMQsP0DuBDXtUMDydeSMhIUu+Aei+BezrJz1ACwnV7/0I+t76n1QZzQmWUk3QwkkTBa+iGdII52PXX/fgStc5bLlOHSK/prc3ttH9yLmclCsjj/zxpNq2ujtciFAZY2hhYOAHiaezwM6pjJfHMeM6ZINhfpR1VuWAAGBt1RkUffJJa8JaMMekubIGOp4B37mHhbjioeTxaDEjiRoA93gQ+jPvQMr0Cg0Wld6zNlen4NhNyKbN7iyBa1tFBZQXHdKns3vY1YcQKxe0aMXflzc70VTwaS9sllA+V2DCVs+c9WjMopBGQeGhoJ1EKaTCQjuZYPftYq+eTW2ftr44gc794F0TqedJkiWrUXN8Xp3lIXvnymn2KnLknskzSgDPQ3igSeMGI7xChm6p0cE=")

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0" # change this to use a different LLM

    llm = BedrockChat(model_id=modelId, client=bedrock_client)


    retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="LPBSTTFNF1",# enter knowledge base id here
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)
    
    query = userQuery

    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    response = qa.invoke(query)
    return response["result"]


    # formatting the user input
    userQueryBody = json.dumps({"inputText": userQuery})
    # creating an embedding of the user input to perform a KNN search with
    userVectors = get_embedding(userQueryBody)
    # the query parameters for the KNN search performed by Amazon OpenSearch with the generated User Vector passed in.
    # TODO: If you wanted to add pre-filtering on the query you could by editing this query!
    query = {
        "size": 1,
        "query": {
            "knn": {
                "vector": {
                    "vector": userVectors, 
                    "k": 1
                }
            }
        },
        "_source": True,
        "fields": ["text"],
    }
    # performing the search on OpenSearch passing in the query parameters constructed above
    response = client.search(
        index="bedrock-knowledge-base-default-index",
        body=query
    )

    # Format Json responses into text
    similaritysearchResponse = ""
    # iterating through all the findings of Amazon openSearch and adding them to a single string to pass in as context
    for i in response["hits"]["hits"]:
        outputtext = i["fields"]["text"]
        similaritysearchResponse = similaritysearchResponse + "Info = " + str(outputtext)

        similaritysearchResponse = similaritysearchResponse
    # Configuring the Prompt for the LLM
    # TODO: EDIT THIS PROMPT TO OPTIMIZE FOR YOUR USE CASE
    prompt_data = f"""\n\nHuman: You are an AI assistant that will help people answer questions they have about [YOUR TOPIC]. Answer the provided question to the best of your ability using the information provided in the Context. 
    Summarize the answer and provide sources to where the relevant information can be found. 
    Include this at the end of the response.
    Provide information based on the context provided.
    Format the output in human readable format - use paragraphs and bullet lists when applicable
    Answer in detail with no preamble
    If you are unable to answer accurately, please say so.
    Please mention the sources of where the answers came from by referring to page numbers, specific books and chapters!

    Question: {userQuery}

    Here is the text you should use as context: {similaritysearchResponse}

    \n\nAssistant:

    """
    # Configuring the model parameters, preparing for inference
    # TODO: TUNE THESE PARAMETERS TO OPTIMIZE FOR YOUR USE CASE
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_data
                    }
                ]
            }
        ]
    }
    # formatting the prompt as a json string
    json_prompt = json.dumps(prompt)
    # invoking Claude3, passing in our prompt
    response = bedrock.invoke_model(body=json_prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                                    accept="application/json", contentType="application/json")
    # getting the response from Claude3 and parsing it to return to the end user
    response_body = json.loads(response.get('body').read())
    # the final string returned to the end user
    answer = response_body['content'][0]['text']
    # returning the final string to the end user
    return answer