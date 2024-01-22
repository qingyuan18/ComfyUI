import websocket
import uuid
import json
import urllib.request
import urllib.parse
import base64
from pydantic import BaseModel, Field
from PIL import Image
import io
import sys
import boto3
import os
import traceback
WORKING_DIR="/tmp"

def get_bucket_and_key(s3uri):
    """
    get_bucket_and_key is helper function
    """
    pos = s3uri.find('/', 5)
    bucket = s3uri[5: pos]
    key = s3uri[pos + 1:]
    return bucket, key

def write_gif_to_s3(images,output_s3uri=""):
    """
    write image to s3 bucket
    """
    s3_client = boto3.client('s3')
    s3_bucket = os.environ.get("s3_bucket", "sagemaker-us-west-2-687912291502")
    prediction = []
    default_output_s3uri = f's3://{s3_bucket}/comfyui_output/images/'
    if output_s3uri is None or output_s3uri=="":
        output_s3uri=default_output_s3uri    
    for node_id in images:
        for image_file in images[node_id]:
            bucket, key = get_bucket_and_key(output_s3uri)
            #GIF_LOCATION = "{}/Comfyui_{}.gif".format(WORKING_DIR, node_id)
            #print(GIF_LOCATION)
            key = f'{key}Comfyui_{node_id}.gif'
            #with open(GIF_LOCATION, "wb") as binary_file:
                # Write bytes to file
            #    binary_file.write(image_data.read())
            s3_client.upload_file(
                Filename=image_file,
                Bucket=bucket,
                Key=key
            )
            print('image: ', f's3://{bucket}/{key}')
            prediction.append(f's3://{bucket}/{key}')
    return prediction

def write_imgage_to_s3(images,output_s3uri=""):
    """
    write image to s3 bucket
    """
    s3_client = boto3.client('s3')
    s3_bucket = os.environ.get("s3_bucket", "")
    key = "/comfyui_output/images/"
    prediction = []
    default_output_s3uri = f's3://{s3_bucket}/comfyui_output/images/'
    if output_s3uri is None or output_s3uri=="":
        output_s3uri=default_output_s3uri    
    for node_id in images:
        for image_file in images[node_id]:
            image_data = open(image_file, 'rb')
            image = Image.open(io.BytesIO(image_data.read()))
            bucket, key = get_bucket_and_key(output_s3uri)
            key = f'{key}{uuid.uuid4()}.jpg'
            buf = io.BytesIO()
            image.save(buf, format='JPEG')
            image_data.close()
            s3_client.put_object(
                Body=buf.getvalue(),
                Bucket=bucket,
                Key=key,
                ContentType='image/jpeg',
                Metadata={
                    "seed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            )
            print('image: ', f's3://{bucket}/{key}')
            prediction.append(f's3://{bucket}/{key}')
    return prediction

class InferenceOpt(BaseModel):
    prompt_id:str = ""
    client_id:str = ""
    prompt: dict = None
    negative_prompt: str = ""
    steps: int = 20
    inference_type: str = "txt2img"
    method:str = ""
