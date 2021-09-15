import json
import boto3
import numpy as np
import tensorflow as tf
import io
import cv2
import datetime


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# Load the prediction model
model = tf.keras.models.load_model("model")

s3 = boto3.client('s3')
dynamo = boto3.resource("dynamodb")

def read_and_decode_from_s3(bucket_name, key):
    """
    This function read the uploaded image from s3, uses opencv to decode and resize it
    accordding to our model input size. We reshape the resized image
    as (1,IMAGE_WIDTH, IMAGE_HEIGHT,3) to create a batch of size=1.

    """

    file_obj = s3.get_object(Bucket=bucket_name, Key=key)
    file_content = file_obj["Body"].read()

    np_array = np.fromstring(file_content, np.uint8)
    # print("array type: ", type(np_array))

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    # print(img.shape)
    img = cv2.resize(img, IMAGE_SHAPE, interpolation = cv2.INTER_AREA)
    # print(img.shape)
    img = img.reshape(1,IMAGE_WIDTH, IMAGE_HEIGHT,3)
    img = img/255.

    return img


def write_item(name, url, prediction, time):
    """
    This function create an item with the predicted image result and write it into dynamodb table.
    """

    table = dynamo.Table('tf-lambda-inferenceTable')
    response = table.put_item(
       Item={
            'id': name,
            'time': time,
            'prediction': prediction,
            'url': url
        }
    )
    print("DONE WRITTING INTO DB")
    return response



def lambda_handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # get the image from s3 bucket and transform it 
    img = read_and_decode_from_s3(bucket_name, key)

    # predict the image class using the loaded model
    pred = model.predict(img)
    pred = pred[0][0]
    print(pred)
    prediction = 'true' if pred > 0.5 else 'false'


    # =================== write to dynamodb =====================
    currentDT = datetime.datetime.now()
    time = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    url = 's3://' + bucket_name + '/' + key
    
    #remove the .jpg in the key to get the ID
    key = key.split('.')[0]
    
    response = write_item(key, url, prediction, time)


    obj = {
        "prediction": prediction
    }

    return {
        'statusCode': 200,
        'body': json.dumps(obj)
    } 
