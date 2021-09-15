# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.8
# FROM public.ecr.aws/ubuntu/ubuntu:20.04

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./
RUN  pip3 install mediapipe opencv-python-headless
# Install the python requirements from requirements.txt
RUN python3.8 -m pip install -r requirements.txt

RUN mkdir ./model
COPY model*  ./model/

COPY app/app.py ./

# Set the CMD to your handler
CMD ["app.lambda_handler"]