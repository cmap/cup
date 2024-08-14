import pandas as pd
import boto3
import io
import json
import botocore
import streamlit as st
from PIL import Image
import plotly.io as pio


def get_file(files, file_string):
    for file in files:
        if file_string in file:
            return file
    # Only print the not found message if no file is found after checking all files
    print(f"File {file_string} not found, skipping.")
    return None


def upload_df_to_s3(df, filename, prefix, bucket_name='cup.clue.io'):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Body=csv_buffer.getvalue().encode('utf-8'), Bucket=bucket_name, Key=f"{prefix}/{filename}")
    print(f"File '{filename}' uploaded to bucket '{bucket_name}'")


def load_df_from_s3(filename, prefix, bucket_name='cup.clue.io'):
    s3 = boto3.client('s3')
    try:
        # Check if the object exists by retreiving metadata
        s3.head_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")

        # If exustsm proceed
        response = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
        csv_bytes = response['Body'].read()
        csv_buffer = io.StringIO(csv_bytes.decode())
        df = pd.read_csv(csv_buffer)
        return df

    except botocore.exceptions.ClientError as e:
        st.text('There is at least one file missing, please regenerate report and try again.')


def load_json_table_from_s3(filename, prefix, bucket_name='cup.clue.io'):
    s3 = boto3.client('s3')
    try:
        # Check if the object exists by retreiving metadata
        s3.head_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")

        # If no exceptions, proceed
        response = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
        json_bytes = response['Body'].read()
        json_str = json_bytes.decode()
        df = pd.read_json(io.StringIO(json_str), orient='records')
        return df

    except botocore.exceptions.ClientError as e:
        st.text('There is at least one file missing, please regenerate report and try again.')


def load_plot_from_s3(filename, prefix, bucket_name='cup.clue.io'):
    s3 = boto3.client('s3')
    try:
        # Check if the object exists by retreiving metadata
        s3.head_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")

        # If no exception, proceed
        response = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
        fig_json = response['Body'].read().decode('utf-8')
        fig = pio.from_json(fig_json)
        st.plotly_chart(fig)

    except botocore.exceptions.ClientError as e:
        st.text('There is at least one file missing, please regenerate report and try again.')


def load_image_from_s3(filename, prefix, bucket_name='cup.clue.io'):
    s3 = boto3.client('s3')
    try:
        # Check if the object exists by retrieving metadata
        s3.head_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
        print(f"File '{filename}' found in bucket '{bucket_name}'")

        # If the previous line didn't raise an exception, proceed with fetching the actual object
        response = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
        content = response['Body'].read()

        # Load image data from buffer
        img_buffer = io.BytesIO(content)
        img = Image.open(img_buffer)

        # Display image in Streamlit
        st.image(img)

    except botocore.exceptions.ClientError as e:
        st.text('There is at least one file missing, please regenerate report and try again.')


def check_file_exists(bucket_name, file_name):
    s3 = boto3.client('s3')

    try:
        s3.head_object(Bucket=bucket_name, Key=file_name)
        return True
    except Exception as e:
        return False


def write_json_to_s3(bucket, filename, data, prefix):
    s3 = boto3.client('s3')
    # Convert your data to a JSON string
    json_data = json.dumps(data)
    # Convert your JSON string to bytes
    json_bytes = json_data.encode()
    # Write the JSON data to an S3 object
    s3.put_object(Body=json_bytes, Bucket=bucket, Key=f"{prefix}/{filename}")


def write_json_table_to_s3(bucket, filename, data, prefix):
    s3 = boto3.client('s3')

    # Convert your JSON string to bytes
    json_bytes = data.encode()

    # Write the JSON data to an S3 object with specified content type
    s3.put_object(Body=json_bytes, Bucket=bucket, Key=f"{prefix}/{filename}", ContentType='application/json')


def read_json_from_s3(bucket_name, filename, prefix):
    s3 = boto3.client('s3')
    # Get the object from S3
    s3_object = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
    # Get the body of the object (the data content)
    s3_object_body = s3_object['Body'].read()
    # Convert bytes to string
    string_data = s3_object_body.decode('utf-8')
    # Convert string data to dictionary
    dict_data = json.loads(string_data)
    return dict_data
