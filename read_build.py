import os
import pandas as pd
import boto3
from io import StringIO, BytesIO
import gzip
import cmapPy.pandasGEXpress
from cmapPy.pandasGEXpress.parse import parse
import tempfile
import shutil
from botocore.exceptions import ClientError


class Build:
    def __init__(self):
        self.inst = None
        self.mfi = None
        self.lfc = None
        self.lfc_combat = None
        self.lfc_rep = None


def parse_gctx(filepath):
    if not (filepath.endswith('.gct') or filepath.endswith('.gctx')):
        raise ValueError("File to parse must be .gct or .gctx!")

    # Parsing code for .gct or .gctx file here
    df = parse(filepath)

    df.data_df['rid'] = df.data_df.index
    df = df.data_df.melt(id_vars=['rid'])
    return df


def read_build_from_s3(build_name, s3_bucket='macchiato.clue.io', prefix='builds', suffix='build', exclude_base=False,
                       data_levels=None):
    if data_levels is None:
        data_levels = ['inst', 'mfi', 'lfc', 'lfc_combat', 'lfc_rep', 'lfc_rep_combat', 'qc', 'count', 'cell']
    s3 = boto3.client('s3')

    build_instance = Build()

    # List all objects in the specified S3 bucket with the given prefix
    objects = s3.list_objects_v2(Bucket=s3_bucket, Prefix=f"{prefix}/{build_name}/{suffix}/")

    if 'Contents' not in objects:
        print("No files found in the specified S3 path.")
        return None

    for obj in objects['Contents']:
        key = obj['Key']
        filename = os.path.basename(key)

        # Check for the required data level
        data_level = None
        if 'inst' in data_levels and (filename.endswith('inst_info.txt') or filename.endswith('inst_info.txt.gz')):
            data_level = 'inst'
        elif ('level3' in data_levels or 'mfi' in data_levels) and 'LEVEL3' in filename:
            data_level = 'mfi'
        elif (
                'level4' in data_levels or 'lfc' in data_levels) and 'LEVEL4_LFC' in filename and 'COMBAT' not in filename:
            data_level = 'lfc'
        elif ('level4_combat' in data_levels or 'lfc_combat' in data_levels) and 'LEVEL4_LFC_COMBAT' in filename:
            data_level = 'lfc_combat'
        elif ('level5' in data_levels or 'lfc_rep' in data_levels) and 'LEVEL5_LFC' in filename and 'COMBAT' not in filename:
            data_level = 'lfc_rep'
        elif ('level5_combat' in data_levels or 'lfc_rep_combat' in data_levels) and 'LEVEL5_LFC_COMBAT' in filename:
            data_level = 'lfc_rep_combat'
        elif 'qc' in data_levels and 'QC_TABLE' in filename:
            data_level = 'qc'
        elif 'count' in data_levels and 'COUNT' in filename:
            data_level = 'count'
        elif 'cell' in data_levels and 'cell_info' in filename:
            data_level = 'cell'

        # Read the data from S3 and load it into a pandas DataFrame
        if data_level:
            obj_data = s3.get_object(Bucket=s3_bucket, Key=key)
            content = obj_data['Body']

            if filename.endswith('.gz') and not filename.endswith('.gctx.gz'):
                with gzip.GzipFile(fileobj=BytesIO(content.read())) as gz:
                    delimiter = '\t' if filename.endswith('.txt.gz') else ','
                    print(f"Reading {data_level} file {filename}")
                    df = pd.read_csv(gz, sep=delimiter, low_memory=False)
            elif filename.endswith('.gctx.gz'):
                with gzip.open(content, 'rb') as gz_file:
                    binary_data = gz_file.read()
                    print(f"Reading {data_level} file {filename}")

                # Create a temporary directory to store the uncompressed file
                temp_dir = tempfile.mkdtemp()

                # Construct the temporary file path with .gctx extension
                temp_filepath = os.path.join(temp_dir, "temp_file.gctx")

                try:
                    # Write the binary data to the temporary file
                    with open(temp_filepath, 'wb') as temp_file:
                        temp_file.write(binary_data)

                    # Call the parse_gctx function with the temporary file path
                    df = parse_gctx(temp_filepath)

                    # Rest of the processing code for the parsed DataFrame
                    # ...
                finally:
                    # Cleanup the temporary directory
                    shutil.rmtree(temp_dir)
            else:
                delimiter = '\t' if filename.endswith('.txt') else ','
                print(f"Reading {data_level} file {filename}")
                df = pd.read_csv(BytesIO(content.read()), sep=delimiter)

            if exclude_base and 'pert_plate' in df.columns:
                df = df[df['pert_plate'] != 'BASE']

            # Set the data level attribute of the build instance
            setattr(build_instance, data_level, df)

    return build_instance


def read_table_from_s3(s3_uri: str) -> pd.DataFrame:
    # Parse the S3 URI
    bucket_name, key = s3_uri.replace("s3://", "").split("/", 1)

    # Initialize the S3 client
    s3 = boto3.client("s3")

    # Check if the S3 object exists
    try:
        s3.head_object(Bucket=bucket_name, Key=key)
    except ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the object does not exist.
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            print(f"S3 object {s3_uri} does not exist.")
            return None
        else:
            # Propagate other exceptions.
            raise

    print(f"Reading file {s3_uri}")

    # Download the file from S3 into a bytes buffer
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    data = BytesIO(obj["Body"].read())

    # Check if the file is gzipped
    if key.endswith('.gz'):
        data = gzip.GzipFile(fileobj=data)

    # Determine the delimiter
    line = data.readline().decode("utf-8")
    data.seek(0)  # Reset the buffer position to the beginning
    delimiter = "," if "," in line else "\t" if "\t" in line else None
    if delimiter is None:
        raise ValueError("Unable to determine the delimiter in the file.")

    # Read the file into a pandas DataFrame using the detected delimiter
    df = pd.read_csv(data, delimiter=delimiter)
    return df


