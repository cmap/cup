import logging
import math
import os
from pathlib import Path
import pandas as pd
import s3fs
import streamlit as st
import df_transform
import plotting_functions
from metadata import prism_metadata
import boto3
import plotly.io as pio
import io
from PIL import Image

logging.basicConfig(filename='./logs/ctg_logs.log')
logging.debug('This message should go to the log file')

base_path = Path(__file__)

# config theme

st.set_page_config(layout='wide',
                   page_title='PRISM QC')

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
st.markdown(hide_table_row_index, unsafe_allow_html=True)  # hide table indices while displayed

# AWS/API setup
API_URL = 'https://api.clue.io/api/'
API_KEY = os.environ['API_KEY']
BUILDS_URL = API_URL + 'data_build_types/prism-builds'

# get list of builds
builds = prism_metadata.get_data_from_db(
    endpoint_url=BUILDS_URL,
    user_key=API_KEY,
    fields=['name']
)

fs = s3fs.S3FileSystem(anon=False)

builds_list = []
for i in builds:
    # insert filter IF here
    builds_list.append(list(i.values())[0])


# USER INPUTS

def update_params():
    st.experimental_set_query_params(option=st.session_state.qp)


query_params = st.experimental_get_query_params()

# Set selectbox value based on query param, or provide a default
ix = 0
if query_params:
    try:
        ix = builds_list.index(query_params['option'][0])
    except ValueError:
        pass

build = st.selectbox(
    "Param", builds_list, index=ix, key="qp", on_change=update_params
)

# Set query param based on selection
st.experimental_set_query_params(option=build)
run = st.button('Run')


def get_lvl3(files):
    for file in files:
        if 'LEVEL3' in file:
            return file


def get_qc_table(files):
    for file in files:
        if 'QC_TABLE' in file:
            return file


def upload_df_to_s3(df, filename, prefix, bucket_name='cup.clue.io'):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Body=csv_buffer.getvalue().encode('utf-8'), Bucket=bucket_name, Key=f"{prefix}/{filename}")
    print(f"File '{filename}' uploaded to bucket '{bucket_name}'")


def load_df_from_s3(filename, bucket_name='cup.clue.io'):
    response = s3.get_object(Bucket=bucket_name, Key=f"{build}/{filename}")
    csv_bytes = response['Body'].read()
    csv_buffer = io.StringIO(csv_bytes.decode())
    df = pd.read_csv(csv_buffer)
    return df


def load_plot_from_s3(filename, prefix, bucket_name='cup.clue.io'):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
    fig_json = response['Body'].read().decode('utf-8')
    fig = pio.from_json(fig_json)
    st.plotly_chart(fig)


def load_image_from_s3(filename, prefix, bucket_name='cup.clue.io'):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
    content = response['Body'].read()

    # Load image data from buffer
    img_buffer = io.BytesIO(content)
    img = Image.open(img_buffer)

    # Display image in Streamlit
    st.image(img)


# Inputs
if run and build:

    # Compare expected plots to files on s3
    s3 = boto3.client('s3')
    bucket = 'cup.clue.io'
    prefix = build

    expected_plots = [f"{prefix}/{filename}" for filename in
                      ['dr_norm.json', 'dr_raw.json', 'pass_by_plate.json', 'pass_by_pool.json',
                       'qc_out.csv', 'mfi_out.csv', 'control_df.csv', 'pass_fail_table.csv',
                       'dmso_perf.png']]
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        objects = response['Contents']
        existing_plots = [obj['Key'] for obj in objects]
        print(f"Found {len(existing_plots)} objects with prefix '{prefix}' in bucket '{bucket}'")
    else:
        print(f"No objects with prefix '{prefix}' found in bucket '{bucket}'")
        existing_plots = []

    if set(expected_plots) == set(existing_plots):
        print(f"All of the necessary plots already exist, generating output.")

        with st.spinner('Loading report...'):
            st.title('PRISM QC report')
            st.header(build)

            # Plot pass rates
            st.header('Pass rates')
            by_plate, by_pool = st.tabs(['By plate', 'By pool'])
            with by_plate:
                load_plot_from_s3(filename= 'pass_by_plate.json', prefix=build)
            with by_pool:
                load_plot_from_s3(filename='pass_by_pool.json', prefix=build)

            st.header('Pass/fail table')
            pass_fail = load_df_from_s3('pass_fail_table.csv')
            st.table(pass_fail.reset_index(drop=True).style.bar(subset=['Pass'], color='#006600', vmin=0, vmax=100).bar(
                subset=['Fail both', 'Fail error rate', 'Fail dynamic range'], color='#d65f5f', vmin=0, vmax=100))


            st.header('Dynamic range')
            dr_norm, dr_raw = st.tabs(['Normalized', 'Raw'])
            with dr_norm:
                load_plot_from_s3(filename='dr_norm.json', prefix=build)
            with dr_raw:
                load_plot_from_s3(filename='dr_raw.json', prefix=build)

            st.header('DMSO performance')
            load_image_from_s3(filename='dmso_perf.png', prefix=build)

######################################################################################################################
    else:
        print(f"The necessary plots DO NOT exist, generating output.")
        build_path = "s3://macchiato.clue.io/builds/" + build + "/build/"
        if fs.exists(build_path):
            file_list = fs.ls(build_path)
            qc_file = 's3://' + get_qc_table(file_list)
            print('QC file found: ' + qc_file)
            mfi_file = 's3://' + get_lvl3(file_list)
            print('MFI file found: ' + mfi_file)

            with st.spinner('Generating report and uploading results...'):

                # Read data
                mfi_cols = ['prism_replicate', 'pool_id', 'ccle_name', 'culture', 'pert_type', 'pert_well', 'replicate',
                            'logMFI_norm', 'logMFI', 'pert_plate', 'pert_iname', 'pert_dose']
                qc_cols = ['prism_replicate', 'ccle_name', 'pool_id', 'culture', 'pert_plate', 'ctl_vehicle_md',
                           'trt_poscon_md', 'ctl_vehicle_mad', 'trt_poscon_mad', 'ssmd', 'nnmd', 'error_rate', 'dr',
                           'pass']

                qc = pd.read_csv(qc_file, usecols=qc_cols)
                mfi = pd.read_csv(mfi_file, usecols=mfi_cols)

                # Transform mfi dataframe and upload to s3
                mfi_out = mfi.pipe(df_transform.add_bc_type)

                upload_df_to_s3(df=mfi_out,
                                prefix=build,
                                filename='mfi_out.csv')

                # Transform qc dataframe and upload to s3
                qc_out = qc.pipe(df_transform.add_pass_rates) \
                    .pipe(df_transform.add_replicate)
                qc_out = df_transform.append_raw_dr(mfi, qc_out)

                upload_df_to_s3(df=qc_out,
                                prefix=build,
                                filename='qc_out.csv')

                # Pivot table for poscon/negcon comparison and upload to s3
                control_df = mfi_out.pipe(df_transform.pivot_dmso_bort)
                control_df = control_df.merge(qc_out,
                                              on=['prism_replicate',
                                                  'ccle_name',
                                                  'pert_plate'],
                                              how='left')

                upload_df_to_s3(df=control_df,
                                prefix=build,
                                filename='control_df.csv')

                # Generate and save plots
                plotting_functions.plot_pass_rates_by_plate(df=qc_out,
                                                            build=build,
                                                            filename='pass_by_plate.json')

                plotting_functions.plot_pass_rates_by_pool(df=qc_out,
                                                            build=build,
                                                            filename='pass_by_pool.json')

                plotting_functions.plot_dynamic_range(df=qc_out,
                                                      metric='dr',
                                                      build=build,
                                                      filename='dr_norm.json')

                plotting_functions.plot_dynamic_range(df=qc_out,
                                                      metric='dr_raw',
                                                      build=build,
                                                      filename='dr_raw.json')

                plotting_functions.plot_dmso_performance(df=mfi_out,
                                                         build=build,
                                                         filename='dmso_perf.png')

                df_transform.generate_pass_fail_tbl(mfi, qc, prefix=build)

        else:
            st.text('Build does not exist; check S3.')
