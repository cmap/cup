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


def check_file_exists(bucket_name, file_name):
    s3 = boto3.client('s3')

    try:
        s3.head_object(Bucket=bucket_name, Key=file_name)
        return True
    except Exception as e:
        return False


# Inputs
if run and build:

    # Compare expected plots to files on s3
    s3 = boto3.client('s3')
    bucket = 'cup.clue.io'
    prefix = build

    expected_plots = [f"{prefix}/{filename}" for filename in
                      ['dr_norm.json', 'dr_raw.json', 'pass_by_plate.json', 'pass_by_pool.json',
                       'qc_out.csv', 'mfi_out.csv', 'control_df.csv', 'pass_fail_table.csv',
                       'dmso_perf.png', 'plate_dist_raw.png', 'plate_dist_norm.png', 'logMFI_heatmaps.png',
                       'logMFI_norm_heatmaps.png', 'liverplot.json', 'banana_norm.json', 'banana_raw.json',
                       'dr_er.json']]
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        objects = response['Contents']
        existing_plots = [obj['Key'] for obj in objects]
        print(f"Found {len(existing_plots)} objects with prefix '{prefix}' in bucket '{bucket}'")
    else:
        print(f"No objects with prefix '{prefix}' found in bucket '{bucket}'")
        existing_plots = []

    if set(expected_plots).issubset(set(existing_plots)):
        print(f"All of the necessary plots already exist, generating output.")

        with st.spinner('Loading report...'):
            st.title('PRISM QC report')
            st.header(build)

            # Plot pass rates
            st.header('Pass rates')
            by_plate, by_pool = st.tabs(['By plate', 'By pool'])
            with by_plate:
                load_plot_from_s3(filename='pass_by_plate.json', prefix=build)
            with by_pool:
                load_plot_from_s3(filename='pass_by_pool.json', prefix=build)

            # Show pass/fail table
            st.header('Pass/fail table')
            pass_fail = load_df_from_s3('pass_fail_table.csv')
            st.table(pass_fail.reset_index(drop=True).style.bar(subset=['Pass'], color='#006600', vmin=0, vmax=100).bar(
                subset=['Fail both', 'Fail error rate', 'Fail dynamic range'], color='#d65f5f', vmin=0, vmax=100))

            # Plot dynamic range
            st.header('Dynamic range')
            dr_norm, dr_raw = st.tabs(['Normalized', 'Raw'])
            with dr_norm:
                load_plot_from_s3(filename='dr_norm.json', prefix=build)
            with dr_raw:
                load_plot_from_s3(filename='dr_raw.json', prefix=build)

            # Liver plots
            st.header('Liver plots')
            load_plot_from_s3(filename='liverplot.json', prefix=build)

            # Banana plots
            st.header('Banana plots')
            banana_normalized, banana_raw = st.tabs(['Normalized', 'Raw'])
            with banana_normalized:
                load_plot_from_s3('banana_norm.json', prefix=build)
            with banana_raw:
                load_plot_from_s3('banana_raw.json', prefix=build)

            # Plot plate distributions
            st.header('Plate distributions')
            norm, raw = st.tabs(['Normalized', 'Raw'])
            with norm:
                load_image_from_s3('plate_dist_norm.png', prefix=build)
            with raw:
                load_image_from_s3('plate_dist_raw.png', prefix=build)

            # Dynamic range versus error rate
            st.header('Error rate and dynamic range')
            load_plot_from_s3('dr_er.json', prefix=build)

            # Plot DMSO performance
            st.header('DMSO performance')
            load_image_from_s3(filename='dmso_perf.png', prefix=build)

            # Plot heatmaps
            st.header('logMFI')
            raw, norm = st.tabs(['Raw', 'Normalized'])
            with raw:
                load_image_from_s3(filename='logMFI_heatmaps.png', prefix=build)
            with norm:
                load_image_from_s3(filename='logMFI_norm_heatmaps.png', prefix=build)

            # Plot correlations

            if check_file_exists(file_name=f"{build}/corrplot_raw.png", bucket_name='cup.clue.io'):
                st.header('Correlations')
                raw, norm = st.tabs(['Raw', 'Normalized'])
                with raw:
                    load_image_from_s3(filename='corrplot_raw.png', prefix=build)
                with norm:
                    load_image_from_s3(filename='corrplot_norm.png', prefix=build)

            # Compare historical performance
            st.header('Historical performance')
            raw, norm = st.tabs(['Raw', 'Normalized'])
            with raw:
                load_plot_from_s3('historical_mfi_raw.json', prefix='historical')
            with norm:
                load_plot_from_s3('historical_mfi_norm.json', prefix='historical')


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

                # Add to historical df if needed
                current = mfi[['pert_type', 'logMFI', 'logMFI_norm']].dropna()
                current = current[current.pert_type.isin(['trt_poscon','ctl_vehicle'])]
                current['build'] = build

                if check_file_exists(bucket_name='cup.clue.io',
                                     file_name='historical/historical_mfi.csv'):
                    response = s3.get_object(Bucket='cup.clue.io', Key='historical/historical_mfi.csv')
                    csv_bytes = response['Body'].read()
                    csv_buffer = io.StringIO(csv_bytes.decode())
                    hist = pd.read_csv(csv_buffer)

                    hist_builds = list(hist['build'].unique())

                    if build not in hist_builds:
                        print(f"Build {build} is being added to historical MFI data")
                        df = pd.concat([current, hist])
                        upload_df_to_s3(df,
                                        prefix='historical',
                                        filename='historical_mfi.csv')
                    else:
                        print(f"Build {build} already exists in historical MFI data, skipping upload")
                else:
                    print(f"No historical data found, creating historical MFI csv with {build} data")
                    upload_df_to_s3(current, filename='historical_mfi.csv', prefix='historical')

                # Download historical data
                response = s3.get_object(Bucket='cup.clue.io', Key='historical/historical_mfi.csv')
                csv_bytes = response['Body'].read()
                csv_buffer = io.StringIO(csv_bytes.decode())
                hist = pd.read_csv(csv_buffer)

                # Make the plot comparing historical performance
                plotting_functions.plot_historical_perf(df=hist,
                                                        metric='logMFI',
                                                        filename='historical_mfi_raw.json')

                plotting_functions.plot_historical_perf(df=hist,
                                                        metric='logMFI_norm',
                                                        filename='historical_mfi_norm.json')

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
                control_df['replicate'] = control_df['prism_replicate'].str.split('_').str[3]

                upload_df_to_s3(df=control_df,
                                prefix=build,
                                filename='control_df.csv')

                # Generate replicate correlation df

                corr_df_norm = mfi[~mfi.pert_plate.str.contains('BASE')].pivot_table(columns=['replicate'],
                                                                                     values='logMFI_norm',
                                                                                     index=['pert_iname', 'pert_dose',
                                                                                            'pert_plate']).dropna().reset_index()

                corr_df_raw = mfi[~mfi.pert_plate.str.contains('BASE')].pivot_table(columns=['replicate'],
                                                                                    values='logMFI',
                                                                                    index=['pert_iname', 'pert_dose',
                                                                                           'pert_plate']).dropna().reset_index()

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

                plotting_functions.plot_distributions_by_plate(mfi_out,
                                                               value='logMFI_norm',
                                                               build=build,
                                                               filename='plate_dist_norm.png')

                plotting_functions.plot_distributions_by_plate(mfi_out,
                                                               value='logMFI',
                                                               build=build,
                                                               filename='plate_dist_raw.png')

                plotting_functions.plot_heatmaps(mfi_out,
                                                 metric='logMFI',
                                                 build=build)

                plotting_functions.plot_heatmaps(mfi_out,
                                                 metric='logMFI_norm',
                                                 build=build)

                plotting_functions.plot_liver_plots(qc_out,
                                                    build=build,
                                                    filename='liverplot.json')

                plotting_functions.plot_banana_plots(control_df,
                                                     build=build,
                                                     x='ctl_vehicle_med_norm',
                                                     y='trt_poscon_med_norm',
                                                     filename='banana_norm.json')

                plotting_functions.plot_banana_plots(control_df,
                                                     build=build,
                                                     x='ctl_vehicle_med',
                                                     y='trt_poscon_med',
                                                     filename='banana_raw.json')

                plotting_functions.plot_dr_error_rate(qc_out,
                                                      build=build,
                                                      filename='dr_er.json')

                if len(mfi.replicate.unique()) > 1:
                    plotting_functions.plot_corrplot(df=corr_df_norm,
                                                     mfi=mfi,
                                                     build=build,
                                                     filename='corrplot_norm.png')

                    plotting_functions.plot_corrplot(df=corr_df_raw,
                                                 mfi=mfi,
                                                 build=build,
                                                 filename='corrplot_raw.png')

                df_transform.generate_pass_fail_tbl(mfi, qc, prefix=build)

        else:
            st.text('Build does not exist; check S3.')
