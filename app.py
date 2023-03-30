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

import plotly.express as px

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
# API_URL = os.environ['API_URL']
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

# build = st.selectbox("Select build", builds_list)

# callback to update query param on selectbox change
def update_params():
    st.experimental_set_query_params(option=st.session_state.qp)


query_params = st.experimental_get_query_params()

# set selectbox value based on query param, or provide a default
ix = 0
if query_params:
    try:
        ix = builds_list.index(query_params['option'][0])
    except ValueError:
        pass

build = st.selectbox(
    "Param", builds_list, index=ix, key="qp", on_change=update_params
)

# set query param based on selection
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


def load_df_from_s3(filename):
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



# Inputs
if run and build:

    s3 = boto3.client('s3')
    bucket_name = 'cup.clue.io'
    prefix = build
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    filenames = ['control_df.csv',
                 'mfi_out.csv',
                 'qc_out.csv']

    build_path = "s3://macchiato.clue.io/builds/" + build + "/build/"
    if fs.exists(build_path):
        file_list = fs.ls(build_path)
        qc_file = 's3://' + get_qc_table(file_list)
        print('QC file found: ' + qc_file)
        mfi_file = 's3://' + get_lvl3(file_list)
        print('MFI file found: ' + mfi_file)

        with st.spinner('Generating report...'):
            # Check if precomputed files exist. If so, read them in. If not, generate them
            if 'Contents' in response:
                existing_files = [obj['Key'].split('/')[-1] for obj in response['Contents']]
                all_files_exist = True
                for filename in filenames:
                    if filename not in existing_files:
                        all_files_exist = False
                        print(f"File '{filename}' does not exist in bucket '{bucket_name}' with prefix '{prefix}'")
                        break
                if all_files_exist:
                    print(f"All files exist in bucket '{bucket_name}' with prefix '{prefix}'")
                    # Load existing dataframes
                    qc_out = load_df_from_s3('qc_out.csv')
                    mfi_out = load_df_from_s3('mfi_out.csv')
                    control_df = load_df_from_s3('control_df.csv')
                    generate_dataframes = False
                    generate_plots = False
                else:
                    print(f"Not all files exist in bucket '{bucket_name}' with prefix '{prefix}', computing dataframes")
                    generate_dataframes = True
                    generate_plots = True
            else:
                print(f"No objects with prefix '{prefix}' exist in bucket '{bucket_name}', computing dataframes")
                generate_dataframes = True
                generate_plots = True

            # Generate the dataframes if necessary
            if generate_dataframes:
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

            # OUTPUT
            st.title('PRISM QC report')
            st.header(build)

            st.header('Pass rates')
            st.write(
                """
                
                Passing rates are determined on a plate and cell line basis by a combination of 2 QC metrics, 
                dynamic range and error rate. 
                
                
                Thresholds are:


                **Dynamic range** > -$log{_2}{0.3}$ (~1.74)

                **Error rate** ≤ 0.05
                
                Note: there is a third possible failure mode. If 2 replicates fail by the above metrics, the third 
                replicate will also be flagged as a failure. In this case, that cell line will be excluded from that 
                plate. 
               
                """)

            by_plate, by_pool = st.tabs(['By plate', 'By pool'])
            with by_plate:
                plotting_functions.plot_pass_rates_by_plate(qc_out)
            with by_pool:
                plotting_functions.plot_pass_rates_by_pool(qc_out)

            st.subheader('Raw pass/fail')
            st.write(
                """
                
                In this table, the pass rates are calculated without regard to performance within sets of replicates. 
                Therefore, the numbers you see here may differ slightly from the graphs shown above. However, 
                this gives a more accurate representation of the performance of individual plates. 
                
                """)
            pass_fail = df_transform.generate_pass_fail_tbl(mfi=mfi_out,
                                                            qc=qc_out)
            st.table(pass_fail.reset_index(drop=True).style.bar(subset=['Pass'], color='#006600', vmin=0, vmax=100).bar(
                subset=['Fail both', 'Fail error rate', 'Fail dynamic range'], color='#d65f5f', vmin=0, vmax=100))

            if generate_plots:
                st.header('QC Metrics')
                dr, ssmd = st.tabs(['Dynamic range', 'SSMD'])
                with dr:
                    dr_norm, dr_raw, dr_comp = st.tabs(['Normalized', 'Raw', 'Comparison'])
                    with dr_norm:
                        plotting_functions.plot_dynamic_range(qc_out, 'dr', build=build, filename='dr_norm.json')
                    with dr_raw:
                        plotting_functions.plot_dynamic_range(qc_out, 'dr_raw', build=build, filename='dr_raw.json')
                    with dr_comp:
                        tab_labels = qc_out.culture.unique().tolist()
                        n = 0
                        for assay in st.tabs(tab_labels):
                            with assay:
                                assay_type = tab_labels[n]
                                n += 1
                                data = qc_out[qc_out.culture == assay_type]
                                plotting_functions.plot_dynamic_range_norm_raw(data)
                with ssmd:
                    plotting_functions.plot_ssmd(qc_out)

                st.header('Dynamic range & error rate')
                tab_labels = qc_out.pert_plate.unique().tolist()
                n = 0
                for pert_plate in st.tabs(tab_labels):
                    with pert_plate:
                        plate = tab_labels[n]
                        n += 1
                        data = qc_out[qc_out.pert_plate == plate]
                        height = math.ceil(data.prism_replicate.unique().shape[0] / 3) * 400
                        plotting_functions.plot_ssmd_error_rate(data, height=height)

                st.header('Banana plots')
                banana_normalized, banana_raw = st.tabs(['Normalized', 'Raw'])
                with banana_normalized:
                    tab_labels = control_df.pert_plate.unique().tolist()
                    n = 0
                    for pert_plate in st.tabs(tab_labels):
                        with pert_plate:
                            plate = tab_labels[n]
                            n += 1
                            data = control_df[control_df.pert_plate == plate]
                            height = math.ceil(data.prism_replicate.unique().shape[0] / 3) * 400
                            plotting_functions.plot_banana_plots(data,
                                                                 x='ctl_vehicle_med_norm',
                                                                 y='trt_poscon_med_norm',
                                                                 height=height)
                with banana_raw:
                    tab_labels = control_df.pert_plate.unique().tolist()
                    n = 0
                    for pert_plate in st.tabs(tab_labels):
                        with pert_plate:
                            plate = tab_labels[n]
                            n += 1
                            data = control_df[control_df.pert_plate == plate]
                            height = math.ceil(data.prism_replicate.unique().shape[0] / 3) * 400
                            plotting_functions.plot_banana_plots(data,
                                                                 x='ctl_vehicle_med',
                                                                 y='trt_poscon_med',
                                                                 height=height)

                st.header('Liver plots')
                cs_labels = list(qc_out.culture.unique())
                n = 0
                for culture in st.tabs(cs_labels):
                    with culture:
                        cell_set = cs_labels[n]
                        n += 1
                        plate_labels = list(qc_out.pert_plate.unique())
                        i = 0
                        for pert_plate in st.tabs(plate_labels):
                            with pert_plate:
                                plate = plate_labels[i]
                                i += 1
                                plotting_functions.plot_liver_plots(
                                    qc_out[(qc_out.pert_plate == plate) & (qc_out.culture == cell_set)])

                st.header('Plate distributions')
                norm, raw = st.tabs(['Normalized', 'Raw'])
                with norm:
                    tab_labels = mfi_out.pert_plate.unique().tolist()
                    n = 0
                    for pert_plate in st.tabs(tab_labels):
                        with pert_plate:
                            plate = tab_labels[n]
                            n += 1
                            data = mfi_out[mfi_out.pert_plate == plate]
                            height = math.ceil(data.prism_replicate.unique().shape[0] / 3) * 400
                            plotting_functions.plot_distributions_by_plate(data,
                                                                           height=height,
                                                                           value='logMFI_norm')
                with raw:
                    tab_labels = mfi_out.pert_plate.unique().tolist()
                    n = 0
                    for pert_plate in st.tabs(tab_labels):
                        with pert_plate:
                            plate = tab_labels[n]
                            n += 1
                            data = mfi_out[mfi_out.pert_plate == plate]
                            height = math.ceil(data.prism_replicate.unique().shape[0] / 3) * 400
                            plotting_functions.plot_distributions_by_plate(data,
                                                                           height=height,
                                                                           value='logMFI')
                if mfi.prism_replicate.unique().size > 1:
                    st.header('Replicate correlations')
                    cs_labels = list(mfi.culture.unique())
                    i = 0
                    for cell_set in st.tabs(cs_labels):
                        with cell_set:
                            cs = cs_labels[i]
                            i += 1
                            tab_labels = mfi[
                                (~mfi.pert_plate.str.contains('BASE')) & (mfi.culture == cs)].pert_plate.unique().tolist()
                            n = 0
                            for pert_plate in st.tabs(tab_labels):
                                with pert_plate:
                                    plate = tab_labels[n]
                                    n += 1
                                    data = mfi[(mfi.pert_plate == plate) & (mfi.culture == cs)]
                                    corr = plotting_functions.reshape_df_for_corr(data, metric='logMFI_norm')

                                    table_dim = plotting_functions.make_dimensions_for_corrtable(df=corr, sub_mfi=data)
                                    st.markdown('R<sup>2</sup> values of normalized log2(MFI) data', unsafe_allow_html=True)
                                    plotting_functions.mk_corr_table(table_dim, mfi)
                                    dimensions = plotting_functions.make_dimensions_for_corrplot(df=corr,
                                                                                                 sub_mfi=data)
                                    plotting_functions.plot_corrplot(df=corr, dim_list=dimensions)
            else:
                st.header('QC Metrics')
                dr, ssmd, comp = st.tabs(['Dynamic range', 'SSMD'])
                with dr:
                    dr_norm, dr_raw = st.tabs(['Normalized', 'Raw'])
                    with dr_norm:
                        load_plot_from_s3(filename='dr_norm.json', prefix=build)
                    with dr_raw:
                        load_plot_from_s3(filename='dr_raw.json', prefix=build)

    else:
        st.text('Build does not exist; check S3.')
