import streamlit as st
import pandas as pd
import plotting_functions
import df_transform
from datetime import date
from pathlib import Path
import logging
import s3fs
import math
import os
from metadata import prism_metadata

logging.basicConfig(filename='./logs/ctg_logs.log')
logging.debug('This message should go to the log file')

base_path = Path(__file__)

# config theme

st.set_page_config(layout='wide')

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
BUILDS_URL = API_URL + 'data/'

# get list of builds
builds = prism_metadata.get_data_from_db(
    endpoint_url=BUILDS_URL,
    user_key=API_KEY,
    fields=['name','data_build_types', 'role']
)

fs = s3fs.S3FileSystem(anon=False)

builds_list = []
for i in builds:
    # insert filter IF here
    builds_list.append(list(i.values())[0])
print(builds_list)

# build_list = fs.ls('s3://macchiato.clue.io/builds')
# builds = []
# for i in build_list:
#    res = i.split('/')[2]
#    builds.append(res)


# USER INPUTS

build = st.selectbox("Select build", builds_list)
run = st.button('Run')

# find files on AWS


def get_lvl3(files):
    for file in files:
        if 'LEVEL3' in file:
            return file


def get_qc_table(files):
    for file in files:
        if 'QC_TABLE' in file:
            return file


# inputs


if run and build:

    build_path = "s3://macchiato.clue.io/builds/" + build + "/build/"
    file_list = fs.ls(build_path)
    qc_file = 's3://' + get_qc_table(file_list)
    print('QC file found: ' + qc_file)
    mfi_file = 's3://' + get_lvl3(file_list)
    print('MFI file found: ' + mfi_file)

    with st.spinner('Generating report...'):
        # read data
        mfi_cols = ['prism_replicate', 'pool_id', 'ccle_name', 'culture', 'pert_type', 'pert_well', 'replicate',
                    'logMFI_norm', 'logMFI', 'pert_plate']
        qc_cols = ['prism_replicate', 'ccle_name', 'pool_id', 'culture', 'pert_plate', 'ctl_vehicle_md',
                   'trt_poscon_md', 'ctl_vehicle_mad', 'trt_poscon_mad', 'ssmd', 'nnmd', 'error_rate', 'dr', 'pass']

        qc = pd.read_csv(qc_file, usecols=qc_cols)
        mfi = pd.read_csv(mfi_file, usecols=mfi_cols)

        # transform mfi dataframe
        mfi_out = mfi.pipe(df_transform.add_bc_type)
        # transform qc dataframe

        qc_out = qc.pipe(df_transform.add_pass_rates)

        # pivoted level for poscon/negcon comparison
        control_df = mfi_out.pipe(df_transform.pivot_dmso_bort)
        control_df = control_df.merge(qc_out,
                                      on=['prism_replicate',
                                          'ccle_name',
                                          'pert_plate'],
                                      how='left')

        # OUTPUT

        st.title('QC report')
        st.header(date.today())

        st.header('Pass rates')
        by_plate, by_pool = st.tabs(['By plate', 'By pool'])
        with by_plate:
            plotting_functions.plot_pass_rates_by_plate(qc_out)
        with by_pool:
            plotting_functions.plot_pass_rates_by_pool(qc_out)

        st.header('QC Metrics')
        lum_pert, ssmd = st.tabs(['Dynamic range', 'SSMD'])
        with lum_pert:
            plotting_functions.plot_dynamic_range(qc_out)
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

        st.subheader('Pass/Fail')
        pass_fail = df_transform.generate_pass_fail_tbl(mfi=mfi_out,
                                                        qc=qc_out)
        st.table(pass_fail.reset_index(drop=True).style.bar(subset=['Pass'], color='#006600', vmin=0, vmax=100).bar(
            subset=['Fail both', 'Fail error rate', 'Fail dynamic range'], color='#d65f5f', vmin=0, vmax=100))

        st.header('Banana plots (raw)')
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

        st.header('Banana plots (normalized)')
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

        st.header('Liver plots')
        tab_labels = qc_out.pert_plate.unique().tolist()
        n = 0
        for pert_plate in st.tabs(tab_labels):
            with pert_plate:
                plate = tab_labels[n]
                n += 1
                plotting_functions.plot_liver_plots(qc_out[qc_out.pert_plate == plate])

        st.header('Plate distributions')
        tab_labels = mfi_out.pert_plate.unique().tolist()
        n = 0
        for pert_plate in st.tabs(tab_labels):
            with pert_plate:
                plate = tab_labels[n]
                n += 1
                data = mfi_out[mfi_out.pert_plate == plate]
                height = math.ceil(data.prism_replicate.unique().shape[0] / 3) * 400
                plotting_functions.plot_distributions_by_plate(data,
                                                               height=height)
