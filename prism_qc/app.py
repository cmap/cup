import streamlit as st
import pandas as pd
import plotting_functions
import df_transform
from datetime import date
from pathlib import Path
import logging
import s3fs
import math

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

# USER INPUTS

build = "PREP_C_PR500_GOOD"  # testing, will need to get from URL eventually

qc_file = st.file_uploader('Upload your qc table here', type='csv')
mfi_file = st.file_uploader('Upload your mfi table here', type='csv')

run = st.button('Run')

# aws fs setup
fs = s3fs.S3FileSystem(anon=False)
build_path = "s3://macchiato.clue.io/builds/" + build + "/build/"

# inputs

if qc_file and mfi_file and run:

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

        control_df = mfi_out.pipe(df_transform.pivot_dmso_bort)  # breaking things
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
        st.table(pass_fail.reset_index(drop=True).style.bar(subset=['Pass'], color='#d65f5f',
                                                            vmin=0, vmax=100))
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
