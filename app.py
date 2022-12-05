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
BUILDS_URL = API_URL + 'data/'

# get list of builds
builds = prism_metadata.get_data_from_db(
    endpoint_url=BUILDS_URL,
    user_key=API_KEY,
    fields=['name', 'data_build_types', 'role']
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


# inputs


if run and build:

    build_path = "s3://macchiato.clue.io/builds/" + build + "/build/"
    if fs.exists(build_path):
        file_list = fs.ls(build_path)
        qc_file = 's3://' + get_qc_table(file_list)
        print('QC file found: ' + qc_file)
        mfi_file = 's3://' + get_lvl3(file_list)
        print('MFI file found: ' + mfi_file)

        with st.spinner('Generating report...'):
            # read data
            mfi_cols = ['prism_replicate', 'pool_id', 'ccle_name', 'culture', 'pert_type', 'pert_well', 'replicate',
                        'logMFI_norm', 'logMFI', 'pert_plate', 'pert_iname', 'pert_dose']
            qc_cols = ['prism_replicate', 'ccle_name', 'pool_id', 'culture', 'pert_plate', 'ctl_vehicle_md',
                       'trt_poscon_md', 'ctl_vehicle_mad', 'trt_poscon_mad', 'ssmd', 'nnmd', 'error_rate', 'dr', 'pass']

            qc = pd.read_csv(qc_file, usecols=qc_cols)
            mfi = pd.read_csv(mfi_file, usecols=mfi_cols)

            # transform mfi dataframe
            mfi_out = mfi.pipe(df_transform.add_bc_type)
            # transform qc dataframe

            qc_out = qc.pipe(df_transform.add_pass_rates) \
                .pipe(df_transform.add_replicate)
            qc_out = df_transform.append_raw_dr(mfi, qc_out)

            # pivoted level for poscon/negcon comparison
            control_df = mfi_out.pipe(df_transform.pivot_dmso_bort)
            control_df = control_df.merge(qc_out,
                                          on=['prism_replicate',
                                              'ccle_name',
                                              'pert_plate'],
                                          how='left')

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

            st.header('QC Metrics')
            dr, ssmd = st.tabs(['Dynamic range', 'SSMD'])
            with dr:
                dr_norm, dr_raw, dr_comp = st.tabs(['Normalized', 'Raw', 'Comparison'])
                with dr_norm:
                    plotting_functions.plot_dynamic_range(qc_out, 'dr')
                with dr_raw:
                    plotting_functions.plot_dynamic_range(qc_out, 'dr_raw')
                with dr_comp:
                    plotting_functions.plot_dynamic_range_norm_raw(qc_out)
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
                                corr = plotting_functions.reshape_df_for_corrplot(mfi[mfi['pert_plate'] == plate])
                                sub_mfi = mfi[mfi['pert_plate'] == plate]
                                print(corr.head())
                                plotting_functions.plot_corrplot(df=corr,
                                                                 sub_mfi=sub_mfi)
    else:
        st.text('Build does not exist; check S3.')
