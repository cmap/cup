import streamlit as st
import pandas as pd
import plotting_functions
import df_transform
from datetime import date
from pathlib import Path
import logging
import s3fs

# logging.basicConfig(filename='./logs/ctg_logs.log')
# logging.debug('This message should go to the log file')

base_path = Path(__file__)

# config theme

st.set_page_config(layout='wide')

# get build information

build = "PREP_C_PR500_GOOD"  # testing, will need to get from URL eventually

# aws fs setup
fs = s3fs.S3FileSystem(anon=False)
build_path = "s3://macchiato.clue.io/builds/" + build + "/build/"
print("Build path is: " + build_path)

# USER INPUTS

if fs.exists(build_path):
    # inputs

    qc_path = '~/Desktop/PSELL_PR300P_GOOD/PSELL_PR300P_GOOD_QC_TABLE.csv'  # read qc file
    print(qc_path)
    qc = pd.read_csv(qc_path)

    mfi_path = '~/Desktop/PSELL_PR300P_GOOD/PSELL_PR300P_GOOD_LEVEL3_LMFI.csv'  # read lvl3 lmfi file
    print(mfi_path)
    mfi = pd.read_csv(mfi_path)

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

    st.header('Build distributions')
    mfi_raw, mfi_norm = st.tabs(['Raw', 'Normalized'])
    with mfi_raw:
        plotting_functions.plot_distributions(mfi_out, 'logMFI')
    with mfi_norm:
        plotting_functions.plot_distributions(mfi_out, 'logMFI_norm')

    st.header('Plate distributions')
    tab_labels = mfi_out.pert_plate.unique().tolist()
    n = 0
    for pert_plate in st.tabs(tab_labels):
        with pert_plate:
            plate = tab_labels[n]
            n += 1
            data = mfi_out[mfi_out.pert_plate == plate]
            plotting_functions.plot_distributions_by_plate(data)

    st.header('Banana plots raw')
    tab_labels = control_df.pert_plate.unique().tolist()
    n = 0
    for pert_plate in st.tabs(tab_labels):
        with pert_plate:
            plate = tab_labels[n]
            n += 1
            plotting_functions.plot_banana_plots(control_df[control_df.pert_plate == plate],
                                                 x='logMFI_DMSO',
                                                 y='logMFI_Bortezomib')

    st.header('Banana plots normalized')
    tab_labels = control_df.pert_plate.unique().tolist()
    n = 0
    for pert_plate in st.tabs(tab_labels):
        with pert_plate:
            plate = tab_labels[n]
            n += 1
            plotting_functions.plot_banana_plots(control_df[control_df.pert_plate == plate],
                                                 x='logMFI_norm_DMSO',
                                                 y='logMFI_norm_Bortezomib')

