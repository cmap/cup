# setup
import pandas as pd
import streamlit as st

dr_threshold = 1.8
er_threshold = 0.05


@st.cache(suppress_st_warning=True)
def add_pass_rates(df):
    pass_rates = df[df['pass'] == True][['pass', 'prism_replicate', 'culture', 'pert_plate']].groupby(
        ['prism_replicate', 'culture', 'pert_plate']).count().reset_index()
    n_total_lines = df['ccle_name'].unique().shape[0]
    pass_rates['pct_pass'] = ((pass_rates['pass'] / n_total_lines) * 100).astype(int)
    res = df.merge(pass_rates[['prism_replicate', 'pct_pass']], on=['prism_replicate'])
    return res


@st.cache(suppress_st_warning=True)
def add_bc_type(df):
    df.loc[df.pool_id == 'CTLBC', 'bc_type'] = 'control'
    df.loc[df.pool_id != 'CTLBC', 'bc_type'] = 'cell_line'
    return df


@st.cache(suppress_st_warning=True)
def pivot_dmso_bort(df):
    # raw data
    merge_cols = ['prism_replicate',
                  'ccle_name',
                  'bc_type',
                  'pert_plate',
                  'culture']
    sub_cols = ['prism_replicate',
                'ccle_name',
                'bc_type',
                'logMFI',
                'pert_plate',
                'culture',
                'pert_type']
    group_cols = ['prism_replicate',
                  'ccle_name',
                  'bc_type',
                  'pert_plate',
                  'culture']

    data = df[sub_cols]

    dmso_med = data[data.pert_type == 'ctl_vehicle'].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI': 'ctl_vehicle_med'})

    dmso_mad = data[data.pert_type == 'ctl_vehicle'].groupby(group_cols).mad().reset_index().rename(
        columns={'logMFI': 'ctl_vehicle_mad'})

    bort_med = data[data.pert_type == 'trt_poscon'].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI': 'trt_poscon_med'})

    bort_mad = data[data.pert_type == 'trt_poscon'].groupby(group_cols).mad().reset_index().rename(
        columns={'logMFI': 'trt_poscon_mad'})
    out = dmso_med.merge(dmso_mad, on=merge_cols).merge(bort_med, on=merge_cols).merge(bort_mad, on=merge_cols)

    # normalized data
    sub_cols = ['prism_replicate',
                'ccle_name',
                'bc_type',
                'logMFI_norm',
                'pert_plate',
                'culture',
                'pert_type']

    data = df[sub_cols]

    dmso_med_norm = data[data.pert_type == 'ctl_vehicle'].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI_norm': 'ctl_vehicle_med_norm'})

    dmso_mad_norm = data[data.pert_type == 'ctl_vehicle'].groupby(group_cols).mad().reset_index().rename(
        columns={'logMFI_norm': 'ctl_vehicle_mad_norm'})

    bort_med_norm = data[data.pert_type == 'trt_poscon'].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI_norm': 'trt_poscon_med_norm'})

    bort_mad_norm = data[data.pert_type == 'trt_poscon'].groupby(group_cols).mad().reset_index().rename(
        columns={'logMFI_norm': 'trt_poscon_mad_norm'})
    out_norm = dmso_med_norm.merge(dmso_mad_norm, on=merge_cols).merge(bort_med_norm, on=merge_cols).merge(
        bort_mad_norm, on=merge_cols)

    res = out.merge(out_norm, on=merge_cols)
    return res


@st.cache(suppress_st_warning=True)
def generate_pass_fail_tbl(mfi, qc):
    df = mfi.merge(qc, on=['prism_replicate', 'ccle_name', 'pert_plate'])
    res = pd.DataFrame(
        columns=['prism_replicate', 'pert_plate', 'Pass', 'Fail both', 'Fail error rate', 'Fail dynamic range'])
    for plate in df.prism_replicate.unique():
        pert_plate = df[df.prism_replicate == plate]['pert_plate'].unique()[0]
        n_samples = df[df.prism_replicate == plate].shape[0]
        fail_dr = int((df.loc[(df.prism_replicate == plate) & (df.dr < dr_threshold) & (
                    df.error_rate <= er_threshold)].shape[0] / n_samples) * 100)
        fail_both = int((df.loc[(df.prism_replicate == plate) & (df.dr < dr_threshold) & (
                    df.error_rate > er_threshold)].shape[0] / n_samples) * 100)
        pass_both = int((df.loc[(df.prism_replicate == plate) & (df.dr >= dr_threshold) & (
                    df.error_rate <= er_threshold)].shape[0] / n_samples) * 100)
        fail_er = int((df.loc[(df.prism_replicate == plate) & (df.dr >= dr_threshold) & (
                    df.error_rate > er_threshold)].shape[0] / n_samples) * 100)
        to_append = {'prism_replicate': plate,
                     'pert_plate': pert_plate,
                     'Pass': pass_both,
                     'Fail both': fail_both,
                     'Fail error rate': fail_er,
                     'Fail dynamic range': fail_dr}
        tmp_df = pd.DataFrame(data=to_append, index=[0])
        res = pd.concat([res, tmp_df])
    return res
