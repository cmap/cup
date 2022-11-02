def add_pass_rates(df):
    pass_rates = df[df['pass'] == True][['pass', 'prism_replicate', 'culture', 'pert_plate']].groupby(
        ['prism_replicate', 'culture', 'pert_plate']).count().reset_index()
    n_total_lines = df['ccle_name'].unique().shape[0]
    pass_rates['pct_pass'] = ((pass_rates['pass'] / n_total_lines) * 100).astype(int)
    res = df.merge(pass_rates[['prism_replicate', 'pct_pass']], on=['prism_replicate'])
    return res


def add_bc_type(df):
    df.loc[df.pool_id == 'CTLBC', 'bc_type'] = 'control'
    df.loc[df.pool_id != 'CTLBC', 'bc_type'] = 'cell_line'
    return df


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
                'culture']
    group_cols = ['prism_replicate',
                  'ccle_name',
                  'bc_type',
                  'pert_plate',
                  'culture']

    dmso_med = df[df.pert_type == 'ctl_vehicle'][sub_cols].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI': 'ctl_vehicle_med'})

    dmso_mad = df[df.pert_type == 'ctl_vehicle'][sub_cols].groupby(group_cols).mad().reset_index().rename(
        columns={'logMFI': 'ctl_vehicle_mad'})

    bort_med = df[df.pert_type == 'trt_poscon'][sub_cols].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI': 'trt_poscon_med'})

    bort_mad = df[df.pert_type == 'trt_poscon'][sub_cols].groupby(group_cols).mad().reset_index().rename(
        columns={'logMFI': 'trt_poscon_mad'})
    out = dmso_med.merge(dmso_mad, on=merge_cols).merge(bort_med, on=merge_cols).merge(bort_mad, on=merge_cols)

    # normalized data
    sub_cols = ['prism_replicate',
                'ccle_name',
                'bc_type',
                'logMFI_norm',
                'pert_plate',
                'culture']

    dmso_med_norm = df[df.pert_type == 'ctl_vehicle'][sub_cols].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI_norm': 'ctl_vehicle_med_norm'})

    dmso_mad_norm = df[df.pert_type == 'ctl_vehicle'][sub_cols].groupby(group_cols).mad().reset_index().rename(
        columns={'logMFI_norm': 'ctl_vehicle_mad_norm'})

    bort_med_norm = df[df.pert_type == 'trt_poscon'][sub_cols].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI_norm': 'trt_poscon_med_norm'})

    bort_mad_norm = df[df.pert_type == 'trt_poscon'][sub_cols].groupby(group_cols).mad().reset_index().rename(
        columns={'logMFI_norm': 'trt_poscon_mad_norm'})
    out_norm = dmso_med_norm.merge(dmso_mad_norm, on=merge_cols).merge(bort_med_norm, on=merge_cols).merge(
        bort_mad_norm, on=merge_cols)

    res = out.merge(out_norm, on=merge_cols)
    print(res.columns)
    return res

#def add_pass_to_control_df(df, qc=qc_out):
#        cols = ['']