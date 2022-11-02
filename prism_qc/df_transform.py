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
    dmso = df[df.pert_type == 'ctl_vehicle'][['prism_replicate', 'ccle_name', 'bc_type', 'logMFI', 'pert_plate','culture']].groupby(['prism_replicate', 'ccle_name', 'bc_type', 'pert_plate', 'culture']).median().reset_index().rename(columns={'logMFI': 'logMFI_DMSO'})
    bort = df[df.pert_type == 'trt_poscon'][['prism_replicate', 'ccle_name', 'bc_type', 'logMFI', 'pert_plate', 'culture']].groupby(['prism_replicate', 'ccle_name', 'bc_type', 'pert_plate', 'culture']).median().reset_index().rename(columns={'logMFI': 'logMFI_Bortezomib'})
    dmso_norm = df[df.pert_type == 'ctl_vehicle'][['prism_replicate', 'ccle_name', 'bc_type', 'logMFI_norm', 'pert_plate','culture']].groupby(['prism_replicate', 'ccle_name', 'bc_type', 'pert_plate', 'culture']).median().reset_index().rename(columns={'logMFI_norm': 'logMFI_norm_DMSO'})
    bort_norm = df[df.pert_type == 'trt_poscon'][['prism_replicate', 'ccle_name', 'bc_type', 'logMFI_norm', 'pert_plate', 'culture']].groupby(['prism_replicate', 'ccle_name', 'bc_type', 'pert_plate', 'culture']).median().reset_index().rename(columns={'logMFI_norm': 'logMFI_norm_Bortezomib'})
    out = dmso.merge(bort, on=['prism_replicate', 'ccle_name', 'bc_type', 'pert_plate', 'culture'])
    out_norm = dmso_norm.merge(bort_norm, on=['prism_replicate', 'ccle_name', 'bc_type', 'pert_plate', 'culture'])
    res = out.merge(out_norm, on=['prism_replicate', 'ccle_name', 'bc_type', 'pert_plate', 'culture'])
    return res