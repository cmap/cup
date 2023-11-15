# setup
import boto3
import pandas as pd
import numpy as np

dr_threshold = -np.log2(0.3)
er_threshold = 0.05


def add_pass_rates(df):
    pass_rates = df[df['pass'] == True][['pass', 'prism_replicate', 'culture', 'pert_plate']].groupby(
        ['prism_replicate', 'culture', 'pert_plate']).count().reset_index()
    n_instances = df[['pass', 'prism_replicate', 'culture', 'pert_plate']].groupby(
        ['prism_replicate', 'culture', 'pert_plate']).count().reset_index().rename(
        columns={'pass': 'n_instances'})
    pass_rates = pass_rates.merge(n_instances, on=['prism_replicate', 'culture', 'pert_plate'])
    pass_rates['pct_pass'] = ((pass_rates['pass'] / pass_rates['n_instances']) * 100).astype(int)
    res = df.merge(pass_rates[['prism_replicate', 'pct_pass']], on=['prism_replicate'])
    return res


def add_bc_type(df):
    df.loc[df.pool_id == 'CTLBC', 'bc_type'] = 'control'
    df.loc[df.pool_id != 'CTLBC', 'bc_type'] = 'cell_line'
    return df


def add_replicate(df):
    df['replicate'] = df.prism_replicate.str.split('_').str[3].str.split('.').str[0]
    return df


def compute_mad(series):
    return abs(series - series.mean()).mean()


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

    data = df[sub_cols + ['pert_type']]

    dmso_med = data[data.pert_type == 'ctl_vehicle'][sub_cols].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI': 'ctl_vehicle_med'})

    dmso_mad = data[data.pert_type == 'ctl_vehicle'][sub_cols].groupby(group_cols).agg(
        lambda x: compute_mad(x)).reset_index().rename(
        columns={'logMFI': 'ctl_vehicle_mad'})

    bort_med = data[data.pert_type == 'trt_poscon'][sub_cols].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI': 'trt_poscon_med'})

    bort_mad = data[data.pert_type == 'trt_poscon'][sub_cols].groupby(group_cols).agg(
        lambda x: compute_mad(x)).reset_index().rename(
        columns={'logMFI': 'trt_poscon_mad'})

    out = dmso_med.merge(dmso_mad, on=merge_cols).merge(bort_med, on=merge_cols).merge(bort_mad, on=merge_cols)

    # normalized data
    sub_cols = ['prism_replicate',
                'ccle_name',
                'bc_type',
                'logMFI_norm',
                'pert_plate',
                'culture']

    data = df[sub_cols + ['pert_type']]

    dmso_med_norm = data[data.pert_type == 'ctl_vehicle'][sub_cols].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI_norm': 'ctl_vehicle_med_norm'})

    dmso_mad_norm = data[data.pert_type == 'ctl_vehicle'][sub_cols].groupby(group_cols).agg(
        lambda x: compute_mad(x)).reset_index().rename(
        columns={'logMFI_norm': 'ctl_vehicle_mad_norm'})

    bort_med_norm = data[data.pert_type == 'trt_poscon'][sub_cols].groupby(group_cols).median().reset_index().rename(
        columns={'logMFI_norm': 'trt_poscon_med_norm'})

    bort_mad_norm = data[data.pert_type == 'trt_poscon'][sub_cols].groupby(group_cols).agg(
        lambda x: compute_mad(x)).reset_index().rename(
        columns={'logMFI_norm': 'trt_poscon_mad_norm'})

    out_norm = dmso_med_norm.merge(dmso_mad_norm, on=merge_cols).merge(bort_med_norm, on=merge_cols).merge(
        bort_mad_norm, on=merge_cols)

    res = out.merge(out_norm, on=merge_cols)
    return res


def generate_pass_fail_tbl(mfi, qc, prefix, bucket='cup.clue.io'):
    mfi_drop_cols = ['logMFI',
                     'logMFI_norm',
                     'pert_type',
                     'replicate']
    qc_drop_cols = ['ssmd',
                    'nnmd',
                    'pool_id',
                    'pass']

    df = mfi.drop(columns=mfi_drop_cols).merge(qc.drop(columns=qc_drop_cols),
                                               on=['prism_replicate', 'ccle_name', 'pert_plate', 'culture'])

    res = pd.DataFrame(
        columns=['prism_replicate', 'pert_plate', 'culture', 'Pass',
                 'Fail both', 'Fail error rate', 'Fail dynamic range'])
    for plate in df.prism_replicate.unique():
        culture = df[df.prism_replicate == plate]['culture'].unique()[0]
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
                     'culture': culture,
                     'Pass': pass_both,
                     'Fail both': fail_both,
                     'Fail error rate': fail_er,
                     'Fail dynamic range': fail_dr}
        tmp_df = pd.DataFrame(data=to_append, index=[0])
        res = pd.concat([res, tmp_df])

        # Convert DataFrame to CSV data
        csv_buffer = res.to_csv(index=False).encode()

        # Upload CSV data to S3 bucket
        s3 = boto3.client('s3')
        s3.put_object(Bucket=bucket, Key=f"{prefix}/pass_fail_table.csv", Body=csv_buffer)

    return res


def append_raw_dr(mfi, qc):
    cols = ['prism_replicate','ccle_name','pert_type','logMFI']
    bort = \
        mfi[mfi.pert_type == 'trt_poscon'][cols].groupby(
            ['prism_replicate', 'ccle_name', 'pert_type']).median(numeric_only=True).reset_index()[
            ['prism_replicate', 'ccle_name', 'logMFI']]
    dmso = \
        mfi[mfi.pert_type == 'ctl_vehicle'][cols].groupby(
            ['prism_replicate', 'ccle_name', 'pert_type']).median(numeric_only=True).reset_index()[
            ['prism_replicate', 'ccle_name', 'logMFI']]
    dr = dmso.merge(bort, on=['prism_replicate', 'ccle_name'], suffixes=('_dmso', '_bort'))
    dr['dr_raw'] = dr['logMFI_dmso'] - dr['logMFI_bort']
    dr = dr[['prism_replicate', 'ccle_name', 'dr_raw']]
    res = qc.merge(dr, on=['prism_replicate', 'ccle_name'], how='left')
    return res


def construct_count_df(count, mfi):
    count['culture'] = count['cid'].str.split('_').str[1]
    count.loc[count.culture == 'PR300P', 'culture'] = 'PR300'
    count['rid'] = count['rid'] + '_' + count['culture']
    res = count.merge(mfi[['profile_id', 'rid', 'prism_replicate', 'pool_id', 'pert_well', 'pert_plate', 'replicate','pert_type', 'ccle_name']],
                      left_on=['rid', 'cid'],
                      right_on=['rid', 'profile_id'], how='left').dropna()
    res.rename(columns={'value': 'count'}, inplace=True)
    return res


# Define a function that will return the quantile of a given value in the cell_line_logMFI distribution
def quantile_of_closest_score(value, scores):
    closest_value_index = (np.abs(scores - value)).argmin()
    return pd.Series(scores).rank(pct=True).iloc[closest_value_index]


def get_instances_removed(inst: pd.DataFrame, mfi: pd.DataFrame, cell: pd.DataFrame) -> pd.DataFrame:
    # Assign values to 'culture' column based on conditions
    conditions_inst = [inst.profile_id.str.contains('PR300'), inst.profile_id.str.contains('PR500')]
    choices_inst = ['PR300', 'PR500']
    inst['culture'] = np.select(conditions_inst, choices_inst, default=None)

    conditions_cell = [cell.davepool_id.str.contains('CS14'), cell.davepool_id.str.contains('CS5.')]
    choices_cell = ['PR300', 'PR500']
    cell['culture'] = np.select(conditions_cell, choices_cell, default=None)

    # Cross join cell and inst on 'culture'
    expected_instances = cell.merge(inst[['pert_plate', 'pert_well', 'replicate', 'culture', 'prism_replicate']],
                                   on='culture', how='outer')

    # Merge and filter to get instances_removed
    instances_removed = mfi.merge(expected_instances,
                                 on=['ccle_name', 'culture', 'pert_plate', 'pert_well', 'replicate', 'pool_id',
                                     'prism_replicate'],
                                 how='right')

    instances_removed = instances_removed[instances_removed.pert_id.isna()]

    # Merge with inst to get additional columns
    instances_removed = instances_removed.merge(
        inst[['prism_replicate', 'pert_well', 'pert_id', 'pert_iname', 'pert_dose']],
        on=['prism_replicate', 'pert_well'],
        how='left')

    # Rename columns
    instances_removed.rename(columns={'pert_iname_y': 'pert_iname',
                                     'pert_dose_y': 'pert_dose',
                                     'pert_id_y': 'pert_id'}, inplace=True)

    return instances_removed


def profiles_removed(df):
    replicates_by_compound = df[~df.ccle_name.str.contains('invariant')].groupby(['culture','pert_plate','ccle_name','pert_iname','pert_dose']).size().reset_index(name='n_profiles')
    res = replicates_by_compound[replicates_by_compound.n_profiles < 2].drop(columns=['n_profiles'])
    return res

# Define a function to calculate analyte ranks within each group
def calculate_ranks(group):
    group['rank'] = group['logMFI'].rank(method='first')
    return group