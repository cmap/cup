import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy import stats
from itertools import combinations

dr_threshold = -np.log2(0.3)
er_threshold = 0.05

# DYNAMIC RANGE


def plot_dynamic_range(df, metric):
    g = px.ecdf(data_frame=df,
                x=metric,
                color='prism_replicate')
    g.add_vline(dr_threshold, line_color='red', line_dash='dash')
    g.update_layout(
        xaxis_title="Dynamic range",
        yaxis_title=""
    )
    st.plotly_chart(g)


def plot_dynamic_range_norm_raw(df):
    g = px.scatter(data_frame=df,
                   x='dr_raw',
                   y='dr',
                   facet_row='pert_plate',
                   facet_col='replicate',
                   width=1000,
                   hover_data=['ccle_name'])
    x_line = (0, 6)
    g.update_traces(marker={'size': 4},
                    opacity=0.7)
    g.add_trace(go.Scatter(x=x_line,
                           y=x_line,
                           line=dict(color='#d65f5f',
                                     dash='dash',
                                     width=1),
                           marker=dict(size=0.1),
                           showlegend=False),
                row='all', col='all', exclude_empty_subplots=True)
    g.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(g, use_container_width=False)

# SSMD


def plot_ssmd(df):
    data = df.sort_values('prism_replicate')
    g = px.histogram(data, x='ssmd',
                     color='prism_replicate',
                     nbins=int(data.pert_plate.unique().shape[0] * 25),
                     hover_data=['ccle_name'],
                     facet_col='pert_plate',
                     facet_col_wrap=2,
                     barmode='overlay')
    g.add_vline(x=-2, line_color='red', line_dash='dash', line_width=3)
    st.plotly_chart(g, use_container_width=True)

# PASS RATES


def plot_pass_rates_by_plate(df):
    g = px.histogram(data_frame=df,
                     x='prism_replicate',
                     y='pct_pass',
                     histfunc='avg',
                     color='pert_plate',
                     hover_data=['pct_pass'])
    g.update_layout(yaxis_range=[0, 100],
                    yaxis_title='Percent pass',
                    xaxis_title='')
    st.plotly_chart(g, use_container_width=False)


def plot_pass_rates_by_pool(df):
    g = px.histogram(data_frame=df,
                     x='pool_id',
                     y='pass',
                     histfunc='count',
                     color='pass')
    st.plotly_chart(g, use_container_width=True)


# DISTRIBUTIONS


def plot_distributions(df, value='logMFI'):
    g = px.histogram(data_frame=df,
                     color='pert_type',
                     x=value,
                     barmode='overlay',
                     histnorm='percent')
    g.update_layout(yaxis_title='')
    st.plotly_chart(g)


def plot_distributions_by_plate(df, height, value='logMFI'):
    data = df
    controls = ['prism invariant 1', 'prism invariant 10']
    data.loc[(data.ccle_name.isin(controls)) & (data.pert_type == 'ctl_vehicle'), 'pert_type'] = \
        data.loc[(data.ccle_name.isin(controls)) & (data.pert_type == 'ctl_vehicle')]['ccle_name']
    g = px.histogram(data_frame=data,
                     color='pert_type',
                     x=value,
                     barmode='overlay',
                     histnorm='percent',
                     facet_col='prism_replicate',
                     facet_col_wrap=3,
                     height=height,
                     nbins=100)
    g.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    g.update_layout(yaxis_title='')
    st.plotly_chart(g, use_container_width=True)


# BANANA PLOTS


def plot_banana_plots(df, x, y, height):
    g = px.scatter(data_frame=df,
                   color='bc_type',
                   facet_col='prism_replicate',
                   facet_col_wrap=3,
                   x=x,
                   y=y,
                   hover_data=['ccle_name'],
                   width=1000,
                   height=height)
    g.update_yaxes(matches=None, showticklabels=True)
    g.update_traces(marker={'size': 4},
                    opacity=0.7)
    x_line = (6, 15)
    g.add_trace(go.Scatter(x=x_line,
                           y=x_line,
                           line=dict(color='#d65f5f',
                                     dash='dash',
                                     width=1),
                           marker=dict(size=0.1),
                           showlegend=False),
                row='all', col='all', exclude_empty_subplots=True)
    g.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(g, use_container_width=False)


# LIVER PLOTS

def plot_liver_plots(df):
    g = px.scatter(data_frame=df,
                   x='ctl_vehicle_md',
                   y='ctl_vehicle_mad',
                   color='pass',
                   marginal_x='histogram',
                   marginal_y='histogram',
                   hover_data=['ccle_name', 'pool_id', 'prism_replicate'],
                   height=600,
                   width=800,
                   color_discrete_map={True: '#66ff66',
                                       False: '#ff0000'})
    g.update_traces(marker=dict(opacity=0.75))
    g.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))
    st.plotly_chart(g)


# ERROR RATE V SSMD

def plot_ssmd_error_rate(df, height):
    g = px.scatter(data_frame=df,
                   facet_col='prism_replicate',
                   facet_col_wrap=3,
                   color='pass',
                   x='dr',
                   y='error_rate',
                   hover_data=['ccle_name', 'pool_id'],
                   height=height,
                   width=1000)
    g.add_vline(x=dr_threshold, line_color='#d65f5f', line_dash='dash')
    g.add_hline(y=er_threshold, line_color='#d65f5f', line_dash='dash')
    g.update_traces(marker={'size': 4},
                    opacity=0.7)
    g.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(g, use_container_width=False)


# REPLICATE CORRELATION

def reshape_df_for_corr(df, metric='logMFI_norm'):
    df['perturbation'] = df['pert_iname'] + ' ' + df['pert_dose'].astype('str')

    cols = [metric,
            'replicate',
            'perturbation']

    res = df[cols].pivot_table(index='perturbation', columns='replicate', values=metric).reset_index()
    return res


def make_dimensions_for_corrplot(df, sub_mfi):
    dimensions = []
    for plate in sub_mfi.replicate.unique():
        out = dict(label=plate,
                   values=df[plate])
        dimensions.append(out)
    print(dimensions)
    print(type(dimensions))
    return dimensions


def make_dimensions_for_corrtable(df, sub_mfi):
    dimensions = {}
    for plate in sub_mfi.replicate.unique():
        out = {plate: df[plate]}
        dimensions.update(out)
    return dimensions


def plot_corrplot(df, dim_list):
    g = go.Figure(go.Splom(
        dimensions=dim_list,
        showupperhalf=False,
        text=df['perturbation']))
    g.update_layout(height=750,
                    width=750,
                    margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(g)


def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2


def mk_corr_table(df, sub_mfi):
    dimensions = {}
    for plate in sub_mfi.replicate.unique():
        out = {plate: df[plate]}
        dimensions.update(out)

    names = []
    for replicate in dimensions:
        names.append(replicate)
    unique = [",".join(map(str, comb)) for comb in combinations(names, 2)]

    corr_table = pd.DataFrame(columns=['Replicates', 'r2'])
    for comb in unique:
        rep_a = comb.split(',')[0]
        rep_b = comb.split(',')[1]
        values_a = df[rep_a]
        values_b = df[rep_b]
        r2 = round(rsquared(values_a, values_b),2)
        tmp = pd.DataFrame([[comb, r2]], columns=['Replicates', 'r2'])
        corr_table = pd.concat([corr_table, tmp])

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(corr_table.columns),
                    fill_color='grey',
                    align='center'),
        cells=dict(values=[corr_table.Replicates, corr_table.r2],
                   fill_color='black',
                   align='center'))
    ])
    fig.update_layout(width=350,
                      height=115,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=False)
