import plotly.express as px
import plotly.graph_objects as go
import streamlit as st



def plot_dynamic_range(df):
    g = px.ecdf(data_frame=df,
                x='dr',
                color='prism_replicate')
    g.add_vline(1.8, line_color='red', line_dash='dash')
    g.update_layout(
        xaxis_title="Dynamic range",
        yaxis_title=""
    )
    st.plotly_chart(g)



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



def plot_distributions(df, value='logMFI'):
    g = px.histogram(data_frame=df,
                     color='pert_type',
                     x=value,
                     barmode='overlay',
                     histnorm='percent')
    g.update_layout(yaxis_title='')
    st.plotly_chart(g)



def plot_distributions_by_plate(df, height, value='logMFI'):
    g = px.histogram(data_frame=df,
                     color='pert_type',
                     x=value,
                     barmode='overlay',
                     histnorm='percent',
                     facet_col='prism_replicate',
                     facet_col_wrap=3,
                     height=height)
    g.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    g.update_layout(yaxis_title='')
    st.plotly_chart(g, use_container_width=True)



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
                           line=dict(color='black',
                                     dash='dash',
                                     width=1),
                           marker=dict(size=0.1),
                           showlegend=False),
                row='all', col='all', exclude_empty_subplots=True)
    g.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(g, use_container_width=False)



def plot_med_mad(df):
    g = px.scatter(data_frame=df,
                   x='ctl_vehicle_md',
                   y='ctl_vehicle_mad',
                   color='pass',
                   marginal_x='histogram',
                   marginal_y='histogram',
                   hover_data=['ccle_name', 'pool_id', 'prism_replicate'],
                   height=700)
    g.update_traces(marker=dict(opacity=0.5))
    st.plotly_chart(g)



def plot_ssmd_error_rate(df, height):
    g = px.scatter(data_frame=df,
                   facet_col='prism_replicate',
                   facet_col_wrap=3,
                   color='pass',
                   x='dr',
                   y='error_rate',
                   hover_data=['ccle_name', 'pool_id'],
                   height=height)
    g.add_vline(x=1.8, line_color='red', line_dash='dash')
    g.add_hline(y=0.05, line_color='red', line_dash='dash')
    g.update_traces(marker={'size': 4},
                    opacity=0.7)
    g.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(g, use_container_width=True)
