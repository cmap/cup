import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy import stats
from itertools import combinations
import seaborn as sns
import io
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.subplots as sp
import plotly.io as pio
from matplotlib.colors import ListedColormap
import df_transform
import math
from plotnine import *
from scipy.stats import pearsonr

dr_threshold = -np.log2(0.3)
er_threshold = 0.05


# DYNAMIC RANGE
def plot_dynamic_range(df, metric, build, filename, bucket_name='cup.clue.io'):
    g = px.ecdf(data_frame=df,
                x=metric,
                color='prism_replicate')
    g.add_vline(dr_threshold, line_color='red', line_dash='dash')
    g.update_layout(
        xaxis_title="Dynamic range",
        yaxis_title=""
    )

    # Upload as json to s3
    s3 = boto3.client('s3')
    fig_json = g.to_json()
    s3.put_object(Bucket=bucket_name, Key=f"{build}/{filename}", Body=fig_json.encode('utf-8'))


def plot_dynamic_range_norm_raw(df, build, filename, bucket_name='cup.clue.io'):
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

    # Upload as json to s3
    s3 = boto3.client('s3')
    fig_json = g.to_json()
    s3.put_object(Bucket=bucket_name, Key=f"{build}/{filename}", Body=fig_json.encode('utf-8'))


# PASS RATES
def plot_pass_rates_by_plate(df, build, filename, bucket_name='cup.clue.io'):
    g = px.histogram(data_frame=df,
                     x='prism_replicate',
                     y='pct_pass',
                     histfunc='avg',
                     color='pert_plate',
                     hover_data=['pct_pass'])
    g.update_layout(yaxis_range=[0, 100],
                    yaxis_title='Percent pass',
                    xaxis_title='')

    # Upload as json to s3
    s3 = boto3.client('s3')
    fig_json = g.to_json()
    s3.put_object(Bucket=bucket_name, Key=f"{build}/{filename}", Body=fig_json.encode('utf-8'))


def plot_pass_rates_by_pool(df, build, filename, bucket_name='cup.clue.io'):
    n_plates = len(df.prism_replicate.unique())
    height = math.ceil(n_plates / 3) * 300
    g = px.histogram(data_frame=df,
                     x='pool_id',
                     y='pass',
                     histfunc='count',
                     color='pass',
                     facet_col='prism_replicate',
                     facet_col_wrap=3,
                     width=1200,
                     height=height)

    # Upload as json to s3
    s3 = boto3.client('s3')
    fig_json = g.to_json()
    s3.put_object(Bucket=bucket_name, Key=f"{build}/{filename}", Body=fig_json.encode('utf-8'))


# DISTRIBUTIONS

def plot_distributions_by_plate(df, build, filename, culture, pert_types=['trt_poscon', 'ctl_vehicle'],
                                bucket_name='cup.clue.io', value='logMFI'):
    data = df[(df.pert_type.isin(pert_types)) & (~df.pert_plate.str.contains('BASE')) & (df.culture == culture)]
    controls = ['prism invariant 1', 'prism invariant 10']
    data.loc[(data.ccle_name.isin(controls)) & (data.pert_type == 'ctl_vehicle'), 'pert_type'] = \
        data.loc[(data.ccle_name.isin(controls)) & (data.pert_type == 'ctl_vehicle')]['ccle_name']
    g = sns.FacetGrid(data=data,
                      hue='pert_type',
                      col='replicate',
                      row='pert_plate',
                      legend_out=True,
                      aspect=2)
    g.map(sns.histplot,
          value)

    g.set(xlim=(0, None))

    g.set_titles(row_template='{row_name}', col_template='{col_name}')

    g.add_legend()

    # Save plot as PNG to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Upload as PNG to S3
    s3 = boto3.client('s3')
    full_filename = f"{culture}_{filename}"
    s3.upload_fileobj(buffer, bucket_name, f"{build}/{full_filename}")


# BANANA PLOTS


def plot_banana_plots(df, x, y, filename, build, bucket_name='cup.clue.io'):
    data = df[~df.pert_plate.str.contains('BASE')]
    data.loc[(~data.ccle_name.str.contains('prism')) & (data['pass'] == False), 'bc_type'] = 'cell_line_fail'
    width = len(data['replicate'].unique()) * 400
    height = len(data['pert_plate'].unique()) * 350
    g = px.scatter(data_frame=data,
                   color='bc_type',
                   facet_col='replicate',
                   facet_row='pert_plate',
                   x=x,
                   y=y,
                   hover_data=['ccle_name'],
                   width=width,
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

    # Upload as json to s3
    s3 = boto3.client('s3')
    fig_json = g.to_json()
    s3.put_object(Bucket=bucket_name, Key=f"{build}/{filename}", Body=fig_json.encode('utf-8'))


# LIVER PLOTS

def plot_liver_plots(df, build, filename, bucket_name='cup.clue.io'):
    width = len(df['replicate'].unique()) * 400
    height = len(df['pert_plate'].unique()) * 350
    g = px.scatter(data_frame=df,
                   x='ctl_vehicle_md',
                   y='ctl_vehicle_mad',
                   color='pass',
                   marginal_x='histogram',
                   marginal_y='histogram',
                   hover_data=['ccle_name', 'pool_id', 'prism_replicate'],
                   height=height,
                   width=width,
                   facet_col='replicate',
                   facet_row='pert_plate',
                   color_discrete_map={True: '#66ff66',
                                       False: '#ff0000'})
    g.update_traces(marker=dict(opacity=0.75))
    g.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))
    g.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Upload as json to s3
    s3 = boto3.client('s3')
    fig_json = g.to_json()
    s3.put_object(Bucket=bucket_name, Key=f"{build}/{filename}", Body=fig_json.encode('utf-8'))


# ERROR RATE V SSMD

def plot_dr_error_rate(df, build, filename, bucket_name='cup.clue.io'):
    data = df[~df.pert_plate.str.contains('BASE')]
    width = len(data['replicate'].unique()) * 400
    height = len(data['pert_plate'].unique()) * 350
    g = px.scatter(data_frame=data,
                   facet_col='replicate',
                   facet_row='pert_plate',
                   color='pass',
                   x='dr',
                   y='error_rate',
                   hover_data=['ccle_name', 'pool_id'],
                   height=height,
                   width=width)
    g.add_vline(x=dr_threshold, line_color='#d65f5f', line_dash='dash')
    g.add_hline(y=er_threshold, line_color='#d65f5f', line_dash='dash')
    g.update_traces(marker={'size': 4},
                    opacity=0.7)
    g.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Upload as json to s3
    s3 = boto3.client('s3')
    fig_json = g.to_json()
    s3.put_object(Bucket=bucket_name, Key=f"{build}/{filename}", Body=fig_json.encode('utf-8'))


# REPLICATE CORRELATION

def make_corrplots(df, pert_plate, build, culture='PR500', metric='logMFI_norm', bucket_name='cup.clue.io'):
    data = df[(df.pert_plate == pert_plate) & (df.culture == culture)]
    pivoted_df = data.pivot_table(index=['pert_iname', 'pert_dose', 'pert_type','ccle_name'], 
                                  columns=['replicate'], 
                                  values=metric).reset_index()

    # Flatten the multi-level column index if necessary
    pivoted_df.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in pivoted_df.columns.values]

    # Get a list of all the unique 'replicate' values
    replicates = sorted([col for col in pivoted_df if col.startswith('X')])

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(len(replicates), len(replicates), figsize=(7, 7))
    
    # Set a title for the figure
    fig.suptitle(pert_plate, fontsize=16)

    # Iterate over each subplot and fill in the appropriate plot
    for i, rep_i in enumerate(replicates):
        for j, rep_j in enumerate(replicates):
            ax = axes[i, j]
            # Hide the axis for upper triangle plots
            if i < j:
                ax.axis('off')
                continue
            
            if i == j:  # Diagonal: KDE plot
                sns.kdeplot(data=pivoted_df, x=rep_i, ax=ax)
                ax.set_xlabel('')  # Remove x-label
                ax.set_ylabel('')  # Remove y-label
            else:  # Lower triangle: Scatter plot
                sns.scatterplot(data=pivoted_df, x=rep_i, y=rep_j, ax=ax, s=3)
                # Calculate and annotate Pearson correlation
                clean_df = pivoted_df[[rep_i, rep_j]].dropna()
                corr, _ = pearsonr(clean_df[rep_i], clean_df[rep_j])
                ax.annotate(f'ρ = {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', size=10)
            
            # Set x-axis labels only for the bottom row subplots
            if i == len(replicates) - 1:
                ax.set_xlabel(rep_j.split('_')[-1])
                ax.tick_params(labelbottom=True)  # Show x-axis ticks
            else:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)  # Hide x-axis ticks
            
            # Set y-axis labels only for the first column subplots
            if j == 0:
                ax.set_ylabel(rep_i.split('_')[-1])
                ax.tick_params(labelleft=True)  # Show y-axis ticks
            else:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)  # Hide y-axis ticks


    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Save plot as PNG to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Upload as PNG to S3
    s3 = boto3.client('s3')
    filename = f"{pert_plate}:{culture}_corrplot.png"
    s3.upload_fileobj(buffer, bucket_name, f"{build}/{filename}")


def plot_plate_heatmaps(df, metric, build, culture, vmax=4, vmin=16, by_type=True):
    metric = metric
    df['row'] = df['pert_well'].str[0]
    df['col'] = df['pert_well'].str[1:3]
    data = df[~df.pert_plate.str.contains('BASE') & (df.culture == culture)][
        ['pert_plate', 'replicate', 'row', 'col', metric]]
    data_agg = data.groupby(['pert_plate', 'replicate', 'row', 'col']).median().reset_index()
    combinations = data_agg[['pert_plate', 'replicate']].drop_duplicates()

    # Find unique pert_plate and replicate values
    unique_pert_plates = data_agg['pert_plate'].unique()
    unique_replicates = data_agg['replicate'].unique()

    # Calculate the dimensions of the grid
    n_rows = len(unique_pert_plates)
    n_cols = len(unique_replicates)

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    axes = axes.reshape(n_rows, n_cols)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    # Create a heatmap for each combination of 'pert_plate' and 'replicate'
    for idx, (index, combo) in enumerate(combinations.iterrows()):
        plate = combo['pert_plate']
        replicate = combo['replicate']

        # Find the row and column indices for the current combination
        row_idx = list(unique_pert_plates).index(plate)
        col_idx = list(unique_replicates).index(replicate)

        ax = axes[row_idx, col_idx]

        # Filter the data for the current combination
        heatmap_data = data_agg[(data_agg['pert_plate'] == plate) & (data_agg['replicate'] == replicate)]

        # Pivot the data for the heatmap
        heatmap_data = heatmap_data.pivot(index='row', columns='col', values=metric)

        # Plot the heatmap
        if metric == 'count':
            sns.heatmap(heatmap_data, cmap="Reds_r", ax=ax, vmin=0, vmax=30)
        else:
            sns.heatmap(heatmap_data, cmap="Reds_r", ax=ax, vmin=4, vmax=16)
        ax.set_title(f"{plate} | {replicate}")
        ax.set_xlabel('')
        ax.set_ylabel('')

        if by_type:
            # Generate annots by pert_type if needed
            annots_agg = df.groupby(['pert_plate', 'replicate', 'row', 'col'])['pert_type'].first().reset_index()
            annots_agg['pert_type_annot'] = ''
            annots_agg.loc[annots_agg.pert_type == 'trt_poscon', 'pert_type_annot'] = 'p'
            annots_agg.loc[annots_agg.pert_type == 'ctl_vehicle', 'pert_type_annot'] = 'v'

            # Filter the data for the current combination in the second DataFrame
            annotations_data = annots_agg[(annots_agg['pert_plate'] == plate) & (annots_agg['replicate'] == replicate)]

            # Pivot the data for the annotations
            annotations_data = annotations_data.pivot(index='row', columns='col', values='pert_type_annot').dropna()

            # Annotate the heatmap
            for text_row_idx, row in enumerate(annotations_data.index):
                for text_col_idx, col in enumerate(annotations_data.columns):
                    ax.text(text_col_idx + 0.5, text_row_idx + 0.5, annotations_data.loc[row, col],
                            ha='center', va='center', fontsize=8, color='black')

        plt.tight_layout()
        # Save the plot to a BytesIO object
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)  # Rewind the file pointer to the beginning

        object_key = f"{build}/{metric}_{culture}_heatmaps.png"  # The desired S3 object key (file name)

        s3 = boto3.client('s3')
        s3.upload_fileobj(img_data, 'cup.clue.io', object_key)


def make_pert_type_heatmaps(df, build, vmax, vmin, metric='logMFI'):
    for culture in df.culture.unique():

        # Filter and sort dataframe
        data = df[(df.culture == culture) & (
            df.pert_type.isin(['trt_poscon', 'ctl_vehicle']))] \
            [[metric, 'prism_replicate', 'ccle_name', 'pool_id', 'profile_id', 'pert_type']].sort_values(
            ['pert_type', 'pool_id']).dropna(subset=[metric])
        data['ccle_pool'] = data.ccle_name + ' ' + data.pool_id
        # Create pivot table
        pivot_table = data.pivot_table(
            values=metric,
            index=['pool_id'],
            columns=['pert_type', 'profile_id'],
            aggfunc='median')

        # Create a colormap for pool_id
        unique_pool_ids = pivot_table.index.unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_pool_ids)))  # use any other colormap if you wish
        color_dict = dict(zip(unique_pool_ids, range(len(unique_pool_ids))))

        # Map pool_ids to integer values
        color_column = pd.DataFrame([color_dict[pool_id] for pool_id in pivot_table.index],
                                    index=pivot_table.index,
                                    columns=['color'])

        # Create a colormap from unique integers to colors
        colormap = ListedColormap(colors)

        # Create the subplots
        fig, (ax1, ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [0.5, 20]}, figsize=(12, 6))

        # Plot the color bar as a heatmap with pool_id as yticklabels
        sns.heatmap(color_column, ax=ax1, cmap=colormap, cbar=False, yticklabels=True, xticklabels=[], vmax=vmax,
                    vmin=vmin)

        # Rotate yticklabels for better visibility
        ax1.yaxis.tick_left()  # Move ticks to the right side of color bar
        for label in ax1.get_yticklabels():
            label.set_rotation(0)

        # Plot the main heatmap
        sns.heatmap(pivot_table, ax=ax2, xticklabels=[], yticklabels=False, vmax=vmax, vmin=vmin)

        # Remove the space between the plots
        plt.subplots_adjust(wspace=0.01)

        # Remove appropriate labels
        ax2.set_ylabel('')
        ax1.set_ylabel('')
        ax2.set_xlabel('')

        ax2.annotate("ctl_vehicle", xy=(0.2, 1.01), annotation_clip=False, xycoords='axes fraction',
                     textcoords='offset points', xytext=(5, 5))
        ax2.annotate("trt_poscon", xy=(0.69, 1.01), annotation_clip=False, xycoords='axes fraction',
                     textcoords='offset points', xytext=(5, 5))

        # Save the plot to a BytesIO object
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)  # Rewind the file pointer to the beginning

        if metric == 'logMFI':
            object_key = f"{build}/{culture}_pert_type_heatmap.png"  # The desired S3 object key (file name)
        else:
            object_key = f"{build}/{metric}_{culture}_pert_type_heatmap.png"  # The desired S3 object key (file name)

        s3 = boto3.client('s3')
        s3.upload_fileobj(img_data, 'cup.clue.io', object_key)


def make_build_count_heatmaps(df, build, metric='count'):
    for culture in df.culture.unique():
        # Filter and sort dataframe
        data = df[(df.culture == culture) & (~df.prism_replicate.str.contains('BASE'))].sort_values(['prism_replicate', 'pert_well'])
        # Create pivot table
        pivot_table = data.pivot_table(
            values=metric,
            index=['prism_replicate'],
            columns=['pert_well'],
            aggfunc='median')

        # Create a colormap for pool_id
        unique_replicates = pivot_table.index.unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_replicates)))  # use any other colormap if you wish
        color_dict = dict(zip(unique_replicates, range(len(unique_replicates))))

        # Map pool_ids to integer values
        color_column = pd.DataFrame([color_dict[prism_replicate] for prism_replicate in pivot_table.index],
                                    index=pivot_table.index,
                                    columns=['color'])

        # Create a colormap from unique integers to colors
        colormap = ListedColormap(colors)

        # Create the subplots
        fig, (ax1, ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [0.5, 20]}, figsize=(12, 6))

        # Plot the color bar as a heatmap with pool_id as yticklabels
        sns.heatmap(color_column, ax=ax1, cmap=colormap, cbar=False, yticklabels=True, xticklabels=[])

        # Rotate yticklabels for better visibility
        ax1.yaxis.tick_left()  # Move ticks to the right side of color bar
        for label in ax1.get_yticklabels():
            label.set_rotation(0)

        # Plot the main heatmap
        sns.heatmap(pivot_table, ax=ax2, xticklabels=[], yticklabels=False, vmin=0, vmax=30)

        # Remove the space between the plots
        plt.subplots_adjust(wspace=0.01)

        # Remove appropriate labels
        ax2.set_ylabel('')
        ax1.set_ylabel('')
        ax2.set_xlabel('')

        ax2.set_xlabel('pert_well', size=12)

        # Pad plot to preserve xtick labels
        plt.tight_layout(pad=1.5)

        # Save the plot to a BytesIO object
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)  # Rewind the file pointer to the beginning

        object_key = f"{build}/{culture}_count_heatmap.png"  # The desired S3 object key (file name)

        s3 = boto3.client('s3')
        s3.upload_fileobj(img_data, 'cup.clue.io', object_key)


def generate_cbc_quantile_plot(df, build, culture):
    # Filter and get unique values
    unique_values = df.prism_replicate[(~df.prism_replicate.str.contains('BASE')) & (df.culture == culture)].unique()

    # Determine rows and columns
    total_plots = len(unique_values)
    rows = np.ceil(total_plots / 3).astype(int)  # round up to get enough rows

    # Create subplots
    fig, axes = plt.subplots(rows, 3, figsize=(10, rows * 3.33))  # Adjust size as needed
    axes = axes.ravel()  # Flatten the axes array

    # Create a plot for each unique value
    for i in range(total_plots):
        subset = df[(df.prism_replicate == unique_values[i]) & (df.culture == culture) & (
            df.pert_type.isin(['ctl_vehicle', 'ctl_untrt']))]

        # calculate the median for each control barcode
        cbc = subset[subset.ccle_name.str.contains('prism invariant')]
        cbc_med = cbc[['ccle_name', 'logMFI']].groupby(['ccle_name']).median()

        # get the logMFI values for cell_line
        cl = subset[~subset.ccle_name.str.contains('prism invariant')]['logMFI']

        # apply function to determine quantiles
        quantiles_bc = cbc_med['logMFI'].apply(lambda x: df_transform.quantile_of_closest_score(x, cl))

        # make dataframe
        data = pd.DataFrame(quantiles_bc).reset_index().rename(columns={'ccle_name': 'bead',
                                                                        'logMFI': 'quantile'})

        # sort data
        data['sort'] = data['bead'].str.split(' ').str[2].astype('int')
        data.sort_values('sort', inplace=True)

        # make plots
        sns.lineplot(data=data, ax=axes[i], x='sort', y='quantile')  # Replace with your function
        axes[i].set_title(f'{unique_values[i]}')  # Optional title for each subplot
        axes[i].plot([0, 10], [0, 1], color='grey', linestyle='--')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # If total plots < axes, remove the extras
    if total_plots % 3 != 0:  # We have some empty subplots
        for i in range(total_plots, len(axes)):  # Loop from last plot index to end of axes
            fig.delaxes(axes[i])  # Remove the extra subplots

    # Set tight layout
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Rewind the file pointer to the beginning

    object_key = f"{build}/{culture}_cb_quantiles.png"  # The desired S3 object key (file name)

    s3 = boto3.client('s3')
    s3.upload_fileobj(img_data, 'cup.clue.io', object_key)


def make_build_mfi_heatmaps(df, build, vmax, vmin, metric='logMFI'):
    for culture in df.culture.unique():
        # Filter and sort dataframe
        data = df[(df.culture == culture) & (~df.prism_replicate.str.contains('BASE'))].sort_values(['prism_replicate', 'pert_well'])
        # Create pivot table
        pivot_table = data.pivot_table(
            values=metric,
            index=['prism_replicate'],
            columns=['pert_well'],
            aggfunc='median')

        # Create a colormap for pool_id
        unique_replicates = pivot_table.index.unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_replicates)))  # use any other colormap if you wish
        color_dict = dict(zip(unique_replicates, range(len(unique_replicates))))

        # Map pool_ids to integer values
        color_column = pd.DataFrame([color_dict[prism_replicate] for prism_replicate in pivot_table.index],
                                    index=pivot_table.index,
                                    columns=['color'])

        # Create a colormap from unique integers to colors
        colormap = ListedColormap(colors)

        # Create the subplots
        fig, (ax1, ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [0.5, 20]}, figsize=(12, 6))

        # Plot the color bar as a heatmap with pool_id as yticklabels
        sns.heatmap(color_column, ax=ax1, cmap=colormap, cbar=False, yticklabels=True, xticklabels=[])

        # Rotate yticklabels for better visibility
        ax1.yaxis.tick_left()  # Move ticks to the right side of color bar
        for label in ax1.get_yticklabels():
            label.set_rotation(0)

        # Plot the main heatmap
        sns.heatmap(pivot_table, ax=ax2, xticklabels=[], yticklabels=False, vmin=vmin, vmax=vmax)

        # Remove the space between the plots
        plt.subplots_adjust(wspace=0.01)

        # Remove appropriate labels
        ax2.set_ylabel('')
        ax1.set_ylabel('')
        ax2.set_xlabel('')

        ax2.set_xlabel('pert_well', size=12)

        # Pad plot to preserve xtick labels
        plt.tight_layout(pad=1.5)

        # Save the plot to a BytesIO object
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)  # Rewind the file pointer to the beginning

        object_key = f"{build}/{culture}_mfi_heatmap.png"  # The desired S3 object key (file name)

        s3 = boto3.client('s3')
        s3.upload_fileobj(img_data, 'cup.clue.io', object_key)

def make_control_violin_plot(df, build, culture):
    # Subset data
    data = df[(df.pert_type == 'ctl_vehicle') & (df.ccle_name.str.contains('prism')) & (df.culture==culture)]
    data['analyte_num'] = data['ccle_name'].str.split(' ').str[2].astype('int')
    data['analyte_num'] = pd.Categorical(data['analyte_num'])
    data.sort_values('analyte_num')

    # Determine the number of unique values for facets
    n_cols = len(data['replicate'].unique())
    n_rows = len(data['pert_plate'].unique())

    # Set figure dimensions based on the number of facets
    fig_width = 3 * n_cols  # Adjust multiplier as needed for width
    fig_height = 3 * n_rows  # Adjust multiplier as needed for height

    # Create plot
    g = (
        ggplot(data, aes(x='analyte_num', y='logMFI')) +
        geom_violin() +
        xlab('') +
        ylab('logMFI') +
        facet_grid('pert_plate ~ replicate') +
        theme(figure_size=(fig_width, fig_height))
    )

    # Save plot to a BytesIO object as PNG
    img_data = io.BytesIO()
    g.save(img_data, format='png', width=fig_width, height=fig_height, dpi=100)
    img_data.seek(0)

    # Upload to S3
    s3 = boto3.client('s3')
    object_key = f"{build}/{culture}_ctl_violin.png"
    s3.upload_fileobj(img_data, 'cup.clue.io', object_key)

# Control barcode rank heatmaps
def make_ctlbc_rank_heatmaps(df, build, culture):
    # Subset data and add row/col
    plot_data = df[~df.prism_replicate.str.contains('BASE')]
    plot_data['row'] = plot_data['pert_well'].str[0]
    plot_data['col'] = plot_data['pert_well'].str[1:3]
    plot_data['row'] = plot_data['row'].astype('category')
    plot_data['col'] = plot_data['col'].astype('category')
    plot_data['row'] = pd.Categorical(plot_data['row'], categories=reversed(plot_data['row'].cat.categories), ordered=True)
    plot_data['analyte_num'] = plot_data['ccle_name'].str.split(' ').str[2].astype('int')
    plot_data['plate'] = plot_data['pert_plate'] + '_' + plot_data['replicate']

    # calculate figure size
    n_plates = plot_data.prism_replicate.unique().shape[0]
    fig_width = 12
    fig_height = n_plates 

    plot_data.sort_values('analyte_num', inplace=True)
    p = (
        ggplot(plot_data, aes(x='col', y='row', fill='rank')) +
        geom_tile() +
        facet_grid('plate ~ analyte_num') +
        theme(
            figure_size=(fig_width,fig_height),
            strip_text_x=element_text(size=10),
            strip_text_y=element_text(size=7),
            axis_text=element_blank(),
            axis_ticks_major=element_blank()
        ) +
        xlab('') +
        ylab('')
    )

    # Save plot to a BytesIO object as PNG
    img_data = io.BytesIO()
    p.save(img_data, format='png', width=fig_width, height=fig_height, dpi=200)
    img_data.seek(0)

    # Upload to S3
    s3 = boto3.client('s3')
    object_key = f"{build}/{culture}_ctlbc_rank_heatmap.png"
    s3.upload_fileobj(img_data, 'cup.clue.io', object_key)

def make_ctlbc_rank_violin(df, build, culture, corrs):
    # Subset data and add row/col
    plot_data = df[(~df.prism_replicate.str.contains('BASE'))&(df.culture==culture)]
    plot_data['analyte_num'] = plot_data['ccle_name'].str.split(' ').str[2].astype('int')
    plot_data['analyte_num'] = pd.Categorical(plot_data['analyte_num'])
    plot_data['plate'] = plot_data['pert_plate'] + '_' + plot_data['replicate']

    # Add pairwise correlation values
    plot_data['correlation'] = plot_data['prism_replicate'].map(corrs)
    plot_data['correlation'] = plot_data['correlation'].map("ρ={:.2f}".format)

    # calculate figure size
    n_cols = plot_data.replicate.unique().shape[0]
    n_rows = plot_data.pert_plate.unique().shape[0]
    fig_width = n_cols * 4
    fig_height = n_rows * 4 

    plot_data.sort_values('analyte_num', inplace=True)
    p = (
        ggplot(plot_data, aes(x='analyte_num', y='rank')) +
        geom_violin() +
        geom_text(aes(label='correlation'), data=plot_data.drop_duplicates('prism_replicate'), x=2, y=9.7, size=15) +
        facet_grid('pert_plate ~ replicate') +
        scale_y_continuous(breaks=range(1, 11)) +
        xlab('Analyte') +
        ylab('Rank') +
        theme(figure_size=(fig_width, fig_height))
    )
    
    # Save plot to a BytesIO object as PNG
    img_data = io.BytesIO()
    p.save(img_data, format='png', dpi=150)
    img_data.seek(0)

    # Upload to S3
    s3 = boto3.client('s3')
    object_key = f"{build}/{culture}_ctlbc_rank_violin.png"
    s3.upload_fileobj(img_data, 'cup.clue.io', object_key)