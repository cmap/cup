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
    g = px.histogram(data_frame=df,
                     x='pool_id',
                     y='pass',
                     histfunc='count',
                     color='pass')

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


def plot_corrplot(df, mfi, filename, build, bucket_name='cup.clue.io'):
    df = df.sort_values('pert_plate')
    pert_plates = df['pert_plate'].unique()
    num_pert_plates = len(pert_plates)

    cols = list(mfi[~mfi.pert_plate.str.contains('BASE')].replicate.unique())
    num_cols = len(cols)

    max_cols_per_row = 3
    num_rows = (num_pert_plates + max_cols_per_row - 1) // max_cols_per_row

    fig, axes = plt.subplots(nrows=num_cols * num_rows, ncols=num_cols * max_cols_per_row,
                             figsize=(10 * max_cols_per_row, 10 * num_rows), sharex='col', sharey='row')

    for idx, pert_plate in enumerate(pert_plates):
        # Calculate row and col index for the current pert_plate
        row_idx = idx // max_cols_per_row
        col_idx = idx % max_cols_per_row

        # Filter dataframe by pert_plate
        df_filtered = df[df['pert_plate'] == pert_plate]

        for i in range(num_cols):
            for j in range(num_cols):
                ax = axes[row_idx * num_cols + i, col_idx * num_cols + j]

                # Calculate the correlation coefficient for the x and y variables
                corr_coef = np.corrcoef(df_filtered[cols[j]], df_filtered[cols[i]])[0, 1]

                # Create scatter plot
                ax.scatter(df_filtered[cols[j]], df_filtered[cols[i]], alpha=0.5)

                # Add diagonal line
                min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
                max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

                # Set axis labels
                if i == num_cols - 1:
                    ax.set_xlabel(cols[j])
                if j == 0:
                    ax.set_ylabel(cols[i])

                # Add the correlation coefficient to the subplot title
                if i != j:
                    ax.set_title(f'{corr_coef:.2f}', x=0.2, y=0.75, fontweight='bold', size=25)

        # Label each grid with the pert_plate it contains
        axes[row_idx * num_cols, col_idx * num_cols].set_title(
            f'{pert_plate}\n' + axes[row_idx * num_cols, col_idx * num_cols].get_title(), x=1.7, fontweight='bold',
            size=20)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=5, hspace=5)

    # Set tight layout
    plt.tight_layout()

    # Save plot as PNG to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Upload as PNG to S3
    s3 = boto3.client('s3')
    s3.upload_fileobj(buffer, bucket_name, f"{build}/{filename}")


def plot_dmso_performance(df, build, filename, bucket_name='cup.clue.io'):
    # Create a FacetGrid with multiple plots
    g = sns.FacetGrid(df[df.pert_type.isin(['ctl_vehicle'])],
                      row='pert_plate',
                      col='replicate',
                      height=4, aspect=1.5,
                      legend_out=True)

    # Map the boxplot to the FacetGrid
    g.map(sns.boxplot, 'pert_well', 'logMFI', 'bc_type', linewidth=1.5, hue='bc_type', fliersize=0)

    # Add row and column titles to the FacetGrid
    for ax in g.axes.flat:
        ax.set_xlabel('')
    g.set_titles(col_template='{col_name}', row_template='{row_name}')

    # Set the labels for the x and y axes
    g.set_axis_labels('', 'logMFI')

    # Rotate the x-axis labels for better readability
    g.set_xticklabels(rotation=90)

    # Move the legend outside of the plot and create a single legend
    g.add_legend(title='bc_type', loc='upper right', bbox_to_anchor=(1, 0.5))

    # Save plot as PNG to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    buffer.seek(0)

    # Upload as PNG to S3
    s3 = boto3.client('s3')
    s3.upload_fileobj(buffer, bucket_name, f"{build}/{filename}")


def plot_plate_heatmaps(df, metric, build, culture, by_type=True):
    metric = metric
    df['row'] = df['pert_well'].str[0]
    df['col'] = df['pert_well'].str[1:3]
    data = df[~df.pert_plate.str.contains('BASE')&(df.culture == culture)][['pert_plate', 'replicate', 'row', 'col', metric]]
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
            sns.heatmap(heatmap_data, cmap="Reds_r", ax=ax, vmin=7, vmax=16)
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


def plot_historical_mfi(df, metric, filename, bucket_name='cup.clue.io'):
    unique_builds = df['build'].unique()
    unique_pert_types = df['pert_type'].unique()

    # Create a subplot with two columns
    fig = sp.make_subplots(cols=2, subplot_titles=unique_pert_types)

    for col, pert_type in enumerate(unique_pert_types, start=1):
        hist_data = []
        group_labels = []

        for build in unique_builds:
            df_filtered = df[(df['pert_type'] == pert_type) & (df['build'] == build)]
            hist_data.append(df_filtered[metric])
            group_labels.append(build)  # Use the build as the group label

        # Create the KDE plot for the current pert_type
        kde_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)

        # Add traces from kde_fig to the main figure
        for trace, build in zip(kde_fig.data, unique_builds):
            trace.showlegend = col == 1  # Only show legend for the first column
            trace.legendgroup = build  # Set legendgroup to the build
            fig.add_trace(trace, row=1, col=col)

    # Update the layout
    fig.update_layout(showlegend=True,
                      height=500,
                      width=1200)

    # Upload as json to s3
    s3 = boto3.client('s3')
    fig_json = fig.to_json()
    s3.put_object(Bucket=bucket_name, Key=f"historical/{filename}", Body=fig_json.encode('utf-8'))


def make_pert_type_heatmaps(df, build, metric='logMFI'):
    for culture in df.culture.unique():
        # Filter and sort dataframe
        data = df[(df.culture == culture)&(~df.ccle_name.str.contains('prism'))&(df.pert_type.isin(['trt_poscon','ctl_vehicle']))]\
        [[metric, 'prism_replicate', 'ccle_name', 'pool_id', 'profile_id', 'pert_type']].sort_values(['pert_type', 'pool_id']).dropna(subset=[metric])
        data['ccle_pool'] = data.ccle_name + ' ' + data.pool_id
        # Create pivot table
        pivot_table = data.pivot_table(
            values=metric,
            index=['pool_id'],
            columns=['pert_type','profile_id'],
            aggfunc='median')

        # Create a colormap for pool_id
        unique_pool_ids = pivot_table.index.unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_pool_ids))) # use any other colormap if you wish
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
        sns.heatmap(color_column, ax=ax1, cmap=colormap, cbar=False, yticklabels=True, xticklabels=[])

        # Rotate yticklabels for better visibility
        ax1.yaxis.tick_left()  # Move ticks to the right side of color bar
        for label in ax1.get_yticklabels():
            label.set_rotation(0)

        # Plot the main heatmap
        sns.heatmap(pivot_table, ax=ax2, xticklabels=[], yticklabels=False)

        # Remove the space between the plots
        plt.subplots_adjust(wspace=0.01)

        # Remove appropriate labels
        ax2.set_ylabel('')
        ax1.set_ylabel('')
        ax2.set_xlabel('')

        ax2.annotate("ctl_vehicle", xy=(0.2,1.01), annotation_clip=False, xycoords='axes fraction', textcoords='offset points', xytext=(5,5))
        ax2.annotate("trt_poscon", xy=(0.69,1.01), annotation_clip=False, xycoords='axes fraction', textcoords='offset points', xytext=(5,5))

        # Save the plot to a BytesIO object
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)  # Rewind the file pointer to the beginning

        object_key = f"{build}/{culture}_pert_type_heatmap.png"  # The desired S3 object key (file name)

        s3 = boto3.client('s3')
        s3.upload_fileobj(img_data, 'cup.clue.io', object_key)


def make_full_count_heatmaps(df, build, metric='count'):
    for culture in df.culture.unique():
        # Filter and sort dataframe
        data = df[(df.culture == culture)&(~df.plate.str.contains('BASE'))].sort_values(['plate', 'pert_well'])
        # Create pivot table
        pivot_table = data.pivot_table(
            values=metric,
            index=['plate'],
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


def make_pert_type_heatmaps_by_plate(df, build, culture, metric='logMFI'):
        data = df[(df.culture == culture) & (~df.ccle_name.str.contains('prism')) & (
            df.pert_type.isin(['trt_poscon', 'ctl_vehicle']))] \
            [[metric, 'prism_replicate', 'ccle_name', 'pool_id', 'profile_id', 'pert_type']].sort_values(
            ['pert_type', 'pool_id']).dropna(subset=[metric])
        data['ccle_pool'] = data.ccle_name + ' ' + data.pool_id

        plates = data.prism_replicate.unique()

        for plate in plates:
            plate_data = data[data.prism_replicate == plate]
            # Create pivot table
            pivot_table = plate_data.pivot_table(
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
            sns.heatmap(color_column, ax=ax1, cmap=colormap, cbar=False, yticklabels=True, xticklabels=[])

            # Rotate yticklabels for better visibility
            ax1.yaxis.tick_left()  # Move ticks to the right side of color bar
            for label in ax1.get_yticklabels():
                label.set_rotation(0)

            # Plot the main heatmap
            sns.heatmap(pivot_table, ax=ax2, xticklabels=[], yticklabels=False)

            # Remove the space between the plots
            plt.subplots_adjust(wspace=0.01)

            # Set the title
            plt.title(plate, y=1.05)

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

            object_key = f"{build}/{plate}_{culture}_pert_type_heatmap.png"  # The desired S3 object key (file name)

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
        subset = df[(df.prism_replicate == unique_values[i]) & (df.culture == culture) & (df.pert_type.isin(['ctl_vehicle', 'ctl_untrt']))]

        # calculate the median for each control barcode
        cbc = subset[subset.ccle_name.str.contains('prism invariant')]
        cbc_med = cbc[['ccle_name','logMFI']].groupby(['ccle_name']).median()

        # get the logMFI values for cell_line
        cl = subset[~subset.ccle_name.str.contains('prism invariant')]['logMFI']

        # apply function to determine quantiles
        quantiles_bc = cbc_med['logMFI'].apply(lambda x: df_transform.quantile_of_closest_score(x, cl))

        # make dataframe
        data = pd.DataFrame(quantiles_bc).reset_index().rename(columns={'ccle_name':'bead',
                                                                        'logMFI':'quantile'})

        # sort data
        data['sort'] = data['bead'].str.split(' ').str[2].astype('int')
        data.sort_values('sort', inplace=True)

        # make plots
        sns.lineplot(data=data, ax=axes[i], x = 'sort', y='quantile')  # Replace with your function
        axes[i].set_title(f'{unique_values[i]}')  # Optional title for each subplot
        axes[i].plot([0,10],[0,1], color='grey', linestyle='--')
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