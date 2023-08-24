import logging
import os
from pathlib import Path
import pandas as pd
import s3fs
import streamlit as st
import descriptions
import df_transform
import plotting_functions
from metadata import prism_metadata
import boto3
import plotly.io as pio
import io
from PIL import Image
import botocore
import json
import read_build
import pymysql

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
API_URL = 'https://api.clue.io/api/'
DEV_URL = 'https://dev-api.clue.io/api/'
API_KEY = os.environ['API_KEY']
BUILDS_URL = API_URL + 'data_build_types/prism-builds'
SCANNER_URL = DEV_URL + 'lims_plate'

# get list of builds
builds = prism_metadata.get_data_from_db(
    endpoint_url=BUILDS_URL,
    user_key=API_KEY,
    fields=['name']
)

fs = s3fs.S3FileSystem(anon=False)

builds_list = []
for i in builds:
    # insert filter IF here
    builds_list.append(list(i.values())[0])


# USER INPUTS

def update_params():
    st.experimental_set_query_params(option=st.session_state.qp)


query_params = st.experimental_get_query_params()

# Set selectbox value based on query param, or provide a default
ix = 0
if query_params:
    try:
        ix = builds_list.index(query_params['option'][0])
    except ValueError:
        pass

build = st.selectbox(
    "Param", builds_list, index=ix, key="qp", on_change=update_params
)

# Set query param based on selection
st.experimental_set_query_params(option=build)
view_report = st.button('View Report')
generate_report = st.button('Generate Report')
corrplot = st.checkbox('Create correlation plots', value=True)
st.text('Note: You may only generate correlation plots if each pert_plate has > 1 replicate.')


def get_file(files, file_string):
    for file in files:
        if file_string in file:
            return file


def upload_df_to_s3(df, filename, prefix, bucket_name='cup.clue.io'):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Body=csv_buffer.getvalue().encode('utf-8'), Bucket=bucket_name, Key=f"{prefix}/{filename}")
    print(f"File '{filename}' uploaded to bucket '{bucket_name}'")


def load_df_from_s3(filename, prefix, bucket_name='cup.clue.io'):
    s3 = boto3.client('s3')
    try:
        # Check if the object exists by retreiving metadata
        s3.head_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")

        # If exustsm proceed
        response = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
        csv_bytes = response['Body'].read()
        csv_buffer = io.StringIO(csv_bytes.decode())
        df = pd.read_csv(csv_buffer)
        return df

    except botocore.exceptions.ClientError as e:
        st.text('There is at least one file missing, please regenerate report and try again.')

def load_json_table_from_s3(filename, prefix, bucket_name='cup.clue.io'):
    s3 = boto3.client('s3')
    try:
        # Check if the object exists by retreiving metadata
        s3.head_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")

        # If no exceptions, proceed
        response = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
        json_bytes = response['Body'].read()
        json_str = json_bytes.decode()
        df = pd.read_json(io.StringIO(json_str), orient='records')
        return df

    except botocore.exceptions.ClientError as e:
        st.text('There is at least one file missing, please regenerate report and try again.')


def load_plot_from_s3(filename, prefix, bucket_name='cup.clue.io'):
    s3 = boto3.client('s3')
    try:
        # Check if the object exists by retreiving metadata
        s3.head_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")

        # If no exception, proceed
        response = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
        fig_json = response['Body'].read().decode('utf-8')
        fig = pio.from_json(fig_json)
        st.plotly_chart(fig)

    except botocore.exceptions.ClientError as e:
        st.text('There is at least one file missing, please regenerate report and try again.')


def load_image_from_s3(filename, prefix, bucket_name='cup.clue.io'):
    s3 = boto3.client('s3')
    try:
        # Check if the object exists by retrieving metadata
        s3.head_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")

        # If the previous line didn't raise an exception, proceed with fetching the actual object
        response = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
        content = response['Body'].read()

        # Load image data from buffer
        img_buffer = io.BytesIO(content)
        img = Image.open(img_buffer)

        # Display image in Streamlit
        st.image(img)

    except botocore.exceptions.ClientError as e:
        st.text('There is at least one file missing, please regenerate report and try again.')


def check_file_exists(bucket_name, file_name):
    s3 = boto3.client('s3')

    try:
        s3.head_object(Bucket=bucket_name, Key=file_name)
        return True
    except Exception as e:
        return False


def write_json_to_s3(bucket, filename, data, prefix):
    s3 = boto3.client('s3')
    # Convert your data to a JSON string
    json_data = json.dumps(data)
    # Convert your JSON string to bytes
    json_bytes = json_data.encode()
    # Write the JSON data to an S3 object
    s3.put_object(Body=json_bytes, Bucket=bucket, Key=f"{prefix}/{filename}")


def write_json_table_to_s3(bucket, filename, data, prefix):
    s3 = boto3.client('s3')

    # Convert your JSON string to bytes
    json_bytes = data.encode()

    # Write the JSON data to an S3 object with specified content type
    s3.put_object(Body=json_bytes, Bucket=bucket, Key=f"{prefix}/{filename}", ContentType='application/json')


def read_json_from_s3(bucket_name, filename, prefix):
    s3 = boto3.client('s3')
    # Get the object from S3
    s3_object = s3.get_object(Bucket=bucket_name, Key=f"{prefix}/{filename}")
    # Get the body of the object (the data content)
    s3_object_body = s3_object['Body'].read()
    # Convert bytes to string
    string_data = s3_object_body.decode('utf-8')
    # Convert string data to dictionary
    dict_data = json.loads(string_data)
    return dict_data


# Inputs
if view_report and build:

    # Compare expected plots to files on s3
    s3 = boto3.client('s3')
    bucket = 'cup.clue.io'
    prefix = build

    expected_plots = [f"{prefix}/{filename}" for filename in
                      ['build_metadata.json', 'plate_metadata.json']]

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        objects = response['Contents']
        existing_plots = [obj['Key'] for obj in objects]
        print(f"Found {len(existing_plots)} objects with prefix '{prefix}' in bucket '{bucket}'")
    else:
        print(f"No objects with prefix '{prefix}' found in bucket '{bucket}'")
        existing_plots = []

    if set(expected_plots).issubset(set(existing_plots)):
        print(f"All of the necessary plots already exist, generating output.")

        # Get list of cultures
        build_metadata = read_json_from_s3(bucket_name=bucket,
                                           filename='build_metadata.json',
                                           prefix=build)
        cultures = build_metadata['culture']
        plates = build_metadata['plates']

        # Get plate metadata
        plate_metadata = read_json_from_s3(bucket_name=bucket,
                                           filename='plate_metadata.json',
                                           prefix=build)
        scanner_table = pd.DataFrame(json.loads(plate_metadata))
        if 'scanner_id' in scanner_table.columns:
            scanner_table['scanner_id'] = scanner_table['scanner_id'].astype('Int64')
            scanner_table['median_count'] = scanner_table['median_count'].astype('Int64')
            scanner_table['iqr_count'] = scanner_table['iqr_count'].astype('Int64')

        # Show report
        with st.spinner('Loading report...'):
            st.title('PRISM QC report')
            st.title(build)

            # Show summary heatmaps
            with st.expander('logMFI'):
                st.header('LogMFI')
                st.subheader('Build')
                st.markdown(descriptions.build_heatmap_ctl_mfi)
                tab_labels = cultures
                tabs = st.tabs(tab_labels)
                for label, tab in zip(tab_labels, tabs):
                    with tab:
                        filename = f"{label}_pert_type_heatmap.png"
                        load_image_from_s3(filename=filename, prefix=build)

                st.subheader('Controls')
                st.markdown(descriptions.plate_heatmap_ctl_mfi)
                tab_labels = cultures
                tabs = st.tabs(tab_labels)
                for label, tab in zip(tab_labels, tabs):
                    with tab:
                        selected_plates = [plate for plate in plates if label in plate and 'BASE' not in plate]
                        for plate in selected_plates:
                            filename = f"{plate}_{label}_pert_type_heatmap.png"
                            load_image_from_s3(filename=filename, prefix=build)

                # Show plate heatmaps
                st.subheader('Plate')
                st.markdown(descriptions.plate_heatmap_mfi)
                raw, norm = st.tabs(['Raw', 'Normalized'])
                with raw:
                    tab_labels = cultures
                    tabs = st.tabs(tab_labels)
                    for label, tab in zip(tab_labels, tabs):
                        with tab:
                            filename = f"logMFI_{label}_heatmaps.png"
                            load_image_from_s3(filename=filename, prefix=build)
                with norm:
                    tab_labels = cultures
                    tabs = st.tabs(tab_labels)
                    for label, tab in zip(tab_labels, tabs):
                        with tab:
                            filename = f"logMFI_norm_{label}_heatmaps.png"
                            load_image_from_s3(filename=filename, prefix=build)

            with st.expander('Bead count'):
                st.header('Bead Count')
                if 'det_plate' in scanner_table:
                    st.dataframe(scanner_table.drop(columns=['det_plate']))
                st.subheader('Build count')
                st.markdown(descriptions.build_heatmap_count)
                tab_labels = cultures
                tabs = st.tabs(tab_labels)
                for label, tab in zip(tab_labels, tabs):
                    with tab:
                        filename = f"{label}_count_heatmap.png"
                        load_image_from_s3(filename=filename, prefix=build)

                st.subheader('Plate count')
                st.markdown(descriptions.plate_heatmap_count)
                tab_labels = cultures
                tabs = st.tabs(tab_labels)
                for label, tab in zip(tab_labels, tabs):
                    with tab:
                        filename = f"count_{label}_heatmaps.png"
                        load_image_from_s3(filename=filename, prefix=build)

            with st.expander('Control barcodes'):
                # control barcode quantiles
                st.header('Control barcode performance')
                st.markdown(descriptions.ctl_quantiles)
                tab_labels = cultures
                tabs = st.tabs(tab_labels)
                for label, tab in zip(tab_labels, tabs):
                    with tab:
                        filename = f"{label}_cb_quantiles.png"
                        load_image_from_s3(filename=filename, prefix=build)

            with st.expander('Data removed'):
                st.header('Instances removed')

                st.markdown(descriptions.instances_removed)

                # Establish columns
                by_plate, by_compound, by_well = st.columns((1, 1.5, 1))

                # Populate columns
                by_plate.subheader('By plate')
                tbl_by_plate = load_json_table_from_s3(filename='instances_removed_by_plate_table.json', prefix=build)
                by_plate.dataframe(tbl_by_plate)

                by_well.subheader('By well')
                tbl = load_json_table_from_s3(filename='instances_removed_by_well.json',
                                              prefix=build)
                by_well.dataframe(tbl)

                by_compound.subheader('By compound')
                tbl = load_json_table_from_s3(filename='instances_removed_by_compound.json',
                                              prefix=build)
                by_compound.dataframe(tbl)

                st.subheader('Plots')
                load_plot_from_s3(filename='plt_rm_instances_by_line.json', prefix=build)
                load_plot_from_s3(filename='plt_rm_instances_by_cp.json', prefix=build)

                st.header('Profiles removed')
                st.markdown(descriptions.profiles_removed)

                # Establish columns
                by_compound, by_cell = st.columns((1,1.5))

                # Populate columns
                by_compound.subheader('By compound')
                tbl = load_json_table_from_s3(filename='profiles_removed_by_compound.json',
                                              prefix=build)
                by_compound.dataframe(tbl)

                by_cell.subheader('By cell line')
                tbl = load_json_table_from_s3(filename='profiles_removed_by_line.json',
                                              prefix=build)
                by_cell.dataframe(tbl)


            with st.expander('Cell line pass/fail'):
                # Plot pass rates
                st.header('Pass rates')
                st.markdown(descriptions.dr_and_er)
                by_plate, by_pool = st.tabs(['By plate', 'By pool'])
                with by_plate:
                    st.markdown(descriptions.pass_by_plate)
                    load_plot_from_s3(filename='pass_by_plate.json', prefix=build)
                with by_pool:
                    st.markdown(descriptions.pass_by_pool)
                    load_plot_from_s3(filename='pass_by_pool.json', prefix=build)

                # Show pass/fail table
                st.subheader('Pass/fail table')
                st.markdown(descriptions.pass_table)
                pass_fail = load_df_from_s3('pass_fail_table.csv', prefix=build)
                st.table(
                    pass_fail.reset_index(drop=True).style.bar(subset=['Pass'], color='#006600', vmin=0, vmax=100).bar(
                        subset=['Fail both', 'Fail error rate', 'Fail dynamic range'], color='#d65f5f', vmin=0,
                        vmax=100))

                # Plot dynamic range
                st.header('Dynamic range')
                st.markdown(descriptions.dr_ecdf)
                dr_norm, dr_raw = st.tabs(['Normalized', 'Raw'])
                with dr_norm:
                    load_plot_from_s3(filename='dr_norm.json', prefix=build)
                with dr_raw:
                    load_plot_from_s3(filename='dr_raw.json', prefix=build)

            with st.expander('Threshold plots'):
                # Liver plots
                st.header('Liver plots')
                st.markdown(descriptions.liver_plots)
                load_plot_from_s3(filename='liverplot.json', prefix=build)

                # Banana plots
                st.header('Banana plots')
                st.markdown(descriptions.banana_plots)
                banana_normalized, banana_raw = st.tabs(['Normalized', 'Raw'])
                with banana_normalized:
                    load_plot_from_s3('banana_norm.json', prefix=build)
                with banana_raw:
                    load_plot_from_s3('banana_raw.json', prefix=build)

                # Dynamic range versus error rate
                st.header('Error rate and dynamic range')
                st.markdown(descriptions.dr_vs_er)
                load_plot_from_s3('dr_er.json', prefix=build)

            with st.expander('Distributions'):
                # Plot plate distributions
                st.header('Plate distributions')
                st.markdown(descriptions.plate_dists)
                raw, norm = st.tabs(['Raw', 'Normalized'])
                with raw:
                    tab_labels = cultures
                    tabs = st.tabs(tab_labels)
                    for label, tab in zip(tab_labels, tabs):
                        with tab:
                            filename = f"{label}_plate_dist_raw.png"
                            load_image_from_s3(filename=filename, prefix=build)
                with norm:
                    tab_labels = cultures
                    tabs = st.tabs(tab_labels)
                    for label, tab in zip(tab_labels, tabs):
                        with tab:
                            filename = f"{label}_plate_dist_norm.png"
                            load_image_from_s3(filename=filename, prefix=build)
            # Plot correlations
            if check_file_exists(file_name=f"{build}/corrplot_raw.png", bucket_name='cup.clue.io'):
                with st.expander('Correlations'):
                    st.header('Correlations')
                    st.markdown(descriptions.corr)
                    norm, raw = st.tabs(['Normalized', 'Raw'])
                    with raw:
                        load_image_from_s3(filename='corrplot_raw.png', prefix=build)
                    with norm:
                        load_image_from_s3(filename='corrplot_norm.png', prefix=build)

    else:
        st.text('Some content is missing from this report, try generating it again.\
        \nIf this problem persists after regeneration, bother John!')

elif generate_report and build:
    s3 = boto3.client('s3')
    bucket = 'cup.clue.io'
    prefix = build

    print(f"The necessary plots DO NOT exist, generating output:")
    build_path = "s3://macchiato.clue.io/builds/" + build + "/build/"
    if fs.exists(build_path):
        file_list = fs.ls(build_path)
        qc_file = 's3://' + get_file(file_list, 'QC_TABLE')
        print('QC file found: ' + qc_file)
        mfi_file = 's3://' + get_file(file_list, 'LEVEL3')
        print('MFI file found: ' + mfi_file)
        lfc_file = 's3://' + get_file(file_list, 'LEVEL5')
        print('LFC file found: ' + lfc_file)

        with st.spinner('Generating report and uploading results'):

            # Read build and create metric dfs
            df_build = read_build.read_build_from_s3(build)
            qc = df_build.qc
            mfi = df_build.mfi
            lfc = df_build.lfc
            count = df_build.count
            inst = df_build.inst
            cell = df_build.cell

            # Get df of instances that are removed
            instances_removed = df_transform.get_instances_removed(inst=inst, mfi=mfi, cell=cell)
            profiles_removed = df_transform.profiles_removed(df=mfi)

            # Annotate count df
            cnt = df_transform.construct_count_df(count, mfi)

            # Save list of cultures to metadata json
            cultures = list(mfi.culture.unique())
            plates = list(mfi.prism_replicate.unique())
            json_data = {'culture': cultures,
                         'plates': plates}

            filename = 'build_metadata.json'
            write_json_to_s3(data=json_data,
                             bucket=bucket,
                             prefix=build,
                             filename=filename)

            # Create plate metadata from lims
            det_plates = list(qc.prism_replicate.unique())
            plate_meta = pd.DataFrame()
            for plate in plates:
                response = prism_metadata.get_data_from_db(
                    endpoint_url=SCANNER_URL,
                    user_key=API_KEY,
                    where={"det_plate": plate},
                    fields=['det_plate', 'scanner_id']
                )
                response = pd.DataFrame(response)
                plate_meta = pd.concat([plate_meta, response])

            # Group by 'prism_replicate' and calculate various statistics
            agg_funcs = {
                'count': ['median', 'std', 'var', lambda x: x.quantile(0.75) - x.quantile(0.25)]
            }
            cnt_meta = cnt.groupby('prism_replicate').agg(agg_funcs).reset_index()
            cnt_meta.columns = ['prism_replicate', 'median_count', 'stdev_count', 'var_count', 'iqr_count']
            if 'det_plate' in plate_meta.columns:
                plate_meta = plate_meta.merge(cnt_meta, left_on='det_plate', right_on='prism_replicate', how='right')
            json_data = plate_meta.to_json()
            write_json_to_s3(data=json_data,
                             bucket=bucket,
                             prefix=build,
                             filename='plate_metadata.json')

            # add count meta to count df
            if 'det_plate' in plate_meta.columns:
                cnt = cnt.merge(plate_meta, on=['prism_replicate'], how='left')
                cnt['plate'] = cnt['prism_replicate'] + "[" + cnt['scanner_id'].astype('str') + "]"
            else:
                cnt['plate'] = cnt['prism_replicate']

            # Transform mfi and qc tables
            mfi_out = mfi.pipe(df_transform.add_bc_type)
            qc_out = qc.pipe(df_transform.add_pass_rates) \
                .pipe(df_transform.add_replicate)
            qc_out = df_transform.append_raw_dr(mfi, qc_out)

            # Pivot table for poscon/negcon comparison and upload to s3
            control_df = mfi_out.pipe(df_transform.pivot_dmso_bort)
            control_df = control_df.merge(qc_out,
                                          on=['prism_replicate',
                                              'ccle_name',
                                              'pert_plate'],
                                          how='left')
            control_df['replicate'] = control_df['prism_replicate'].str.split('_').str[3]

            print(f"Generating replicate correlation dataframes.....")
            corr_df_norm = mfi[~mfi.pert_plate.str.contains('BASE')].pivot_table(columns=['replicate'],
                                                                                 values='logMFI_norm',
                                                                                 index=['pert_iname', 'pert_dose',
                                                                                        'pert_plate']).dropna().reset_index()

            corr_df_raw = mfi[~mfi.pert_plate.str.contains('BASE')].pivot_table(columns=['replicate'],
                                                                                values='logMFI',
                                                                                index=['pert_iname', 'pert_dose',
                                                                                       'pert_plate']).dropna().reset_index()

            # Make tables of excluded instances
            print(f"Generating tables of excluded instances....")
            # By plate
            instances_removed_by_plate = instances_removed.groupby(['culture', 'prism_replicate']).size().reset_index(
                name='instances_removed')
            json_data = instances_removed_by_plate.to_json(orient='records')
            write_json_table_to_s3(bucket=bucket,
                                   filename='instances_removed_by_plate_table.json',
                                   data=json_data,
                                   prefix=prefix)
            # By well
            instances_removed_by_well = instances_removed.groupby(['culture','pert_well']).size().reset_index(name='instances_removed')
            json_data = instances_removed_by_well.to_json(orient='records')
            write_json_table_to_s3(bucket=bucket,
                                   filename='instances_removed_by_well.json',
                                   data=json_data,
                                   prefix=prefix)

            # By compound
            instances_removed_by_compound = instances_removed.groupby(
                ['culture', 'prism_replicate', 'pert_iname']).size().reset_index(name='instances_removed')
            json_data = instances_removed_by_compound.to_json(orient='records')
            write_json_table_to_s3(bucket=bucket,
                                   filename='instances_removed_by_compound.json',
                                   data=json_data,
                                   prefix=prefix)

            # Compound/doses with <2 replicates
            replicates_by_compound = mfi[~mfi.ccle_name.str.contains('invariant')].groupby(
                ['culture', 'pert_plate', 'ccle_name', 'pert_iname', 'pert_dose']).size().reset_index(name='n_instances')
            collapsed_instances_removed = replicates_by_compound[replicates_by_compound.n_instances < 2].drop(
                columns=['n_instances'])
            json_data = collapsed_instances_removed.to_json(orient='records')
            write_json_table_to_s3(bucket=bucket,
                                   filename='compound_dose_removed.json',
                                   data=json_data,
                                   prefix=prefix)

            # Make tables of excluded profiles
            print(f"Generating tables of excluded profiles...")
            # By compound
            profiles_removed_by_compound = profiles_removed.groupby(['culture','pert_plate','pert_iname']).size().reset_index(name='n_profiles')
            json_data = profiles_removed_by_compound.to_json(orient='records')
            write_json_table_to_s3(bucket=bucket,
                                   filename='profiles_removed_by_compound.json',
                                   data=json_data,
                                   prefix=prefix)

            # By cell line
            profiles_removed_by_line = profiles_removed.groupby(['culture','pert_plate','ccle_name']).size().reset_index(name='n_profiles')
            json_data = profiles_removed_by_line.to_json(orient='records')
            write_json_table_to_s3(bucket=bucket,
                                   filename='profiles_removed_by_line.json',
                                   data=json_data,
                                   prefix=prefix)



            # Generate and save plots
            print("Generating profile removal plots....")
            plotting_functions.plot_instances_removed_by_compound(df=instances_removed,
                                                                 build=build)
            plotting_functions.plot_instances_removed_by_line(df=instances_removed,
                                                             build=build)

            print("Generating control barcode quantile plots....")
            for culture in cultures:
                plotting_functions.generate_cbc_quantile_plot(df=mfi,
                                                              build=build,
                                                              culture=culture)

            print(f"Generating mfi heatmaps by plate/pool.....")
            for culture in cultures:
                plotting_functions.make_pert_type_heatmaps_by_plate(df=mfi,
                                                                    build=build,
                                                                    culture=culture)

            print(f"Generating MFI heatmaps.....")
            plotting_functions.make_pert_type_heatmaps(df=mfi,
                                                       build=build)

            print(f"Generating COUNT heatmaps.....")
            plotting_functions.make_full_count_heatmaps(df=cnt,
                                                        build=build)

            print(f"Generating pass rate plots.....")
            plotting_functions.plot_pass_rates_by_plate(df=qc_out,
                                                        build=build,
                                                        filename='pass_by_plate.json')
            plotting_functions.plot_pass_rates_by_pool(df=qc_out,
                                                       build=build,
                                                       filename='pass_by_pool.json')

            print("Generating dynamic range plots.....")
            plotting_functions.plot_dynamic_range(df=qc_out,
                                                  metric='dr',
                                                  build=build,
                                                  filename='dr_norm.json')
            plotting_functions.plot_dynamic_range(df=qc_out,
                                                  metric='dr_raw',
                                                  build=build,
                                                  filename='dr_raw.json')

            print(f"Generating plate distribution plots.....")
            for culture in cultures:
                plotting_functions.plot_distributions_by_plate(mfi_out,
                                                               value='logMFI_norm',
                                                               build=build,
                                                               filename='plate_dist_norm.png',
                                                               culture=culture)
                plotting_functions.plot_distributions_by_plate(mfi_out,
                                                               value='logMFI',
                                                               build=build,
                                                               filename='plate_dist_raw.png',
                                                               culture=culture)

            print(f"Generating plate heatmaps.....")
            for culture in cultures:
                plotting_functions.plot_plate_heatmaps(mfi_out,
                                                       metric='logMFI',
                                                       build=build,
                                                       culture=culture)
                plotting_functions.plot_plate_heatmaps(mfi_out,
                                                       metric='logMFI_norm',
                                                       build=build,
                                                       culture=culture)
                plotting_functions.plot_plate_heatmaps(cnt,
                                                       metric='count',
                                                       by_type=False,
                                                       build=build,
                                                       culture=culture)
                plotting_functions.plot_plate_heatmaps(cnt,
                                                       metric='count',
                                                       by_type=False,
                                                       build=build,
                                                       culture=culture)

            print(f"Generating liver plots.....")
            try:
                plotting_functions.plot_liver_plots(qc_out,
                                                    build=build,
                                                    filename='liverplot.json')
            except:
                print("Could not generate liver plots.")
            print(f"Generating banana plots.....")
            plotting_functions.plot_banana_plots(control_df,
                                                 build=build,
                                                 x='ctl_vehicle_med_norm',
                                                 y='trt_poscon_med_norm',
                                                 filename='banana_norm.json')

            plotting_functions.plot_banana_plots(control_df,
                                                 build=build,
                                                 x='ctl_vehicle_med',
                                                 y='trt_poscon_med',
                                                 filename='banana_raw.json')

            print(f"Generating error rate plot.....")
            try:
                plotting_functions.plot_dr_error_rate(qc_out,
                                                      build=build,
                                                      filename='dr_er.json')
            except:
                print(f"Error generating error rate plot.")

            print(f"Generating pass/fail table.....")
            df_transform.generate_pass_fail_tbl(mfi, qc, prefix=build)

            if len(mfi.replicate.unique()) > 1:
                if corrplot:
                    print(f"There are multiple replicates, generating correlation plots.....")
                    plotting_functions.plot_corrplot(df=corr_df_norm,
                                                     mfi=mfi,
                                                     build=build,
                                                     filename='corrplot_norm.png')
                    plotting_functions.plot_corrplot(df=corr_df_raw,
                                                     mfi=mfi,
                                                     build=build,
                                                     filename='corrplot_raw.png')

            print(f"Report generation is complete!")

    else:
        st.text('Build does not exist; check S3.')
