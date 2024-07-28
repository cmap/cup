import os
from pathlib import Path
import pandas as pd
import s3fs
import streamlit as st
import descriptions
import jwt
import utils
from metadata import prism_metadata
import boto3
import json
import io_functions
from jwta import Authenticator
import botocore
import botocore.session
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig
import json


client = botocore.session.get_session().create_client('secretsmanager')
cache_config = SecretCacheConfig()
cache = SecretCache( config = cache_config, client = client)

secret = cache.get_secret_string('CLUE/general')
json_object = json.loads(secret)
jwtTokenSecret = json_object["jwtTokenSecret"]


# Set base path
base_path = Path(__file__)
# Configure theme
st.set_page_config(layout='wide',page_title='PRISM QC')
st.header("Login")
# AWS/API setup
API_URL = 'https://api.clue.io/api/'
API_KEY = os.environ['API_KEY']

BUILDS_URL = API_URL + 'data_build_types/prism-builds'
# Your API url to get JWT token
TOKEN_URL = API_URL + 'registration/prism-portal-auth'

# Create Authenticator instance
authenticator = Authenticator(TOKEN_URL)

# Add login form
authenticator.login()

# Check user logged-in
def update_params():
    st.experimental_set_query_params(option=st.session_state.qp)

if st.session_state["email"] and st.session_state["init"] and st.session_state["init"]["api_key"]:
  # Write application content
  decoded = jwt.decode(st.session_state["init"]["api_key"], jwtTokenSecret,algorithms=["HS256"])

  if decoded and decoded["roles"]:
      if 'ADMIN' in decoded["roles"] or 'PRISM_CORE' in decoded["roles"] or 'CMAP_CORE' in decoded["roles"]:
          # Add logout button
          authenticator.logout()
          hide_table_row_index = """
                      <style>
                      thead tr th:first-child {display:none}
                      tbody th {display:none}
                      </style>
                      """
          st.markdown(hide_table_row_index, unsafe_allow_html=True)  # hide table indices while displayed


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

          # Inputs
          if build:

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

                  # Get build metadata
                  build_metadata = io_functions.read_json_from_s3(bucket_name=bucket,
                                                                  filename='build_metadata.json',
                                                                  prefix=build)
                  cultures = build_metadata['culture']
                  plates = build_metadata['plates']
                  pert_plates = build_metadata['pert_plates']
                  pert_plates = [plate for plate in pert_plates if 'BASE' not in plate]  # remove BASE plates

                  # Get plate metadata
                  plate_metadata = io_functions.read_json_from_s3(bucket_name=bucket,
                                                                  filename='plate_metadata.json',
                                                                  prefix=build)

                  # Get scanner metadata
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
                                  io_functions.load_image_from_s3(filename=filename, prefix=build)

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
                                      io_functions.load_image_from_s3(filename=filename, prefix=build)
                          with norm:
                              tab_labels = cultures
                              tabs = st.tabs(tab_labels)
                              for label, tab in zip(tab_labels, tabs):
                                  with tab:
                                      filename = f"logMFI_norm_{label}_heatmaps.png"
                                      io_functions.load_image_from_s3(filename=filename, prefix=build)

                      with st.expander('Bead count'):
                          st.header('Bead Count')
                          if 'det_plate' in scanner_table:
                              st.dataframe(scanner_table.drop(columns=['det_plate']))
                          st.subheader('Count by pool')
                          st.markdown(descriptions.count_by_pool)
                          tab_labels = cultures
                          tabs = st.tabs(tab_labels)
                          for label, tab in zip(tab_labels, tabs):
                              with tab:
                                  filename = f"count_{label}_pert_type_heatmap.png"
                                  io_functions.load_image_from_s3(filename=filename, prefix=build)

                          st.subheader('Count by plate')
                          st.markdown(descriptions.build_heatmap_count)
                          tab_labels = cultures
                          tabs = st.tabs(tab_labels)
                          for label, tab in zip(tab_labels, tabs):
                              with tab:
                                  filename = f"{label}_count_heatmap.png"
                                  io_functions.load_image_from_s3(filename=filename, prefix=build)

                          st.subheader('Plate count')
                          st.markdown(descriptions.plate_heatmap_count)
                          tab_labels = cultures
                          tabs = st.tabs(tab_labels)
                          for label, tab in zip(tab_labels, tabs):
                              with tab:
                                  filename = f"count_{label}_heatmaps.png"
                                  io_functions.load_image_from_s3(filename=filename, prefix=build)

                      with st.expander('Pool behavior'):
                          st.header('Pool level deltaLMFI and correlations')
                          st.markdown(descriptions.deltaLMFI)
                          tab_labels = plates
                          tabs = st.tabs(tab_labels)
                          for label, tab in zip(tab_labels, tabs):
                              with tab:
                                  # First row of two columns
                                  row1_col1, row1_col2 = st.columns(2)
                                  with row1_col1:
                                      delta_lmfi_heatmap_filename = f"{label}_deltaLMFI_heatmaps.png"
                                      io_functions.load_image_from_s3(filename=delta_lmfi_heatmap_filename, prefix=build)

                                  with row1_col2:
                                      corr_heatmap_filename = f"{label}_pool_correlation_heatmaps.png"
                                      io_functions.load_image_from_s3(filename=corr_heatmap_filename, prefix=build)

                                  # Second row of two columns
                                  row2_col1, row2_col2 = st.columns(2)
                                  with row2_col1:
                                      delta_lmfi_histogram_filename = f"{label}_deltaLMFI_histograms.png"
                                      io_functions.load_image_from_s3(filename=delta_lmfi_histogram_filename, prefix=build)

                                  with row2_col2:
                                      corr_histogram_filename = f"{label}_pool_correlation_histograms.png"
                                      io_functions.load_image_from_s3(filename=corr_histogram_filename, prefix=build)

                      with st.expander('Control barcodes'):
                          # control barcode quantiles
                          st.header('Control barcode quantiles')
                          st.markdown(descriptions.ctl_quantiles)
                          tab_labels = cultures
                          tabs = st.tabs(tab_labels)
                          for label, tab in zip(tab_labels, tabs):
                              with tab:
                                  filename = f"{label}_cb_quantiles.png"
                                  io_functions.load_image_from_s3(filename=filename, prefix=build)

                          # control barcode variability
                          st.header('Control barcode variability')
                          st.markdown(descriptions.ctlbc_violin)
                          tab_labels = cultures
                          tabs = st.tabs(tab_labels)
                          for label, tab in zip(tab_labels, tabs):
                              with tab:
                                  filename = f"{label}_ctl_violin.png"
                                  io_functions.load_image_from_s3(filename=filename, prefix=build)

                          # control barcode ranks
                          st.header('Control barcode ranks')
                          st.markdown(descriptions.ctlbc_ranks)
                          tab_labels = cultures
                          tabs = st.tabs(tab_labels)
                          for label, tab in zip(tab_labels, tabs):
                              with tab:
                                  heatmap_filename = f"{label}_ctlbc_rank_heatmap.png"
                                  violin_filename = f"{label}_ctlbc_rank_violin.png"
                                  st.subheader(f"By well")
                                  io_functions.load_image_from_s3(filename=heatmap_filename, prefix=build)
                                  st.subheader(f"By plate")
                                  io_functions.load_image_from_s3(filename=violin_filename, prefix=build)

                      with st.expander('Normalization'):
                          st.markdown(descriptions.norm_impact)
                          st.header('Impact on positive controls')
                          tab_labels = cultures
                          tabs = st.tabs(tab_labels)
                          for label, tab in zip(tab_labels, tabs):
                              with tab:
                                  filename = f"{label}_trt_poscon_norm.png"
                                  io_functions.load_image_from_s3(filename=filename, prefix=build)

                          st.header('Impact on vehicle controls')
                          tab_labels = cultures
                          tabs = st.tabs(tab_labels)
                          for label, tab in zip(tab_labels, tabs):
                              with tab:
                                  filename = f"{label}_ctl_vehicle_norm.png"
                                  io_functions.load_image_from_s3(filename=filename, prefix=build)

                      with st.expander('Data removed'):
                          st.header('Instances removed')

                          st.markdown(descriptions.instances_removed)

                          # Establish columns
                          by_plate, by_compound, by_well = st.columns((1, 1.5, 1))

                          # Populate columns
                          by_plate.subheader('By plate')
                          tbl_by_plate = io_functions.load_json_table_from_s3(filename='instances_removed_by_plate_table.json', prefix=build)
                          by_plate.dataframe(tbl_by_plate)

                          by_well.subheader('By well')
                          tbl = io_functions.load_json_table_from_s3(filename='instances_removed_by_well.json',
                                                                     prefix=build)
                          by_well.dataframe(tbl)

                          by_compound.subheader('By compound')
                          tbl = io_functions.load_json_table_from_s3(filename='instances_removed_by_compound.json',
                                                                     prefix=build)
                          by_compound.dataframe(tbl)

                          st.header('Profiles removed')
                          st.markdown(descriptions.profiles_removed)

                          # Establish columns
                          by_compound, by_cell = st.columns((1, 1.5))

                          # Populate columns
                          by_compound.subheader('By compound')
                          tbl = io_functions.load_json_table_from_s3(filename='profiles_removed_by_compound.json',
                                                                     prefix=build)
                          by_compound.dataframe(tbl)

                          by_cell.subheader('By cell line')
                          tbl = io_functions.load_json_table_from_s3(filename='profiles_removed_by_line.json',
                                                                     prefix=build)
                          by_cell.dataframe(tbl)

                      with st.expander('Cell line pass/fail'):
                          # Plot pass rates
                          st.header('Pass rates')
                          st.markdown(descriptions.dr_and_er)
                          by_plate, by_pool = st.tabs(['By plate', 'By pool'])
                          with by_plate:
                              st.markdown(descriptions.pass_by_plate)
                              io_functions.load_plot_from_s3(filename='pass_by_plate.json', prefix=build)
                          with by_pool:
                              st.markdown(descriptions.pass_by_pool)
                              tab_labels = cultures
                              tabs = st.tabs(tab_labels)
                              for label, tab in zip(tab_labels, tabs):
                                  with tab:
                                      filename = f"{label}_pass_by_pool.png"
                                      io_functions.load_image_from_s3(filename=filename, prefix=build)

                          # Show pass/fail table
                          st.subheader('Pass/fail table')
                          st.markdown(descriptions.pass_table)
                          pass_fail = io_functions.load_df_from_s3('pass_fail_table.csv', prefix=build)
                          st.table(
                              pass_fail.reset_index(drop=True).style.bar(subset=['Pass'], color='#006600', vmin=0, vmax=100).bar(
                                  subset=['Fail both', 'Fail error rate', 'Fail dynamic range'], color='#d65f5f', vmin=0,
                                  vmax=100))

                          # Plot dynamic range
                          st.header('Dynamic range')
                          st.markdown(descriptions.dr_ecdf)
                          dr_norm, dr_raw = st.tabs(['Normalized', 'Raw'])
                          with dr_norm:
                              io_functions.load_plot_from_s3(filename='dr_norm.json', prefix=build)
                          with dr_raw:
                              io_functions.load_plot_from_s3(filename='dr_raw.json', prefix=build)

                      with st.expander('Threshold plots'):
                          # Liver plots
                          st.header('Liver plots')
                          st.markdown(descriptions.liver_plots)
                          io_functions.load_plot_from_s3(filename='liverplot.json', prefix=build)

                          # Banana plots
                          st.header('Banana plots')
                          st.markdown(descriptions.banana_plots)
                          banana_normalized, banana_raw = st.tabs(['Normalized', 'Raw'])
                          with banana_normalized:
                              io_functions.load_plot_from_s3('banana_norm.json', prefix=build)
                          with banana_raw:
                              io_functions.load_plot_from_s3('banana_raw.json', prefix=build)

                          # Dynamic range versus error rate
                          st.header('Error rate and dynamic range')
                          st.markdown(descriptions.dr_vs_er)
                          io_functions.load_plot_from_s3('dr_er.json', prefix=build)

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
                                      io_functions.load_image_from_s3(filename=filename, prefix=build)
                          with norm:
                              tab_labels = cultures
                              tabs = st.tabs(tab_labels)
                              for label, tab in zip(tab_labels, tabs):
                                  with tab:
                                      filename = f"{label}_plate_dist_norm.png"
                                      io_functions.load_image_from_s3(filename=filename, prefix=build)

                      # Plot correlations
                      with st.expander('Correlations'):
                          st.header('Correlations')
                          st.markdown(descriptions.corr)
                          tab_labels = cultures
                          tabs = st.tabs(tab_labels)
                          for label, tab in zip(tab_labels, tabs):
                              with tab:
                                  for pert_plate in pert_plates:
                                      filename = f"{pert_plate}:{label}_corrplot.png"
                                      io_functions.load_image_from_s3(filename=filename, prefix=build)


              else:
                  st.text('Some content is missing from this report, try generating it again.\
                  \nIf this problem persists after regeneration, bother John!')


