import plotting_functions
import pandas as pd
import read_build
import df_transform
import io_functions
from metadata import prism_metadata
import s3fs
import streamlit as st


def generate_report(build, api_key):
    # AWS/API setup
    API_URL = 'https://api.clue.io/api/'
    API_KEY = api_key
    SCANNER_URL = API_URL + 'lims_plate'

    # S3FS setup
    fs = s3fs.S3FileSystem(anon=False)
    bucket = 'cup.clue.io'

    # Check if build exists
    build_path = "s3://macchiato.clue.io/builds/" + build + "/build/"
    if fs.exists(build_path):
        print(f"Generating report for {build}.....")

        file_list = fs.ls(build_path)
        qc_file = f"s3://{io_functions.get_file(file_list, 'QC_TABLE')}"
        print('QC file found: ' + qc_file)
        mfi_file = f"s3://{io_functions.get_file(file_list, 'LEVEL3')}"
        print('MFI file found: ' + mfi_file)
        lfc_file = f"s3://{io_functions.get_file(file_list, 'LEVEL5')}"
        print('LFC file found: ' + lfc_file)

        if io_functions.get_file(file_list, 'removed_instances_count'):
            rm_inst_cnt_file = f"s3://{io_functions.get_file(file_list, 'removed_instances_count')}"
            print('Record of low count removal instances found: ' + rm_inst_cnt_file)

        # Read build and create metric dfs
        df_build = read_build.read_build_from_s3(build, data_levels=['qc', 'mfi', 'lfc', 'inst', 'cell'])
        qc = df_build.qc
        mfi = df_build.mfi
        inst = df_build.inst
        cell = df_build.cell
        df_well = df_transform.annotate_pert_types(df_transform.median_plate_well(mfi))

        # Get df of instances that are removed
        instances_removed = df_transform.get_instances_removed(inst=inst, mfi=mfi, cell=cell)
        profiles_removed = df_transform.profiles_removed(df=mfi)

        # Save list of cultures to metadata json
        cultures = list(mfi.culture.unique())
        plates = list(mfi.prism_replicate.unique())
        pert_plates = list(mfi.pert_plate.unique())

        json_data = {'culture': cultures,
                     'plates': plates,
                     'pert_plates': pert_plates}

        filename = 'build_metadata.json'
        io_functions.write_json_to_s3(data=json_data,
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
        cnt_meta = mfi[['prism_replicate', 'count']].groupby('prism_replicate').agg(agg_funcs).reset_index()
        cnt_meta.columns = ['prism_replicate', 'median_count', 'stdev_count', 'var_count', 'iqr_count']
        if 'det_plate' in plate_meta.columns:
            plate_meta = plate_meta.merge(cnt_meta, left_on='det_plate', right_on='prism_replicate', how='right')
        json_data = plate_meta.to_json()
        io_functions.write_json_to_s3(data=json_data,
                                      bucket=bucket,
                                      prefix=build,
                                      filename='plate_metadata.json')

        # add count meta to mfi df
        if 'det_plate' in plate_meta.columns:
            mfi = mfi.merge(plate_meta, on=['prism_replicate'], how='left')
            mfi['plate'] = mfi['prism_replicate'] + "[" + mfi['scanner_id'].astype('str') + "]"
        else:
            mfi['plate'] = mfi['prism_replicate']

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

        # Calculate plate/pool level delta and correlation
        print(f"Calculating deltaLMFI and pool level correlations.....")
        delta_lmfi, pool_corr = df_transform.calculate_delta_lmfi_corr(mfi)

        # Make tables of excluded instances
        print(f"Generating tables of excluded instances....")

        # By plate
        instances_removed_by_plate = instances_removed.groupby(['culture', 'prism_replicate']).size().reset_index(
            name='instances_removed')
        json_data = instances_removed_by_plate.to_json(orient='records')
        io_functions.write_json_table_to_s3(bucket=bucket,
                                            filename='instances_removed_by_plate_table.json',
                                            data=json_data,
                                            prefix=build)

        # By well
        instances_removed_by_well = instances_removed.groupby(['culture', 'pert_well']).size().reset_index(
            name='instances_removed')
        json_data = instances_removed_by_well.to_json(orient='records')
        io_functions.write_json_table_to_s3(bucket=bucket,
                                            filename='instances_removed_by_well.json',
                                            data=json_data,
                                            prefix=build)

        # By compound
        instances_removed_by_compound = instances_removed.groupby(
            ['culture', 'prism_replicate', 'pert_iname']).size().reset_index(name='instances_removed')
        json_data = instances_removed_by_compound.to_json(orient='records')
        io_functions.write_json_table_to_s3(bucket=bucket,
                                            filename='instances_removed_by_compound.json',
                                            data=json_data,
                                            prefix=build)

        # Compound/doses with <2 replicates
        replicates_by_compound = mfi[~mfi.ccle_name.str.contains('invariant')].groupby(
            ['culture', 'pert_plate', 'ccle_name', 'pert_iname', 'pert_dose']).size().reset_index(
            name='n_instances')
        collapsed_instances_removed = replicates_by_compound[replicates_by_compound.n_instances < 2].drop(
            columns=['n_instances'])
        json_data = collapsed_instances_removed.to_json(orient='records')
        io_functions.write_json_table_to_s3(bucket=bucket,
                                            filename='compound_dose_removed.json',
                                            data=json_data,
                                            prefix=build)

        # Make tables of excluded profiles
        print(f"Generating tables of excluded profiles...")
        # By compound
        profiles_removed_by_compound = profiles_removed.groupby(
            ['culture', 'pert_plate', 'pert_iname']).size().reset_index(name='n_profiles')
        json_data = profiles_removed_by_compound.to_json(orient='records')
        io_functions.write_json_table_to_s3(bucket=bucket,
                                            filename='profiles_removed_by_compound.json',
                                            data=json_data,
                                            prefix=build)

        # By cell line
        profiles_removed_by_line = profiles_removed.groupby(
            ['culture', 'pert_plate', 'ccle_name']).size().reset_index(name='n_profiles')
        json_data = profiles_removed_by_line.to_json(orient='records')
        io_functions.write_json_table_to_s3(bucket=bucket,
                                            filename='profiles_removed_by_line.json',
                                            data=json_data,
                                            prefix=build)

        # Generate and save plots
        print("Generating deltaLMFI heatmaps....")
        plotting_functions.plot_delta_lmfi_heatmaps(df=pool_corr, build=build)

        print("Generating deltaLMFI histograms....")
        plotting_functions.plot_delta_lmfi_histograms(df=pool_corr, build=build)

        print("Generating pool correlation heatmaps....")
        plotting_functions.plot_pool_correlations_heatmaps(df=pool_corr, build=build)

        print("Generating pool correlation histograms....")
        plotting_functions.plot_pool_correlation_histograms(df=pool_corr, build=build)

        print("Generating control compound normalization plots....")
        for culture in cultures:
            plotting_functions.make_control_norm_plots(mfi=mfi,
                                                       qc=qc,
                                                       culture=culture,
                                                       build=build)

        print("Generating control variability violin plots....")
        for culture in cultures:
            plotting_functions.make_control_violin_plot(df=mfi,
                                                        build=build,
                                                        culture=culture)

        print("Generating control barcode quantile plots....")
        for culture in cultures:
            plotting_functions.generate_cbc_quantile_plot(df=mfi,
                                                          build=build,
                                                          culture=culture)

        print(f"Generating mfi heatmaps by plate/pool.....")
        for culture in cultures:
            plotting_functions.make_build_mfi_heatmaps(df=mfi,
                                                       build=build,
                                                       vmax=16,
                                                       vmin=4)

        print(f"Generating MFI heatmaps.....")
        plotting_functions.make_pert_type_heatmaps(df=mfi,
                                                   build=build,
                                                   vmax=16,
                                                   vmin=4)

        print(f"Generating COUNT heatmaps.....")
        plotting_functions.make_build_count_heatmaps(df=mfi,
                                                     build=build)
        plotting_functions.make_pert_type_heatmaps(df=mfi,
                                                   build=build,
                                                   metric='count',
                                                   vmax=30,
                                                   vmin=0)

        print(f"Generating pass rate plots.....")
        plotting_functions.plot_pass_rates_by_plate(df=qc_out,
                                                    build=build,
                                                    filename='pass_by_plate.json')
        for culture in cultures:
            plotting_functions.plot_pass_rates_by_pool(df=qc_out[qc_out.culture == culture],
                                                       build=build,
                                                       culture=culture)

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
            plotting_functions.heatmap_plate(df=df_well,
                                             metric='logMFI',
                                             build=build,
                                             culture=culture,
                                             facet_method='grid',
                                             facets='pert_plate ~ replicate',
                                             limits=(4, 16))
            plotting_functions.heatmap_plate(df=df_well,
                                             metric='logMFI_norm',
                                             build=build,
                                             culture=culture,
                                             facet_method='grid',
                                             facets='pert_plate ~ replicate',
                                             limits=(4, 16))
            plotting_functions.heatmap_plate(df=df_well,
                                             metric='count',
                                             build=build,
                                             culture=culture,
                                             facet_method='grid',
                                             facets='pert_plate ~ replicate',
                                             limits=(0, 30))

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
            print(f"There are multiple replicates, generating correlation plots.....")
            for plate in mfi[~mfi.pert_plate.str.contains('BASE')].pert_plate.unique():
                check_df = mfi[mfi.pert_plate == plate]
                for culture in check_df.culture.unique():
                    plotting_functions.make_corrplots(df=mfi,
                                                      pert_plate=plate,
                                                      metric='logMFI_norm',
                                                      build=build,
                                                      culture=culture)
                    plotting_functions.make_corrplots(df=mfi,
                                                      pert_plate=plate,
                                                      metric='logMFI_norm',
                                                      build=build,
                                                      culture=culture)
        # Rank control barcodes in each plate
        ctls = ['prism invariant ' + str(i) for i in range(1, 11)]
        data = mfi[mfi.ccle_name.isin(ctls)]

        # Compute pairwise correlations of CTLBCs in each plate
        ranked_ctls = data.groupby(['prism_replicate', 'pert_plate', 'pert_well', 'culture']).apply(
            df_transform.calculate_ranks)
        ctl_pairwise_corr = df_transform.calculate_avg_spearman_correlation(ranked_ctls)

        # Generate rank plots
        print(f"Generating CTLBC rank heatmaps.... ")
        for culture in ranked_ctls.culture.unique():
            plotting_functions.make_ctlbc_rank_heatmaps(df=ranked_ctls, build=build, culture=culture)

        print("Generating CTLBC rank violin plots....")
        for culture in ranked_ctls.culture.unique():
            plotting_functions.make_ctlbc_rank_violin(df=ranked_ctls, build=build, culture=culture,
                                                      corrs=ctl_pairwise_corr)

        print(f"Report generation is complete!")

    else:
        print(f"Build {build} does not exist; check S3.")
