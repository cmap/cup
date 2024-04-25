build_heatmap_ctl_mfi = 'This heatmap shows the un-normalized median logMFI value for each pool within each detection ' \
                        'well across all control wells in the build. Rows are different pools, while columns are ' \
                        'detection plate wells sorted by plate and well_id. This plot is intended to allow us to pick up ' \
                        'any gross treatment plate based issues such as apparent killing by our vehicle control, ' \
                        'failure to kill by our positive control(s), or an issue with the lysis or collapse of a given ' \
                        'treatment plate.'

plate_heatmap_ctl_mfi = 'These heatmaps are subsets of the above heatmaps faceted by detection plate. They are intended ' \
                        'to allow closer inspection of potential issues.'

build_heatmap_count = 'A build-wide assessment of bead counts. The rows here are detection plates, while the columns ' \
                      'are wells. This plot is useful for identifying cross-bead related issues that may be caused by ' \
                      'instrumentation errors such as the failure to add bead to a well, bead removal during the ' \
                      'staining and cleanup process, scanner clogs or laser failures. The scale here ends at 30, ' \
                      'as we are perfectly happy with counts greater than that.'

plate_heatmap_mfi = 'These heatmaps are intended to give plate-wide overviews of median signal. The values are the ' \
                    'median logMFI across all non-control barcodes in a well. Note that vehicle controls and positive ' \
                    'controls are annotated with *v* and *p* respectively.'

plate_heatmap_count = 'These heatmaps show plate views of the median bead counts for non-control barcodes in each ' \
                      'well. It is intended to help diagnose instrumentation issues such as clogged tips, ' \
                      'scanner clogs, or bead washout.'

ctl_quantiles = 'These line plots are intended to give us an idea of how evenly our control barcodes are spanning the ' \
                'range of our cell line bar codes. The x-axis are our control analytes (1-10) and the y-axis is the ' \
                'quantile of all cell line barcodes that the given median control barcode falls into. Note that this ' \
                'is limited to vehicle wells. The ideal distribution is signified by the dashed line on the plot.'

dr_and_er = 'Dynamic range, error rate and floor range are the two key QC metrics that we use when determining whether or not a ' \
            'cell line passes or fails on a given detection plate. **Dynamic range** is given by the difference ' \
            'between the logMFI of the vehicle control and the positive control. The **error rate** is a measure of the ' \
            'overlap between the positive and negative control values for each cell line and is given as **ER = (FP - FN)/n** ' \
            'where FP is the false positive rate, FN is the false negative rate, and n is the total number of controls./n** ' \
            '**Floor range** is defined as the difference between the signal from a cell line in vehicle and a bead that ' \
            'has been coupled to a sequence that lacks any matching sequence.'

pass_by_plate = 'Fractions of cell lines within each detection plate that pass our thresholds for both dynamic range ' \
                'and error rate.'

pass_by_pool = 'Total number of cell lines that pass or fail by pool. This is useful for picking our particularly ' \
               'problematic pools.'

pass_table = 'This table shows the pass and failure rates for cell lines on each detection plate, as well as ' \
             'highlighting the reason for failure (dynamic range or error rate).'

dr_ecdf = 'Cumulative distribution plot of dynamic range values for each detection plate. Note that there are both ' \
          'pre and post normalized versions of these data.'

liver_plots = 'These plots compare the mean absolute deviation (MAD) to the median of the vehicle signal for each ' \
              'cell line in each replicate. Plates are colored by their QC status. This can help to assess potential ' \
              'reasons for failure (was it due to low signal, high variability etc).'

banana_plots = 'These plots compare the median logMFI value of vehicle samples to the median logMFI value of the ' \
               'positive control samples within each plate and for each cell line. Cell lines are highlighted in ' \
               'blue, while control barcodes are highlighted in red. These plots are helpful for comparing overall ' \
               'brightness for both sets of controls and their relationship to the control barocdes.  Note that there ' \
               'are both pre and post-normalzied versions of these plots.'

plate_dists = 'Simple distributions of logMFI values for vehicle controls and positive controls. These plots can be ' \
              'used to get a feel for the separation between the two and can be investigated for signs of signal ' \
              'hittingt the assay ceiling or the noise floor. It is also useful to compare overall signal intensity ' \
              'across plates.'

dr_vs_er = 'Dynamic range versus error rate for each cell line on each detection plate.'

corr = 'Correlation of normalized logMFI values across replicates. These correlations are calculated on a per compound/dose ' \
       'level across each cell line. This means that each point on this plot represents a given compound/dose in a given cell line. ' \
       'Note: replicate sets with only a single replicate will not be included here.'

profiles_removed = 'Profiles refer to replicate collapsed level 5 data. If a profile is removed,' \
                   'it means that we have no data available for a particular compound & dose combination. This occurs' \
                    'when we have less than 2 valid instances for a given condition.'

instances_removed = 'Instances refer to individual datapoints, in this case a particular cell line in a particular ' \
                     'detection well. If an instance is removed, we lose a single replicate of a profile. This does ' \
                     'not necessarily mean that we have no profile for a given compound & does, as profiles are ' \
                     'constructed when we have at least 2 available instances. Possible reasons for an instance being ' \
                    'removed may be low bead count, low control signal, or a treatment well being skipped by lab ' \
                     'instrumentation.'

count_by_pool = 'Median count across all cell lines in a given pool.'

ctlbc_violin = 'Raw logMFI values for each control barcode for each plate. This is subset to only vehicle wells.'

ctlbc_ranks = 'Ranks are calculated for each control barcode in each well. Ideally, BC1 will always rank 1 and ' \
              'BC2 will always rank 2 etc. The violin plots show distribution of ranks for each barcode within ' \
              'each detection plate. Instances of barcode swapping can be seen as multi-modality here where different ' \
              'barcodes overlap. The coefficient given in the violin plots is the average of the pairwise spearman correlations '\
              'of ranks on each plate.'
              
norm_impact = 'These plots show the relationship between pre and post-normalization data for our positive and vehicle controls.' \
       'Generally speaking, we expect to see that normalization leads to lower signal in our poscon and largely similar signal ' \
              'in our vehicle control. When we deviate from these expectations, we tend to see higher numbers of  failures. ' \
                     'The number in each plot represents the fraction of cell lines that pass in that detection plate.'