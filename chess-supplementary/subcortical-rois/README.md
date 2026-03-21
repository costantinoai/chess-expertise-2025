# Subcortical ROI Analysis (CAB-NP Atlas)

## Overview

This supplementary analysis extends the main cortical multivariate pattern analysis (MVPA) to subcortical structures. The main manuscript uses the Glasser parcellation (22 bilateral cortical ROIs) for both SVM decoding and RSA correlation analyses (Figs 4-5). Reviewer #2 noted that subcortex was excluded from the neural RSA analysis (Fig 4) but included in the Neurosynth correlation analysis (Fig 7), and suggested using the Cole-Anticevic Brain-wide Network Partition (CAB-NP; Ji et al., 2019, NeuroImage) to address this gap.

This analysis is **exploratory and supplementary**. All existing cortical analyses remain untouched. The analysis replicates the **exact same MVPA pipeline** (both SVM decoding and RSA correlations) used for cortical ROIs, substituting only the atlas. The goal is to test whether the expertise-related effects observed in cortex extend to subcortical structures associated with memory, navigation, and procedural learning.

## Motivation

The Neurosynth meta-analytic correlations (Fig 7) revealed that expert representations correlate with memory retrieval and navigation networks, which prominently include subcortical structures (hippocampus, caudate, thalamus). Recent work has also highlighted subcortical involvement in task learning and practice:

- **Mill & Cole (2025, Nature Communications)**: Showed that conjunctive task representations originate in subcortex (hippocampus, cerebellum) and spread to cortex with practice. Used the CAB-NP atlas.
- **Tan, Orlando et al. (2025, bioRxiv)**: Highlighted subcortical involvement in task performance and practice.

These findings motivate testing whether chess expertise modulates neural representations in subcortical regions that support memory encoding, habit formation, and cortico-subcortical integration.

## Atlas: Cole-Anticevic Brain-wide Network Partition (CAB-NP)

**Reference**: Ji, J. L., Spronk, M., Kulkarni, K., Repovs, G., Anticevic, A., & Cole, M. W. (2019). Mapping the human brain's cortical-subcortical functional network organization. NeuroImage, 185, 35-57.

**What it provides**:
- 360 cortical parcels (identical to Glasser HCP-MMP1) + 358 subcortical parcels
- All parcels assigned to 12 functional networks
- Volumetric MNI-space version available (CIFTI format with subcortical volume)

**What we use**: Only the 358 subcortical parcels. We do NOT replace our existing Glasser cortical ROIs.

**Atlas source**: https://github.com/ColeLab/ColeAnticevicNetPartition

### Subcortical ROI Grouping

The 358 subcortical parcels from the CAB-NP atlas were grouped into 9 bilateral anatomical ROIs. Each parcel label in the CAB-NP atlas encodes its anatomical structure (e.g., "Default-01_R-Caudate", "Visual1-04_L-Hippocampus"). Parcels were assigned to structures by matching label name keywords, then all left and right hemisphere parcels for each structure were merged into a single bilateral mask with a unique integer label.

| ROI | # Parcels | # Voxels (native) | # Voxels (resampled) | Group | Rationale |
|-----|-----------|-------------------|---------------------|-------|-----------|
| 1. Hippocampus | 29 | 1,559 | 696 | MTL | Memory encoding/retrieval, spatial navigation, relational binding |
| 2. Amygdala | 11 | 647 | 355 | MTL | Salience detection, value-based evaluation of chess positions |
| 3. Caudate | 17 | 1,483 | 1,483 | Basal Ganglia | Procedural learning, goal-directed behavior, habit formation |
| 4. Putamen | 18 | 2,070 | 1,968 | Basal Ganglia | Motor planning, skill automatisation, stimulus-response associations |
| 5. Pallidum | 20 | 557 | 557 | Basal Ganglia | Basal ganglia output, action selection |
| 6. Thalamus | 78 | 3,954 | 3,950 | Diencephalon | Relay station, cortico-subcortical integration, attentional gating |
| 7. NAcc | 13 | 275 | 216 | Basal Ganglia | Reward processing, motivation, effort-reward computation |
| 8. Cerebellum | 125 | 17,853 | 17,846 | Cerebellum | Prediction, internal models, temporal sequencing, error correction |
| 9. Brainstem | 47 | 3,472 | 3,461 | Brainstem | Arousal, neuromodulation, sensory relay |

**Overlap handling**: After resampling, 1,327 voxels overlapped with the cortical Glasser atlas. These were zeroed in the subcortical atlas to avoid any cortical-subcortical conflict. This ensures the subcortical and cortical analyses operate on strictly non-overlapping brain regions.

### ROI Group Colors (Figure Legend)

Bar plots color each ROI by its anatomical group, matching the convention used for cortical Glasser ROIs. A horizontal legend at the bottom of each panel shows the group-color mapping:

| Group | Color | ROIs |
|-------|-------|------|
| MTL | Red (`#e41a1c`) | Hippocampus, Amygdala |
| Basal Ganglia | Blue (`#377eb8`) | Caudate, Putamen, Pallidum, NAcc |
| Diencephalon | Green (`#4daf4a`) | Thalamus |
| Cerebellum | Purple (`#984ea3`) | Cerebellum |
| Brainstem | Orange (`#ff7f00`) | Brainstem |

Expert and novice bars share the same color per ROI (experts = solid fill, novices = hatched). Colorblind-safe alternatives are stored in the `color_cb` column of `region_info.tsv`.

### Atlas Preparation (00_prepare_atlas.py)

1. **Download**: Clone the ColeAnticevicNetPartition GitHub repository
2. **Extract subcortical parcels**: Load the CIFTI dlabel file and extract subcortical brain model voxels by iterating over CIFTI brain model structures, excluding CORTEX_LEFT and CORTEX_RIGHT
3. **Map parcel labels to structures**: For each of the 358 unique subcortical parcel labels, parse the label name (from the CIFTI label table) to identify the anatomical structure (e.g., "Hippocampus", "Caudate", "Thalamus")
4. **Create bilateral ROI volume**: Assign all parcels belonging to the same structure (both hemispheres) a single integer label in a 3D volume using the CIFTI volumetric geometry
5. **Resample**: Use nilearn.image.resample_to_img with nearest-neighbor interpolation to resample from native CIFTI space to the target functional space (MNI152NLin2009cAsym, 2 mm isotropic), using the existing Glasser 22-region atlas as the reference image to ensure exact spatial alignment
6. **Remove overlap**: Zero any voxels that overlap with the cortical Glasser atlas
7. **Save**: Output NIfTI atlas and region_info.tsv metadata file

**Output files**:
```
rois/cab-np/
  tpl-MNI152NLin2009cAsym_res-02_atlas-CABNP_desc-subcortical_bilateral_resampled.nii.gz
  tpl-MNI152NLin2009cAsym_res-02_atlas-CABNP_desc-subcortical_bilateral_resampled.nii
  region_info.tsv
```

## Pipeline: Exact Replication of Cortical MVPA

The analysis replicates the **exact same MVPA pipeline** used for cortical ROIs (both SVM decoding and RSA correlations), with the CAB-NP subcortical atlas in place of the Glasser-22 atlas. Every statistical procedure, parameter, and threshold is identical.

### Step 1: Subject-Level SVM Decoding and RSA (MATLAB / CoSMoMVPA)

**Script**: subcortical_rsa.m (mirrors chess-mvpa/01_roi_mvpa_main.m)

**Identical elements**:
- Same cosmo_fmri_dataset loading from SPM.mat (beta estimates from unsmoothed GLM)
- Same cosmo_remove_useless_data preprocessing
- Same parse_label_regressors function to derive target vectors from SPM label strings
- Same three target dimensions: checkmate (binary), strategy (multi-class, 40 boards), visual_similarity (20-class)
- Same cosmo_fx run-averaging by stimulus identity for RSA
- Same cosmo_target_dsm_corr_measure with center_data=true for RSA correlations
- Same cosmo_classify_svm (LIBSVM) with cosmo_nfold_partitioner and cosmo_balance_partitions for SVM decoding
- Same model RDM construction using compute_rdm with Hamming distance

**Only differences from cortical script**:
1. Atlas points to CAB-NP subcortical bilateral NIfTI instead of Glasser-22
2. Output filenames use roi-cabnp instead of roi-glasser
3. Output directories are mvpa-rsa-subcortical/ and mvpa-decoding-subcortical/
4. Feature-to-ROI mapping uses explicit voxel coordinate lookup (ds.fa.i/j/k -> atlas value) instead of direct 3D mask, because the dataset has a brain-masked feature subset
5. Beta files (.nii.gz) are decompressed to a temporary directory before CoSMoMVPA loading, then cleaned up after loading
6. ROIs with fewer than 6 voxels after cosmo_remove_useless_data are skipped (same threshold as cortical)

**Output per subject** (same format as cortical):
```
BIDS/derivatives/mvpa-decoding-subcortical/sub-XX/
    sub-XX_space-MNI152NLin2009cAsym_roi-cabnp_accuracy.tsv
BIDS/derivatives/mvpa-rsa-subcortical/sub-XX/
    sub-XX_space-MNI152NLin2009cAsym_roi-cabnp_rdm.tsv
```

Each TSV has columns: target, then one column per subcortical ROI name. Rows: checkmate, strategy, visual_similarity. Values: SVM accuracy (0-1) or Spearman correlation (r).

### Step 2: Group-Level RSA Statistics (Python)

**Script**: 02_subcortical_group_rsa.py (mirrors chess-mvpa/02_mvpa_group_rsa.py)

**Procedure** (identical to cortical):

1. **Data loading**: Find all subject-level RSA TSV files using find_subject_tsvs(). Consolidate into a group DataFrame using build_group_dataframe() with expert/novice labels from participants.tsv via get_participants_with_expertise().

2. **Statistical tests per target per ROI** (3 targets x 9 ROIs):

   a. **Experts vs Chance (zero)**: One-sample one-tailed t-test (greater) testing whether expert mean correlation exceeds zero. Null: mu_expert <= 0. Implementation: scipy.stats.ttest_1samp with alternative='greater'.

   b. **Novices vs Chance (zero)**: One-sample one-tailed t-test (greater) testing whether novice mean correlation exceeds zero. Null: mu_novice <= 0. Same implementation.

   c. **Experts vs Novices**: Welch two-sample two-tailed t-test comparing expert and novice mean correlations. Does not assume equal variances. Null: mu_expert = mu_novice. Implementation: scipy.stats.ttest_ind with equal_var=False.

3. **FDR correction**: Benjamini-Hochberg FDR correction applied separately within each target and test type, across the **9 subcortical ROIs** (not 22 as in cortical). Alpha = 0.05. Implementation: statsmodels.stats.multitest.multipletests with method='fdr_bh'.

4. **Effect sizes**: Cohen's d with 95% confidence intervals for all comparisons.

5. **Descriptive statistics**: Group means and 95% CIs for experts and novices per ROI.

### Step 3: Group-Level Decoding Statistics (Python)

**Script**: 03_subcortical_group_decoding.py (mirrors chess-mvpa/03_mvpa_group_decoding.py)

**Procedure** (identical to cortical):

1. Same data loading and group assignment as Step 2, but reading from mvpa-decoding-subcortical/.

2. **Chance-level determination**: Target-specific chance levels derived from stimulus design via derive_target_chance_from_stimuli():
   - checkmate: chance = 0.5 (binary: check vs non-check)
   - strategy: chance = 1/n_classes (multi-class)
   - visual_similarity: chance = 1/n_classes

3. **Statistical tests**: Identical three-test battery (vs chance, between groups) with FDR across 9 ROIs.

4. **Output**: Same CSV format as cortical. Results saved into the unified subcortical_group_stats.pkl under the "svm" key.

### Step 4: Figures (Python)

**Script 91**: 91_plot_subcortical_rsa.py (mirrors chess-mvpa/92_plot_mvpa_rsa.py)
- Three-panel bar plot (Visual Similarity, Strategy, Checkmate RSA)
- Uses centralized PLOT_YLIMITS['rsa_subcortical'] = (-0.03, 0.06) for y-axis
- Uses PLOT_PARAMS['ylabel_correlation_r'] for y-label
- Bars colored by ROI anatomical group; labels grayed for non-significant ROIs
- Expert vs Novice grouped bars with 95% CI error bars
- FDR-corrected significance stars from between-group Welch t-tests
- Figure size: 11.43 x 16.00 cm (matches cortical RSA panel)
- No pial surface maps (not applicable to subcortical structures)

**Script 93**: 93_plot_subcortical_decoding.py (mirrors chess-mvpa/93_plot_mvpa_decoding.py)
- Six-panel figure: SVM decoding (left column) + RSA correlations (right column)
- SVM panels: chance-subtracted accuracy, PLOT_YLIMITS['decoding_subcortical'] = (-0.03, 0.06)
- RSA panels: raw Spearman correlation, PLOT_YLIMITS['rsa_subcortical'] = (-0.03, 0.06)
- Figure size: 18.29 x 13.50 cm (matches cortical decoding panel)
- No RDM visualization panels (model RDMs are identical to cortical)

**Script 92**: 92_plot_atlas_on_mni.py (supplementary atlas visualization)
- Axial slice montages of CAB-NP subcortical atlas overlaid on MNI152 template
- Glasser cortical atlas visualization for comparison
- Combined cortical+subcortical overlay
- Glass brain views via nilearn

## Existing Cortical Pipeline Reference

The cortical pipeline that this analysis replicates:

```
chess-mvpa/
  01_roi_mvpa_main.m           # Subject-level SVM + RSA (MATLAB/CoSMoMVPA)
  02_mvpa_group_rsa.py         # Group-level RSA statistics (Python)
  03_mvpa_group_decoding.py    # Group-level decoding statistics (Python)
  92_plot_mvpa_rsa.py          # RSA bar plots + pial surfaces
  93_plot_mvpa_decoding.py     # SVM + RSA + RDM combined panel
```

## Output Structure

```
chess-supplementary/subcortical-rois/
  README.md                                # This file
  00_prepare_atlas.py                      # Atlas download, extraction, bilateral merge, resample
  subcortical_rsa.m                        # Subject-level SVM + RSA (MATLAB)
  02_subcortical_group_rsa.py              # Group-level RSA statistics
  03_subcortical_group_decoding.py         # Group-level decoding statistics
  91_plot_subcortical_rsa.py               # RSA bar plots
  92_plot_atlas_on_mni.py                  # Atlas visualization on MNI anatomy
  93_plot_subcortical_decoding.py          # SVM + RSA combined panel
  results/
    subcortical_atlas_prep/                # Atlas preparation logs
    subcortical_rois/
      analysis.log
      subcortical_group_stats.pkl          # Unified results (rsa_corr + svm blocks)
      ttest_rsa_corr_*.csv                 # RSA group statistics (9 files)
      ttest_svm_*.csv                      # SVM group statistics (9 files)
      figures/
        subcortical_rsa__RSA_*.svg         # Individual RSA bar axes
        subcortical_svm__SVM_*.svg         # Individual SVM bar axes
        subcortical_svm__RSA_*.svg         # Individual RSA bar axes (decoding panel)
        atlas_subcortical_axial.pdf        # CAB-NP atlas on MNI
        atlas_cortical_axial.pdf           # Glasser atlas on MNI
        atlas_combined_axial.pdf           # Both atlases combined
        atlas_glass_brain.pdf              # Glass brain views
        panels/
          subcortical_rsa_panel.pdf
          subcortical_svm_panel.pdf
```

## Statistical Assumptions and Limitations

- **Normality**: t-tests assume normally distributed correlation coefficients / accuracies within each group and ROI. With n=20 per group, the central limit theorem provides robustness to moderate deviations.
- **Independence**: Subject-level values are assumed independent across participants.
- **Equal variances**: Welch's t-test relaxes the equal variance assumption for between-group comparisons.
- **Spatial dependence**: Subcortical ROIs, while anatomically separated, are functionally connected. FDR correction partially accounts for this.
- **Fisher z-transformation**: Correlation coefficients were not Fisher z-transformed for group-level testing, consistent with the cortical analysis.
- **Exploratory framing**: This analysis tests 9 ROIs (fewer than the 22 cortical ROIs), providing slightly more statistical power per ROI but addressing a broader anatomical scope with fewer targeted hypotheses.
- **ROI size variability**: Subcortical ROIs vary substantially in size (216 voxels for NAcc to 17,846 for Cerebellum), which may affect signal-to-noise ratio. ROIs with fewer than 6 usable voxels after preprocessing are excluded.

## Results

### RSA Correlations

No subcortical ROI showed a significant between-group difference (experts vs novices) for any of the three model RDMs after FDR correction across 9 ROIs (all pFDR > 0.05). RSA correlation magnitudes were substantially smaller than in cortical ROIs (subcortical mean |r| ~ 0.01; cortical mean |r| ~ 0.05-0.10 in visual areas).

- **Checkmate RSA**: No ROI showed significant expert-novice differences. Amygdala showed a nominal trend (t = 1.97, p_uncorr = 0.06, pFDR = 0.54).
- **Strategy RSA**: No significant effects in any subcortical ROI.
- **Visual Similarity RSA**: No significant effects. As expected, subcortical structures do not encode low-level visual similarity.

### SVM Decoding

No subcortical ROI showed significant above-chance decoding or between-group decoding differences after FDR correction. Chance-subtracted accuracy values were near zero across all ROIs and targets (mean ~0.01 for checkmate, ~0.005 for strategy and visual similarity).

### Interpretation

The null results are informative and constrain the interpretation of the main cortical findings. The expertise-related representational shift from visual to strategic encoding appears to be **predominantly cortical** in this task paradigm. Several factors may contribute to the subcortical null:

1. **Effect size**: Subcortical correlations were an order of magnitude smaller than cortical ones, suggesting that subcortical contributions to chess position encoding — if present — are below the sensitivity of ROI-based RSA with n=20 per group.
2. **Signal quality**: Subcortical fMRI signal is inherently noisier due to physiological noise, partial volume effects, and susceptibility artefacts, particularly in MTL and brainstem structures.
3. **Task design**: The 1-back preference task may not optimally engage subcortical memory and learning systems, which might contribute more during active problem-solving or training.
4. **ROI granularity**: Merging 29 hippocampal parcels into a single bilateral ROI may average over functionally distinct subregions (e.g., anterior vs posterior hippocampus) that show opposing effects.

These results do not rule out subcortical involvement in chess expertise more broadly, but they demonstrate that the expertise-modulated representational structure observed in cortex (Fig 4 in the manuscript) does not extend to subcortical structures at the spatial and temporal resolution of this paradigm.

### Y-Axis Scale Note

Subcortical effect sizes are substantially smaller than cortical ones. To make the data readable, subcortical panels use a tighter y-axis range ((-0.03, 0.06)) compared to cortical panels ((-0.06, 0.25)). This range is consistent across both the RSA and SVM subcortical panels.

## Manuscript Integration

Results are reported in a supplementary section ("Subcortical Representational and Decoding Analysis") with:
- Bar plot panels for RSA (91_plot) and SVM+RSA combined (93_plot), matching the style of the main cortical MVPA figures
- Atlas visualization showing the subcortical ROI locations on MNI anatomy
- A brief mention in the main text Discussion noting that expertise effects are predominantly cortical
- A note in the rebuttal letter for Reviewer #2, Major Comment 4, explaining that the CAB-NP subcortical atlas was used to test subcortical involvement, with null results constraining interpretation

## References

- Ji, J. L., Spronk, M., Kulkarni, K., Repovs, G., Anticevic, A., & Cole, M. W. (2019). Mapping the human brain's cortical-subcortical functional network organization. NeuroImage, 185, 35-57.
- Mill, R. D., & Cole, M. W. (2025). Dynamically shifting from compositional to conjunctive brain representations supports cognitive task learning. Nature Communications, 16, 10084.
- Glasser, M. F., et al. (2016). A multi-modal parcellation of human cerebral cortex. Nature, 536(7615), 171-178.
