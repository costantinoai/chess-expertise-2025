# Task Engagement and Novice Diagnostics

## Overview

This analysis characterises how experts and novices engage with the fMRI
1-back preference task. It addresses whether novice preferences reflect
perceptual (visual) or relational (checkmate/strategy) features, and
provides diagnostic metrics for task compliance.

## Methods

### Rationale

Participants performed a two-alternative forced choice (2-AFC) 1-back
preference judgment during fMRI. On each trial they saw a chess board and
pressed one of two buttons to indicate whether they preferred the current
board or the previous one. Button mapping was counterbalanced across runs.

- 40 stimuli: 20 checkmate positions, 20 non-checkmate positions
- 20 visual pairs: each pair consists of one checkmate and one non-checkmate
  board matched for visual appearance (same position with/without checkmate)
- 80 trials per run, 6--10 runs per subject
- First trial of each run excluded (no previous board to compare)

### Data Sources

**Participants**: N=40 (20 experts, 20 novices); sub-04 excluded (button box
malfunction, zero valid responses in all runs)
**Task**: 1-back preference task during fMRI scanning
**Data**: BIDS events files with pairwise preference judgments between 40 chess
board stimuli

### 1. Response Rate

**What it measures**: The proportion of non-first trials on which the
participant pressed either button (current_preferred or previous_preferred).
This is a measure of task compliance (button-pressing rate), not of
discriminative ability.

**Why all non-first trials are included**: In the 2-AFC design, participants
are instructed to make a choice on every trial. There is no condition where
a response is "not expected." Both buttons are available on every trial. A
response rate of 100% means the participant pressed a button on every trial.
A rate below 100% reflects missed button presses (too slow, distracted, or
button-box issues), not selective responding.

**Computation**: For each subject, count trials with a valid preference
(not n/a), divide by total non-first trials. First trials are excluded
because no comparison is possible (only one board has been seen).

**Expected result**: High response rates (~85--96%) for both groups,
confirming participants treated this as a forced choice.

### 2. Checkmate Preference

**What it measures**: Among 1-back comparisons where a checkmate board
appears consecutively with its visually matched non-checkmate partner (i.e.,
the two members of a visual pair appear on adjacent trials), how often is the
checkmate board preferred?

**Why only visual pairs**: By restricting to within-pair comparisons, all
visual features are controlled -- the only difference between the two boards
is the presence or absence of checkmate. This isolates the checkmate signal
from any visual preference.

**Computation**: For each subject and run, iterate through consecutive
trials. When the current and previous stimulus form a visual pair (one
checkmate, one non-checkmate from the same pair), record which was
preferred. The checkmate preference proportion = (number of times checkmate
preferred) / (total visual-pair comparisons with a valid response).

**Baseline**: 50% (chance). If participants cannot detect checkmate, they
should prefer each member equally. Tested with one-sample t-test vs 0.5.

**Expected result**: Experts significantly above 50% (they detect and prefer
checkmate); novices at chance.

**Note**: This is a conservative metric because it only uses the subset of
trials where visual-pair members happen to appear consecutively. The random
stimulus sequence makes this relatively rare (~5--15 such comparisons per
subject per run). The advantage is perfect visual matching; the cost is lower
statistical power per subject.

### 3. Within-Subject Transitivity

**What it measures**: Preference consistency at the individual level. If a
participant prefers board A over B, and B over C, do they also prefer A over
C? Transitive preferences indicate a stable internal ranking; intransitive
preferences indicate noisy or context-dependent judgments.

**Computation**: For each subject:
1. Extract all pairwise preferences from 1-back comparisons across all runs.
2. For each pair (A, B), determine the net preference direction: A > B if A
   was preferred over B more often than B over A. Ties (equal counts) are
   excluded.
3. For all ordered triples (A, B, C) where A > B and B > C, test whether
   A > C. The transitivity proportion = (number of transitive triples) /
   (total testable triples).

**Baseline**: In a complete random tournament (every pair compared), the
expected transitivity under random preferences is 75%. However, the 1-back
task produces incomplete pairwise data (not all pairs are compared), so the
75% baseline is not directly applicable to absolute values. The group
comparison (expert vs novice) remains valid because both groups have
comparable pairwise coverage.

**Expected result**: Both groups near or below the 75% baseline, with no
significant group difference. This confirms that within-category preference
rankings are noisy for both groups, as expected from the sparse 1-back
design.

### 4. Board Preference Profile (C-NC Within-Pair Correlation)

**What it measures**: Whether preferences are driven by visual similarity
(both members of a visual pair liked/disliked together) or by relational
structure (checkmate member preferred over non-checkmate, regardless of
visual pair).

**Computation**:
1. For each board (1--40), compute the marginal selection frequency: how
   often it is chosen as "current preferred" when it appears as the current
   stimulus, across all opponents and all subjects in the group. This is the
   board's "win rate" against random opponents.
2. For each of the 20 visual pairs, extract the checkmate board's frequency
   (C_freq) and the non-checkmate board's frequency (NC_freq).
3. Correlate C_freq with NC_freq across the 20 pairs (Pearson r).

**Interpretation**:
- **High positive C-NC r** (e.g., novice per-subject mean r = 0.675 (group-level r = 0.87)): Both members of a
  visual pair have similar selection frequencies. This means preferences
  are driven by visual features shared within the pair (e.g., complexity,
  color balance), not by checkmate status.
- **Near-zero or negative C-NC r** (e.g., expert r ≈ −0.06): The checkmate
  and non-checkmate members of a pair have unrelated frequencies. This means
  preferences depend on checkmate status (a relational feature), not on the
  visual properties shared within the pair.

**Note on the selection frequency metric**: `current_chosen` = 1 when the
board is preferred as the current stimulus, 0 when the previous board is
preferred instead. This is a positional metric (tied to being the "current"
board). However, since there is no systematic positional bias -- P(choose
current) ≈ 0.50 for same-status trials in both groups -- the selection
frequency is equivalent to a win rate in a tournament with random opponents.

**Computed at two levels**:
- Group-level: average selection frequency across all subjects in the group,
  then correlate C vs NC across 20 pairs. Produces one r per group.
- Per-subject: compute selection frequency per subject, correlate C vs NC
  per subject. Produces one r per subject, enabling group comparison with
  a t-test.

**Fisher z-test**: Tests whether the group-level C-NC correlation differs
between experts and novices.

### 5. Board Preference Feature Drivers

**What it measures**: Which objective board features predict selection frequency in each group. This analysis goes beyond the aggregate C-NC correlation (Diagnostic 4) by identifying the specific visual and structural properties that drive individual board preferences.

**Rationale**: The C-NC correlation shows *that* novices use visual rather than relational features, but does not specify *which* visual features. This analysis extracts 16 board-level features (from FEN: piece count, officer count, center density, material, etc.) and 4 image-level features (entropy, edge density, luminance, contrast) and tests each against selection frequency using Spearman correlations with FDR correction.

**Procedure**:
1. For each board (1-40), extract board-level features from FEN (via python-chess) and image-level features from stimulus PNGs (Shannon entropy, Sobel edge density, luminance statistics).
2. Correlate each feature with the board's mean selection frequency per group (Spearman rho).
3. Apply Benjamini-Hochberg FDR correction across all features within each group (alpha = 0.05).

### 6. Perceptual-to-Relational Feature Gradient

**What it measures**: The structure of the expert-novice preference dissociation across a gradient of 8 features ordered from purely perceptual (image statistics) to deeply relational (checkmate detection), using bivariate correlations, partial correlations, and hierarchical variance partitioning.

**Rationale**: Diagnostic 5 identified individual feature predictors, but many of them are correlated (more pieces → more edges → higher entropy). This analysis (a) selects 8 features spanning the perceptual→relational gradient, (b) computes partial correlations to isolate unique contributions after removing shared variance, and (c) decomposes total explained variance into three ordered blocks (Perceptual, Structural, Strategic-Relational).

**Features** (perceptual → relational):
1. Image entropy (Shannon entropy of pixel histogram)
2. Edge density (proportion of Sobel edge pixels)
3. Piece count (total pieces)
4. Officer count (N+B+R+Q)
5. Center occupation (pieces in c3-f6)
6. King advantage (opponent king exposure minus own king exposure; positive = opponent more threatened; participants play white)
7. Attack advantage (white attack coverage minus black; positive = white controls more squares)
8. Checkmate status (binary)

**Procedure**:
1. Extract all 8 features for each of 40 boards.
2. Bivariate Spearman correlations with FDR (8 features per group).
3. Partial Spearman correlations: residualise both feature and preference on all other 7 features (OLS), then correlate residuals. FDR-corrected.
4. Hierarchical variance partitioning: enter features in 3 blocks (Perceptual → Structural → Strategic-Relational); report delta-R² per block.

**Important caveat**: These 8 features were not balanced or manipulated in the stimulus design — the 40-board stimulus set was constructed to vary checkmate status, strategy type, and visual pairing, not these secondary properties. As a result, features are not orthogonal (e.g., piece count and officer count correlate), and with n=40 boards statistical power is limited. This analysis is therefore **exploratory**: it characterises the pattern of preference drivers but should not be interpreted as causal evidence for specific feature-driven mechanisms.

### Statistical Tests

- **Welch two-sample t-test**: Compares experts vs novices (unequal variance
  assumed). Reports t-statistic, degrees of freedom (Welch-Satterthwaite),
  and two-tailed p-value.
- **One-sample t-test**: Tests group mean against a theoretical baseline
  (50% for checkmate preference, 75% for transitivity, 0 for C-NC r).
- **Cohen's d**: Effect size using pooled SD:
  d = (M_expert − M_novice) / sqrt((SD²_expert + SD²_novice) / 2).
- **Fisher z-test**: Tests difference between two independent Pearson
  correlations. Uses the Fisher z-transformation and normal approximation.

## Dependencies

- Python 3.9+ with packages: numpy, pandas, scipy, matplotlib, statsmodels, python-chess
- Common utilities from `common/` (bids_utils, plotting, logging_utils, script_utils)
- Local `modules/io.py` for familiarisation data loading
- Image processing: scipy.ndimage (Sobel edge detection) for stimulus feature extraction

## Data Requirements

### Input Files

- **BIDS events files**: `BIDS_ROOT/sub-XX/func/sub-XX_task-exp_run-N_events.tsv`
  - Required columns: `stim_id`, `preference` (current_preferred, previous_preferred, or n/a), `onset`
- **Stimulus metadata**: `CONFIG['STIMULI_FILE']` (stimuli.tsv)
  - Required columns: `stim_id`, `check_status`, `visual` (pair mapping)
- **Familiarisation data**: Subject-level pre-scan familiarisation task logs
- **Stimulus images**: PNG files for image-level feature extraction (entropy, edge density, luminance)

### Data Location

Configure the external data root in `common/constants.py`:

```python
# Base folder containing BIDS/, rois/, neurosynth/, stimuli/
_EXTERNAL_DATA_ROOT = Path("/path/to/manuscript-data")
```

## Running the Analysis

### Step 1: Run task engagement diagnostics

```bash
cd chess-supplementary/task-engagement
python 01_task_engagement.py
```

Computes response rate, checkmate preference, within-subject transitivity, and board preference profile (C-NC correlation) for all participants.

### Step 2: Run familiarisation accuracy analysis

```bash
python 02_familiarisation_accuracy.py
```

Analyses pre-scan familiarisation task accuracy (move detection on checkmate boards) per subject and group.

### Step 3: Run board preference feature drivers

```bash
python 04_quantify_preference_drivers.py
```

Extracts 16 board-level (FEN) and 4 image-level features per stimulus, correlates each with selection frequency per group, and applies FDR correction.

### Step 4: Generate task engagement diagnostic figures

```bash
python 91_plot_novice_diagnostics.py
```

Produces the combined 2-row panel: Row 1 (a--d) engagement bar plots; Row 2 (e--f) board preference scatters.

### Step 5: Generate preference feature driver figures

```bash
python 92_plot_preference_features.py
```

Produces board-image panels of top-preferred boards per group and feature-preference correlation plots.

### Step 6: Run perceptual-to-relational gradient analysis

```bash
python 05_perceptual_relational_gradient.py
```

Extracts 8 features along the perceptual→relational gradient, computes bivariate and partial correlations, and performs hierarchical variance partitioning.

### Step 7: Generate gradient figure

```bash
python 93_plot_gradient_panel.py
```

Produces a combined panel with bivariate vs partial correlation bars and stacked variance partitioning.

## Key Results

### Response Rate

Expert M = 0.856 (SD = 0.195), Novice M = 0.949 (SD = 0.108). Both groups
show high task compliance. The slightly lower expert rate reflects a few
subjects with intermittent button-box issues (sub-03: 36%, sub-06: 50%).
Group difference not significant after accounting for these outliers.

### Checkmate Preference

Expert M = 0.756 (SD = 0.317), significantly above 50%: t(18) = 3.52,
p = 0.002. Novice M = 0.511 (SD = 0.131), not above 50%: t(19) = 0.37,
p = 0.71. Group difference: t(23.7) = 3.12, p = 0.005, d = 1.01.

Experts reliably prefer the checkmate member within visual pairs; novices
show no preference. Three experts show reversed preferences (sub-07, sub-08,
sub-16) -- these were identified as button-swap errors and corrected in the
v3 events conversion (see chess-bh-debugging/tasks.md).

### Transitivity

Expert M = 0.373 (SD = 0.113), Novice M = 0.336 (SD = 0.052). Both groups
significantly below the 0.75 random baseline (experts: p < 0.001, novices:
p < 0.001). Group difference not significant.

Low transitivity in both groups reflects the sparse pairwise coverage of the
1-back design rather than genuinely intransitive preferences. With most pairs
compared only once or twice per subject, the net preference direction is
dominated by noise, producing many apparent intransitivities.

### Board Preference Profile

**Group-level C-NC correlations**:
- Expert: r = −0.41 (p = 0.073, trend), ρ = −0.48 (p = 0.032, significant)
- Novice: r = 0.87 (p < 0.001), ρ = 0.84 (p < 0.001)

**Per-subject C-NC correlations**:
- Expert: M = −0.059 (SD = 0.273), not different from 0 (p = 0.358)
- Novice: M = 0.675 (SD = 0.203), significantly above 0 (p < 0.001)
- Group difference: t(33.2) = −9.50, p < 0.001, d = −3.05

Experts' preferences are not driven by visual similarity (C-NC r near zero):
they discriminate between checkmate and non-checkmate members of visual
pairs. Novices' preferences are strongly driven by visual similarity (C-NC
per-subject mean r = 0.675, group-level r = 0.87): both members of a pair
are liked or disliked together, based on
shared visual features rather than checkmate status.

### Board Preference Feature Drivers

**Results** (FDR-corrected):

**Experts** -- Only one predictor survives FDR:
- **Checkmate status**: r = +0.87, pFDR < 0.001. Whether a board is checkmate or non-checkmate explains 75% of variance in expert preference. No other feature reaches even nominal significance. Piece count, material, visual complexity -- all irrelevant.

**Novices** -- Three visual complexity features survive FDR:
- **Officer count** (N + B + R + Q): r = +0.59, pFDR = 0.001. More knights, bishops, rooks, and queens = more preferred.
- **White material**: r = +0.42, pFDR = 0.049. Higher summed piece value = more preferred.
- **Edge density** (image-level): r = +0.41, pFDR = 0.049. More visual edges in the image = more preferred.
- Checkmate status: r = 0.12, p = 0.46 (completely non-significant for novices).

**Interpretation of extreme boards**:

*Expert top 3* (stim 18, 19, 20; all f > 0.89): All three are checkmate boards from the "Easy" category (strategy group 5) -- positions with checkmate in 1 move ("straightforward checkmate" motif). These are the simplest, most immediately recognisable checkmate positions in the stimulus set. Experts identify and prefer them overwhelmingly, confirming that checkmate detection drives their preference.

*Expert bottom 3* (stim 36, 34, 23; all f < 0.12): All three are non-checkmate boards -- positions that look complex but contain no checkmate threat. Experts actively reject these.

*Novice top 3* (stim 24, 4, 10; f = 0.69-0.71): A mix of checkmate and non-checkmate boards. Critically, stim 24 (NC) and stim 4 (C) are members of the *same visual pair* (pair 4) -- their nearly identical selection frequencies (0.71 and 0.69) confirm that novices cannot distinguish checkmate from non-checkmate within a visual pair. These boards share higher piece counts (16-22 pieces) and more officer pieces.

*Novice bottom 3* (stim 2, 1, 21; f = 0.39-0.42): Again a mix of C and NC. Stim 1 (C) and stim 21 (NC) are members of the *same visual pair* (pair 1), again with near-identical frequencies (0.42 and 0.39). These boards have fewer pieces (16-17), fewer officers, and sparser visual layouts.

**Key dissociation**: Expert preferences are driven exclusively by chess-relational content (whether there is a forced checkmate). Novice preferences are driven by low-level visual complexity (more pieces, more officers, busier images). This double dissociation provides direct behavioural evidence for the representational shift from perceptual to strategic encoding that the main RSA analysis measures neurally.

### Perceptual-to-Relational Gradient

**Bivariate correlations** (FDR < 0.05):
- Experts: only checkmate status (r = +0.87). No other feature reaches significance. King advantage and attack advantage show trends (r ≈ +0.24) but do not survive FDR.
- Novices: 3 of 8 features significant — officer count (r = +0.59), edge density (r = +0.41), image entropy (r = +0.38). Advantage features (king, attack) and checkmate status are non-significant.

**Partial correlations** (unique contribution after controlling for all 7 other features):
- Experts: checkmate status (partial r = +0.94, pFDR < 0.001) — strengthened by partialling, confirming it is the sole independent driver.
- Novices: officer count (partial r = +0.55, pFDR = 0.002) — the unique driver after removing shared variance. Image entropy and edge density collapse to near-zero (fully mediated by piece/officer counts). Advantage features contribute no unique variance.

**Variance partitioning** (hierarchical R²):
- Experts: Perceptual 0.7%, Structural 2.3%, Strategic-Relational **92.7%** (total R² = 0.96).
- Novices: Perceptual **12.8%**, Structural **29.2%**, Strategic-Relational 3.4% (total R² = 0.45).

Expert preferences are almost entirely explained by the strategic-relational block (checkmate). Novice preferences are distributed across perceptual and structural blocks, with minimal contribution from strategic features. This quantitatively confirms the representational shift from perceptual to relational encoding with expertise.

**Caveat**: These features were not balanced in the stimulus design (see Methods, Diagnostic 6). Results are exploratory.

## File Structure

```
chess-supplementary/task-engagement/
├── 01_task_engagement.py              # Diagnostics 1-4 (engagement metrics)
├── 02_familiarisation_accuracy.py     # Pre-scan familiarisation task accuracy
├── 04_quantify_preference_drivers.py  # Diagnostic 5: feature-preference correlations (20 features, FDR)
├── 05_perceptual_relational_gradient.py  # Diagnostic 6: 8-feature gradient, partial correlations, variance partitioning
├── 91_plot_novice_diagnostics.py      # Diagnostics 1-4 figure panel
├── 92_plot_preference_features.py     # Diagnostic 5 figure: board images + scatter
├── 93_plot_gradient_panel.py          # Diagnostic 6 figure: bivariate vs partial + variance bars
├── README.md
├── modules/
│   ├── __init__.py
│   └── io.py                          # Familiarisation data loading utilities
└── results/
    └── novice_diagnostics/
        ├── response_rate.csv
        ├── checkmate_preference.csv
        ├── transitivity.csv
        ├── board_preference_profile.csv
        ├── board_preference_group.csv
        ├── preference_ranking_expert.csv
        ├── preference_ranking_novice.csv
        ├── extreme_boards_summary.csv
        ├── feature_matrix.csv                       # Diagnostic 5 (20 features)
        ├── feature_correlations_full.csv            # Diagnostic 5 (20 features, FDR)
        ├── gradient_feature_matrix.csv              # Diagnostic 6 (8 features)
        ├── gradient_bivariate_correlations.csv      # Diagnostic 6 bivariate r + FDR
        ├── gradient_partial_correlations.csv        # Diagnostic 6 partial r + FDR
        ├── gradient_variance_partitioning.csv       # Diagnostic 6 delta-R2 per block
        └── figures/
            ├── novice_diagnostics_panel.pdf         # Diagnostics 1-4
            ├── preference_drivers_panel.pdf         # Diagnostic 5
            ├── gradient_panel.svg                   # Diagnostic 6
            └── panels/
                ├── preference_features_panel.pdf    # Diagnostic 5
                └── gradient_panel.pdf               # Diagnostic 6
```
