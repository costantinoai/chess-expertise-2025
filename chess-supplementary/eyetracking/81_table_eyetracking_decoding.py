"""
Eyetracking decoding — LaTeX table (summary metrics).

Loads results.json from the latest eyetracking decoding analysis and produces
a concise LaTeX table with accuracy (mean, 95% CI), balanced accuracy, F1,
ROC AUC, and one-sample t-test vs chance.
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common.io_utils import find_latest_results_directory
from common.logging_utils import setup_analysis_in_dir, log_script_end
from common.report_utils import generate_latex_table


RESULTS_BASE = Path(__file__).parent / 'results'
RESULTS_DIR = find_latest_results_directory(
    RESULTS_BASE,
    pattern='*_eyetracking_decoding',
    create_subdirs=['tables'],
    require_exists=True,
    verbose=True,
)

config, _, logger = setup_analysis_in_dir(
    results_dir=RESULTS_DIR,
    script_file=__file__,
    extra_config={"RESULTS_DIR": str(RESULTS_DIR)},
    suppress_warnings=True,
    log_name='tables_eyetracking.log',
)

with open(RESULTS_DIR / 'results.json', 'r') as fh:
    res = json.load(fh)

df = pd.DataFrame([
    dict(
        Metric='Accuracy',
        Estimate=f"{res['accuracy_mean']:.3f}",
        CI=f"[{res['accuracy_ci_low']:.3f}, {res['accuracy_ci_high']:.3f}]",
        p_value=f"{res['ttest_vs_chance_p']:.3g}",
    ),
    dict(Metric='Balanced Accuracy', Estimate=f"{res['balanced_accuracy']:.3f}", CI='', p_value=''),
    dict(Metric='F1', Estimate=f"{res['f1']:.3f}", CI='', p_value=''),
    dict(Metric='ROC AUC', Estimate=f"{res['roc_auc']:.3f}", CI='', p_value=''),
])

tables_dir = RESULTS_DIR / 'tables'
csv_path = tables_dir / 'eyetracking_decoding_summary.csv'
tex_path = tables_dir / 'eyetracking_decoding_summary.tex'
df.to_csv(csv_path, index=False)

generate_latex_table(
    df=df,
    output_path=tex_path,
    caption='Eyetracking decoding summary metrics.',
    label='tab:eyetracking_decoding_summary',
    column_format='lccc',
    escape=False,
    logger=logger,
)

log_script_end(logger)

