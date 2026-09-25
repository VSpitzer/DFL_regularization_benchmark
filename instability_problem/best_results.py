"""
Best-Results Summary for the Instability Problem Benchmark

Reads the CSV files produced in 'Rslt/' by test_instability_problem.py (one
file per model/loss/instance combination, named
"{model}ip{loss}{instance}.csv", where instance 1/2/3 correspond to the
instability parameter b = 1/10/100) and reports, for every model and every
instance, the hyperparameter combination with the lowest mean validation
regret across the 10 seeds, together with its corresponding mean test regret.

This script is only meaningful for a hyperparameter-grid-sweep run --
'--config config_grid.json' (instances 1/2/3) or '--config config_4.json'
(the b=25 instance) -- any rows in Rslt/ that came from a different config
(e.g. a '--config config.json' tuned-hyperparameters run) are silently
ignored, since there's nothing to pick a "best" among for a single already-
tuned combination. The 'config' column itself is never shown in the output.

Usage:
    python best_results.py                     # reads ./Rslt, prints the summary
    python best_results.py --rslt_dir Rslt      # same, explicit directory
    python best_results.py --out summary.csv    # also saves the summary table

Selection rule: for each (model, instance), hyperparameter combinations are
grouped and averaged over seeds, then the combination with the smallest mean
'val_regret' is selected as "best"; its mean 'test_regret' is the reported
held-out result. This mirrors standard practice in the DFL benchmark
literature (select on validation regret, report test regret).

Alongside the mean, the table also reports 'test_regret_std' -- the standard
deviation of that same best combination's test regret across its seeds --
so the reported test performance can be read together with how much it
varies seed-to-seed, not just its average. (MSE is not reported here; only
regret.)

The table also reports 'cost_norm_avg', 'cost_norm_max' and 'cost_norm_min'
-- the average, max and min L2 norm of the *raw* predicted cost vector
y_hat throughout training (before any of a model's own internal
normalization), averaged across the same 10 seeds. These come from
log_cost_norm() in Trainer/PO_models.py, which logs a per-epoch mean/max/min
during training_step; test_instability_problem.py reduces those across a
run's epochs (regret_loss_tracker_{model}_{instance}.json's per-seed
'train_cost_norm_avg/max/min'), and this script averages that across seeds.
To find the entry a summary row's cost-norm figures came from, this script
opens Rslt/regret_loss_tracker_{model}_{instance}.json and looks for the
one key whose own (parsed-back) hyperparameters match the row's -- it does
NOT try to recompute that key from a config file, since the exact same
hyperparameters can serialize to a different-looking key depending on which
config file produced them (field order, int vs float). If no tracker file
or no matching key is found -- most commonly because the Rslt/ data
predates this instrumentation, or came from a run whose config file didn't
include that combination -- the three columns are left blank (NaN) for
that row rather than raising an error.
"""

import argparse
import ast
import csv
import glob
import json
import math
import os
import re
from collections import Counter

import pandas as pd

# Columns that are never hyperparameters (metrics, bookkeeping, pandas artifacts)
METRIC_PREFIXES = ("val_", "test_", "train_", "ptl/")
NON_HYPERPARAM_COLS = {"seed", "time", "epoch", "step", "hp_metric", "config"}

# Files that hold per-instance detailed regret dumps rather than summary rows
EXCLUDE_NAME_PATTERNS = ("Regret", "sampleOutput")

# '--config' values that represent a hyperparameter-grid sweep (as opposed to
# a single already-tuned combination, e.g. config.json) -- rows from any of
# these are what best_results.py picks a "best" combination among. Add a new
# name here whenever a new grid config is introduced for another instance.
GRID_CONFIG_NAMES = {"config_grid.json", "config_4.json"}

# If present, this is the column that identifies the problem "difficulty"
# instance within a single Rslt file (only relevant when a file can mix
# several instance/degree values together; harmless when it can't).
ID_COL_CANDIDATES = ("instance", "deg")


def is_hyperparam_col(col):
    if col in NON_HYPERPARAM_COLS:
        return False
    if col.startswith("Unnamed"):
        return False
    if col.startswith(METRIC_PREFIXES):
        return False
    return True


def row_length_counts(path):
    """Read `path` as raw CSV rows (ignoring blank rows) and return a Counter
    of field-count -> number of rows, or None if the file couldn't be read
    as text. Shared by the consistency precheck, the diagnostic message, and
    the recovery routine, so all three agree on what "inconsistent" means."""
    try:
        with open(path, "r", newline="") as f:
            return Counter(len(row) for row in csv.reader(f) if row)
    except Exception:
        return None


def diagnose_malformed_csv(path, lengths=None):
    """Return a short diagnostic string if the file has inconsistent field
    counts across its rows, else None. This is almost always caused by a
    Rslt/*.csv file being appended to by two different versions of the
    training script (e.g. a partial/older run's rows mixed with a later,
    fixed run's rows), which pandas can't parse as a single table."""
    if lengths is None:
        lengths = row_length_counts(path)
    if lengths is None:
        return "could not be read as text"

    if len(lengths) <= 1:
        return None

    summary = ", ".join(
        "{} row(s) with {} fields".format(n, length)
        for length, n in sorted(lengths.items(), key=lambda kv: -kv[1])
    )
    return (
        "inconsistent field counts across rows ({}) -- this file almost certainly mixes "
        "rows appended by different versions/runs of the training script. Delete Rslt/ "
        "and re-run for a clean, comparable result set.".format(summary)
    )


def find_sibling_schema(path, majority_len):
    """Look at other *.csv files in the same directory whose name differs
    from `path` only by the trailing digits before '.csv' (i.e. same model,
    a different instance/degree), and whose header parses cleanly with
    exactly `majority_len` columns including 'val_regret'. Returns that
    header (a list of column names) or None if no such sibling is found."""
    directory = os.path.dirname(path) or "."
    base = os.path.basename(path)
    m = re.match(r"^(.*?)\d*\.csv$", base)
    if not m:
        return None
    prefix = m.group(1)
    try:
        candidates = sorted(os.listdir(directory))
    except OSError:
        return None
    for fname in candidates:
        if fname == base or not re.match(re.escape(prefix) + r"\d*\.csv$", fname):
            continue
        try:
            cols = list(pd.read_csv(os.path.join(directory, fname), nrows=1).columns)
        except Exception:
            continue
        if len(cols) == majority_len and "val_regret" in cols:
            return cols
    return None


def try_recover(path):
    """Best-effort recovery for a CSV whose rows have inconsistent field
    counts (typically because it was appended to by two different
    versions/runs of the training script): keep only the rows matching the
    majority field count, naming their columns either from the file's own
    header (if the header itself matches the majority length) or, failing
    that, from a sibling file's header (same model, different instance/
    degree) that parses cleanly. Returns (df, note) on success, or
    (None, None) if recovery isn't possible."""
    try:
        with open(path, "r", newline="") as f:
            raw_rows = [r for r in csv.reader(f) if r]
    except Exception:
        return None, None

    if not raw_rows:
        return None, None

    lengths = Counter(len(r) for r in raw_rows)
    if len(lengths) <= 1:
        return None, None  # rows are already consistent -- nothing to recover

    majority_len, _ = lengths.most_common(1)[0]
    header = raw_rows[0]

    if len(header) == majority_len:
        columns = header
        data_rows = [r for r in raw_rows[1:] if len(r) == majority_len]
        dropped = len(raw_rows) - 1 - len(data_rows)
    else:
        columns = find_sibling_schema(path, majority_len)
        if columns is None:
            return None, None
        data_rows = [r for r in raw_rows if len(r) == majority_len]
        dropped = len(raw_rows) - len(data_rows)

    if not data_rows:
        return None, None

    # Match pandas' own read_csv convention: an empty column name (the usual
    # header pandas' own to_csv writes for the unlabeled index column) is
    # normalized to "Unnamed: N". Without this, a blank header taken
    # literally from the raw file (as opposed to going through a sibling
    # file's pd.read_csv, which already normalizes it) would slip past
    # is_hyperparam_col's "Unnamed"-prefix check and get treated as a
    # hyperparameter -- since it holds each row's original unique index
    # value, that silently breaks every downstream grouping into
    # singleton (n_seeds=1) groups instead of raising anything.
    columns = [c if str(c).strip() else "Unnamed: {}".format(i) for i, c in enumerate(columns)]

    df = pd.DataFrame(data_rows, columns=columns)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            df[c] = df[c].map({"True": True, "False": False}).fillna(df[c])

    note = ("recovered {} row(s) with the majority column layout; dropped {} row(s) "
            "with a different column count, likely from an earlier/partial run".format(
                len(data_rows), dropped))
    return df, note


def find_tracker_key(tracker, model, instance, row, hp_cols):
    """Find the single key in `tracker` (a regret_loss_tracker_{model}_
    {instance}.json dict, as loaded) whose parsed-back hyperparameters match
    `row`'s values for `hp_cols`. Each key is literally str(parameters) for
    some run's config entry (see test_instability_problem.py), so it's
    parsed back with ast.literal_eval rather than reconstructed from a
    config file -- reconstruction is fragile because the *same* numeric
    hyperparameters can produce a differently-ordered/-typed dict (and thus
    a different string) depending on which config file produced them: e.g.
    config_grid.json's plain-JSON "alpha": 2 versus config.json's "alpha":
    2.0 with 'max_epochs' listed before 'alpha'. Matching by parsed value
    instead of by string makes this robust to that, and to any other config
    file naming/ordering convention. Floats are compared with a relative
    tolerance; NaN values in `row` (a hyperparameter that doesn't apply to
    this model) are skipped rather than treated as a mismatch. Returns the
    matching key (the original string, unparsed, to look the tracker up
    with), or None if zero or more than one key matches."""
    matches = []
    for key in tracker.keys():
        try:
            parsed = ast.literal_eval(key)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(parsed, dict) or parsed.get("model") != model:
            continue
        try:
            if not math.isclose(float(parsed.get("instance")), float(instance)):
                continue
        except (TypeError, ValueError):
            continue
        ok = True
        for col in hp_cols:
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            if col not in parsed:
                ok = False
                break
            try:
                if not math.isclose(float(val), float(parsed[col]), rel_tol=1e-9, abs_tol=1e-12):
                    ok = False
                    break
            except (TypeError, ValueError):
                if val != parsed[col]:
                    ok = False
                    break
        if ok:
            matches.append(key)
    return matches[0] if len(matches) == 1 else None


def add_cost_norm_columns(summary, rslt_dir):
    """Add 'cost_norm_avg'/'cost_norm_max'/'cost_norm_min' columns to
    `summary` in place: for each row, open Rslt/regret_loss_tracker_
    {model}_{instance}.json, find the one key whose own hyperparameters
    match the row's (see find_tracker_key), and average that key's per-seed
    'train_cost_norm_avg/max/min' across seeds. This works regardless of
    which config file ('--config config.json', 'config_grid.json', ...)
    produced that tracker data, since matching is done by parsing each
    tracker key back into a dict and comparing values, not by recomputing
    the key from a config file (see find_tracker_key's docstring for why
    that would be fragile). Rows with no tracker file, no matching key, or
    a tracker predating this instrumentation are left as NaN -- this is
    enrichment on a best-effort basis, never a reason to fail the rest of
    the report."""
    cost_norm_cols = ("cost_norm_avg", "cost_norm_max", "cost_norm_min")
    for col in cost_norm_cols:
        summary[col] = float("nan")

    if "model" not in summary.columns or "instance" not in summary.columns:
        return

    # 'scheduler' (like 'config', already excluded via NON_HYPERPARAM_COLS
    # upstream) is a bookkeeping field append_csv_matching_header() stamps
    # onto every CSV row -- it is never part of a config-file entry, so it
    # never appears in a tracker key's parsed dict. Treating it as a
    # hyperparameter to match on would reject every row where it's not NaN
    # (i.e. every row from an older run that had a 'scheduler' column),
    # since "key not in parsed" always fails for it.
    exclude = {"n_seeds", "source_file", "val_regret", "test_regret", "test_regret_std",
               "model", "instance", "scheduler"}
    hp_cols = [c for c in summary.columns if c not in exclude and c not in cost_norm_cols]

    tracker_cache = {}
    for idx, row in summary.iterrows():
        model = row.get("model")
        instance = row.get("instance")
        if pd.isna(model) or pd.isna(instance):
            continue

        cache_key = (model, int(instance))
        if cache_key not in tracker_cache:
            tracker_path = os.path.join(
                rslt_dir, "regret_loss_tracker_{}_{}.json".format(model, int(instance)))
            tracker = None
            if os.path.exists(tracker_path):
                try:
                    with open(tracker_path) as f:
                        tracker = json.load(f)
                except Exception:
                    tracker = None
            tracker_cache[cache_key] = tracker
        tracker = tracker_cache.get(cache_key)
        if not tracker:
            continue

        key = find_tracker_key(tracker, model, instance, row, hp_cols)
        if key is None:
            continue
        seed_dict = tracker.get(key)
        if not seed_dict:
            continue

        avgs, maxs, mins = [], [], []
        for seed_metrics in seed_dict.values():
            if "train_cost_norm_avg" in seed_metrics:
                avgs.append(seed_metrics["train_cost_norm_avg"])
                maxs.append(seed_metrics["train_cost_norm_max"])
                mins.append(seed_metrics["train_cost_norm_min"])
        if avgs:
            summary.at[idx, "cost_norm_avg"] = sum(avgs) / len(avgs)
            summary.at[idx, "cost_norm_max"] = sum(maxs) / len(maxs)
            summary.at[idx, "cost_norm_min"] = sum(mins) / len(mins)


def summarize_file(path):
    """Return (results, basename, error, note). 'results' is a list of 'best
    row' Series (one per distinct id_col value found in this file, usually
    just one). 'error' is None on success (including a legitimately-empty/
    unusable file) or a short diagnostic string if the file could not be
    parsed or recovered at all -- in which case 'results' is always empty.
    'note' is set when the file was successfully but only partially
    recovered (some rows had to be dropped).

    The field-count consistency of the file is checked up front, rather than
    only reacting to a pandas parse error: when a file's rows have mixed
    field counts because it was appended to by an older and a newer run,
    pandas can sometimes "succeed" anyway (its C parser silently treats an
    extra leading field as an implicit index when the first data row has one
    more field than the header), which would otherwise let corrupted/
    misaligned data through uncaught instead of triggering recovery."""
    note = None
    lengths = row_length_counts(path)
    if lengths is not None and len(lengths) > 1:
        df, note = try_recover(path)
        if df is None:
            diag = diagnose_malformed_csv(path, lengths)
            return [], os.path.basename(path), diag or "inconsistent row lengths", None
    else:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            df, note = try_recover(path)
            if df is None:
                diag = diagnose_malformed_csv(path, lengths)
                return [], os.path.basename(path), diag or str(e), None

    if df.empty:
        return [], os.path.basename(path), None, note

    # best_results.py is only meaningful for a hyperparameter-grid sweep;
    # silently ignore any rows that came from a non-grid run (e.g. config.json)
    if "config" in df.columns:
        df = df[df["config"].isin(GRID_CONFIG_NAMES)]
        if df.empty:
            return [], os.path.basename(path), None, note

    hp_cols = [c for c in df.columns if is_hyperparam_col(c)]
    metric_cols = [c for c in ("val_regret", "test_regret") if c in df.columns]

    if not hp_cols or "val_regret" not in metric_cols:
        return [], os.path.basename(path), None, note

    grouped = df.groupby(hp_cols, dropna=False)
    means = grouped[metric_cols].mean()
    # Dispersion across seeds for the test-set metric specifically (the
    # user cares how much the held-out result varies seed-to-seed, not just
    # its average). ddof=1 (pandas' default) -- sample std over the seeds in
    # this group; NaN for a group with a single seed, which is expected.
    stds = grouped[metric_cols].std()
    for c in ("test_regret",):
        if c in metric_cols:
            means[c + "_std"] = stds[c]
    means["n_seeds"] = grouped.size()
    means = means.reset_index()

    id_col = next((c for c in ID_COL_CANDIDATES if c in hp_cols), None)

    results = []
    if id_col is not None:
        subgroups = means.groupby(id_col, dropna=False)
    else:
        means["_all"] = 0
        subgroups = means.groupby("_all")

    for _, sub in subgroups:
        best_row = sub.loc[sub["val_regret"].idxmin()]
        results.append(best_row)

    return results, os.path.basename(path), None, note


def main():
    parser = argparse.ArgumentParser(
        description="Print the best hyperparameter configuration (by mean validation "
                    "regret across seeds) for each model/instance in a Rslt/ directory "
                    "produced by test_instability_problem.py."
    )
    parser.add_argument("--rslt_dir", type=str, default="Rslt",
                         help="Directory containing the result CSVs produced by test_instability_problem.py (default: Rslt)")
    parser.add_argument("--out", type=str, default=None,
                         help="Optional path to also save the summary table as a CSV")
    args = parser.parse_args()

    files = sorted(
        f for f in glob.glob(os.path.join(args.rslt_dir, "*.csv"))
        if not any(p in os.path.basename(f) for p in EXCLUDE_NAME_PATTERNS)
    )

    if not files:
        print("No result files found in '{}'. Run test_instability_problem.py first.".format(args.rslt_dir))
        return

    rows = []
    skipped = []
    errors = []
    notes = []
    for f in files:
        results, fname, error, note = summarize_file(f)
        if error:
            errors.append((fname, error))
            continue
        if note:
            notes.append((fname, note))
        if not results:
            skipped.append(fname)
            continue
        for best_row in results:
            row = best_row.to_dict()
            row["source_file"] = fname
            rows.append(row)

    if not rows:
        print("Found {} result file(s) in '{}', but none had usable data.".format(len(files), args.rslt_dir))
        if notes:
            print("\n{} file(s) were partially recovered:".format(len(notes)))
            for fname, note in notes:
                print("  - {}: {}".format(fname, note))
        if errors:
            print("\n{} file(s) could not be parsed:".format(len(errors)))
            for fname, err in errors:
                print("  - {}: {}".format(fname, err))
        return

    summary = pd.DataFrame(rows)

    add_cost_norm_columns(summary, args.rslt_dir)

    # Put the most informative columns first
    front_cols = [c for c in ("model", "instance", "deg", "lr", "kappa", "alpha", "lambda_val", "sigma",
                              "num_samples", "val_regret", "test_regret", "test_regret_std",
                              "cost_norm_avg", "cost_norm_max", "cost_norm_min",
                              "n_seeds") if c in summary.columns]
    other_cols = [c for c in summary.columns if c not in front_cols and c != "source_file"]
    summary = summary[front_cols + other_cols + ["source_file"]]

    sort_cols = [c for c in ("instance", "deg", "model") if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\nBest hyperparameters per model (selected on lowest mean val_regret across {} seeds; "
          "test_regret_std is the std dev of that combination's test regret across seeds; "
          "cost_norm_avg/max/min are the average/max/min L2 norm of the raw predicted cost vector "
          "throughout training, averaged across seeds -- blank where no matching tracker data was found):\n".format(
        int(summary["n_seeds"].max()) if "n_seeds" in summary.columns else "?"))
    print(summary.to_string(index=False, float_format=lambda x: "{:.6g}".format(x)))

    if skipped:
        print("\nSkipped {} file(s) with no usable 'val_regret' column: {}".format(
            len(skipped), ", ".join(skipped)))

    if notes:
        print("\n{} file(s) had inconsistent rows and were partially recovered (included above):".format(len(notes)))
        for fname, note in notes:
            print("  - {}: {}".format(fname, note))

    if errors:
        print("\n{} file(s) could not be parsed and were excluded from the summary above:".format(len(errors)))
        for fname, err in errors:
            print("  - {}: {}".format(fname, err))

    if args.out:
        summary.to_csv(args.out, index=False)
        print("\nSaved summary table to '{}'".format(args.out))


if __name__ == "__main__":
    main()
