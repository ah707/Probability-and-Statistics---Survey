import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==============================
# 0. SETTINGS
# ==============================
CSV_PATH = "AI_Tool_Usage_and_Academic_Performance_Survey__INHA_SGCS_Students[1].csv"
# ==============================
# Helper functions for printing
# ==============================
def print_title(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
def print_subtitle(subtitle):
    print("\n" + "-" * 60)
    print(f" {subtitle}")
    print("-" * 60)
def describe_numeric(series):
    """Return a dictionary with descriptive stats for one numeric series."""
    s = series.dropna()
    if len(s) == 0:
        return {}
    desc = {}
    desc["count"] = len(s)
    desc["mean"] = s.mean()
    desc["median"] = s.median()
    desc["mode"] = list(s.mode()) if not s.mode().empty else []
    desc["min"] = s.min()
    desc["max"] = s.max()
    desc["range"] = s.max() - s.min()
    desc["std"] = s.std()
    desc["q1"] = s.quantile(0.25)
    desc["q3"] = s.quantile(0.75)
    return desc
def print_descriptive_table(df, numeric_cols, short_labels):
    rows = []
    for col in numeric_cols:
        stats_dict = describe_numeric(df[col])
        if stats_dict:
            short_name = short_labels.get(col, col)
            rows.append({
                "Variable": short_name,
                "N": stats_dict["count"],
                "Mean": round(stats_dict["mean"], 3),
                "Median": round(stats_dict["median"], 3),
                "Std": round(stats_dict["std"], 3),
                "Min": round(stats_dict["min"], 3),
                "Max": round(stats_dict["max"], 3),
                "Q1": round(stats_dict["q1"], 3),
                "Q3": round(stats_dict["q3"], 3),
                "Range": round(stats_dict["range"], 3),
                "Mode(s)": stats_dict["mode"],
            })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
def confidence_interval_mean(series, confidence=0.95):
    s = series.dropna()
    n = len(s)
    if n < 2:
        return np.nan, np.nan, n
    mean = s.mean()
    sem = stats.sem(s)
    t_val = stats.t.ppf((1 + confidence) / 2.0, n - 1)
    margin = t_val * sem
    return mean - margin, mean + margin, n
def print_ci_table(df, numeric_cols, short_labels, confidence=0.95):
    rows = []
    for col in numeric_cols:
        lower, upper, n = confidence_interval_mean(df[col], confidence=confidence)
        if not np.isnan(lower):
            short_name = short_labels.get(col, col)
            rows.append({
                "Variable": short_name,
                "N": n,
                f"Lower {int(confidence*100)}% CI": round(lower, 3),
                f"Upper {int(confidence*100)}% CI": round(upper, 3),
            })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
def print_frequency_table(series, top_n=None):
    counts = series.value_counts(dropna=False)
    perc = counts / counts.sum() * 100
    table = pd.DataFrame({"Count": counts, "Percent": perc.round(2)})
    if top_n is not None:
        table = table.head(top_n)
    print(table.to_string())
# ==============================
# 1. Loading data
# ==============================
print_title("LOAD DATA")
df = pd.read_csv(CSV_PATH)
print(" CSV loaded successfully.")
print(f"Number of rows (responses): {len(df)}")
print("\n===== Column Names in CSV =====")
for col in df.columns:
    print(col)
# ==============================
# 2. Auto-detecting column names by keyword
# ==============================
def find_col(keyword):
    """
    Find the first column whose name contains the given keyword (case-insensitive).
    Raises KeyError if not found.
    """
    keyword = keyword.lower()
    for c in df.columns:
        if keyword in c.lower():
            return c
    raise KeyError(f"Column containing '{keyword}' not found. "
                   f"Available columns: {list(df.columns)}")
# Adjusting keywords if needed to match your CSV
COL_MAJOR = find_col("major")
COL_AI_EXPERIENCE = find_col("experience level with ai tools")
COL_AI_PURPOSE = find_col("purpose do you most often use ai tools")
COL_AI_BENEFIT = find_col("positively benefited your academic learning")
COL_TOTAL_HOURS = find_col("total hours per week")
COL_AI_HOURS = find_col("actively using ai tools")
COL_GPA = find_col("current cumulative gpa")

print_subtitle("Detected columns (original names)")
print("Major:                ", COL_MAJOR)
print("AI experience:        ", COL_AI_EXPERIENCE)
print("AI purpose:           ", COL_AI_PURPOSE)
print("AI benefit (Q4):      ", COL_AI_BENEFIT)
print("Total study hours Q5: ", COL_TOTAL_HOURS)
print("AI study hours Q6:    ", COL_AI_HOURS)
print("GPA Q7:               ", COL_GPA)

# Short names used for all outputs and graphs
short_labels = {
    COL_AI_BENEFIT: "AI_Benefit",
    COL_TOTAL_HOURS: "Study_Hours",
    COL_AI_HOURS: "AI_Hours",
    COL_GPA: "GPA",
    COL_MAJOR: "Major",
    COL_AI_EXPERIENCE: "AI_Exp",
    COL_AI_PURPOSE: "AI_Purpose",
}
NUMERIC_COLS = [COL_AI_BENEFIT, COL_TOTAL_HOURS, COL_AI_HOURS, COL_GPA]
# ==============================
# 3. Convert numeric columns
# ==============================
print_title("DATA CLEANING: NUMERIC CONVERSION")

for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"Converted to numeric: {short_labels.get(col, col)}")

print("\nPreview of numeric columns (first 5 rows):")
preview = df[NUMERIC_COLS].head().copy()
preview.columns = [short_labels.get(c, c) for c in preview.columns]
print(preview.to_string(index=False))
# ==============================
# 4. Descriptive statistics
# ==============================
print_title("DESCRIPTIVE STATISTICS (NUMERIC VARIABLES)")

print_descriptive_table(df, NUMERIC_COLS, short_labels)
# ==============================
# 5. Frequency tables (categorical)
# ==============================
print_title("FREQUENCY TABLES (CATEGORICAL VARIABLES)")
print_subtitle("Major")
print_frequency_table(df[COL_MAJOR])
print_subtitle("AI Experience Level (AI_Exp)")
print_frequency_table(df[COL_AI_EXPERIENCE])
print_subtitle("Main AI Purpose (AI_Purpose)")
print_frequency_table(df[COL_AI_PURPOSE])
# ==============================
# 6. Confidence intervals for means
# ==============================
print_title("CONFIDENCE INTERVALS FOR MEANS (95%)")
print_ci_table(df, NUMERIC_COLS, short_labels, confidence=0.95)
# ==============================
# 7. Correlation analysis (Pearson)
# ==============================
print_title("CORRELATION ANALYSIS (PEARSON)")
numeric_df = df[NUMERIC_COLS].copy()
corr_matrix = numeric_df.corr()
# Re-label matrix with short names for printing
corr_short = corr_matrix.copy()
corr_short.columns = [short_labels.get(c, c) for c in corr_short.columns]
corr_short.index = [short_labels.get(c, c) for c in corr_short.index]
print_subtitle("Correlation matrix (short labels)")
print(corr_short.round(3).to_string())
print_subtitle("Pairwise correlations with p-values (short labels)")
for i in range(len(NUMERIC_COLS)):
    for j in range(i + 1, len(NUMERIC_COLS)):
        col_x = NUMERIC_COLS[i]
        col_y = NUMERIC_COLS[j]
        name_x = short_labels.get(col_x, col_x)
        name_y = short_labels.get(col_y, col_y)
        sub = df[[col_x, col_y]].dropna()
        if len(sub) > 2:
            r, p = stats.pearsonr(sub[col_x], sub[col_y])
            print(f"{name_x}  vs  {name_y}")
            print(f"  r = {r:.3f},  p-value = {p:.4f}")
        else:
            print(f"{name_x} vs {name_y}: Not enough data points.")
        print()
# ==============================
# 8. Chi-square test (Major vs AI Experience)
# ==============================
print_title("CHI-SQUARE TEST: Major \u00d7 AI_Exp")
contingency = pd.crosstab(df[COL_MAJOR], df[COL_AI_EXPERIENCE])
print_subtitle("Contingency table (Major \u00d7 AI_Exp)")
print(contingency.to_string())
chi2_ok = False
if contingency.shape[0] > 1 and contingency.shape[1] > 1:
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    chi2_ok = True
    print_subtitle("Chi-square test result")
    print(f"Chi-square statistic: {chi2:.3f}")
    print(f"Degrees of freedom:   {dof}")
    print(f"p-value:              {p:.4f}")
else:
    print("Not enough categories for chi-square test.")
# ==============================
# 9. ANOVA: GPA by Major
# ==============================
print_title("ONE-WAY ANOVA: GPA differences by Major")
grouped = df[[COL_MAJOR, COL_GPA]].dropna().groupby(COL_MAJOR)
means = grouped[COL_GPA].mean().round(3)
print_subtitle("Mean GPA by Major")
print(means.to_string())
groups_for_anova = [grp[COL_GPA].values for _, grp in grouped]
anova_ok = False
if len(groups_for_anova) >= 2:
    f_stat, p_val = stats.f_oneway(*groups_for_anova)
    anova_ok = True
    print_subtitle("ANOVA result")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"p-value:     {p_val:.4f}")
else:
    print("Not enough groups for ANOVA (need at least 2).")
# ==============================
# 10. Linear regression
#     (a) GPA ~ AI_Hours
#     (b) GPA ~ AI_Benefit
# ==============================
print_title("LINEAR REGRESSION")
# (a) GPA ~ AI_Hours
print_subtitle("Regression: GPA ~ AI_Hours")
reg_data = df[[COL_AI_HOURS, COL_GPA]].dropna()
reg_hours_ok = False
if len(reg_data) > 2:
    slope, intercept, r_val, p_val, std_err = stats.linregress(
        reg_data[COL_AI_HOURS],
        reg_data[COL_GPA]
    )
    reg_hours_ok = True
    print(f"GPA = {intercept:.3f} + {slope:.3f} * AI_Hours")
    print(f"R-squared: {r_val**2:.3f}")
    print(f"p-value:   {p_val:.4f}")
else:
    print("Not enough data points for regression (AI_Hours vs GPA).")

# (b) GPA ~ AI_Benefit
print_subtitle("Regression: GPA ~ AI_Benefit")
reg_data2 = df[[COL_AI_BENEFIT, COL_GPA]].dropna()
reg_benefit_ok = False
if len(reg_data2) > 2:
    slope2, intercept2, r_val2, p_val2, std_err2 = stats.linregress(
        reg_data2[COL_AI_BENEFIT],
        reg_data2[COL_GPA]
    )
    reg_benefit_ok = True
    print(f"GPA = {intercept2:.3f} + {slope2:.3f} * AI_Benefit")
    print(f"R-squared: {r_val2**2:.3f}")
    print(f"p-value:   {p_val2:.4f}")
else:
    print("Not enough data points for regression (AI_Benefit vs GPA).")
# ==============================
# 11. Z-scores for GPA
# ==============================
print_title("Z-SCORES FOR GPA")
gpa_series = df[COL_GPA].dropna()
z_df = None
if len(gpa_series) > 1:
    mean_gpa = gpa_series.mean()
    std_gpa = gpa_series.std()
    z_scores = (gpa_series - mean_gpa) / std_gpa
    z_df = pd.DataFrame({
        "GPA": gpa_series,
        "GPA_z_score": z_scores
    })
    print(z_df.head(10).to_string(index=False))
else:
    print("Not enough GPA data to compute z-scores.")
# ==============================
# 12. GRAPHS WITH SHORT LABELS
# ==============================

print_title("GENERATING GRAPHS (SHORT LABELS)")

# ---- 12.1 Histograms for each numeric variable ----
for col in NUMERIC_COLS:
    plt.figure()
    df[col].dropna().plot(kind="hist", bins=10)
    plt.title(f"Histogram - {short_labels[col]}")
    plt.xlabel(short_labels[col])
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

# ---- 12.2 Bar charts for categorical variables ----

# Major
plt.figure()
df[COL_MAJOR].value_counts().plot(kind="bar")
plt.title("Count by Major")
plt.xlabel("Major")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# AI Experience
plt.figure()
df[COL_AI_EXPERIENCE].value_counts().plot(kind="bar")
plt.title("AI Experience Levels")
plt.xlabel("AI_Exp")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# AI Purpose
plt.figure()
df[COL_AI_PURPOSE].value_counts().plot(kind="bar")
plt.title("Main AI Purpose")
plt.xlabel("AI_Purpose")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# ---- 12.3 CI Plot for numeric variables ----
means_list, lowers, uppers, labels = [], [], [], []

for col in NUMERIC_COLS:
    low, up, n = confidence_interval_mean(df[col])
    if not np.isnan(low):
        labels.append(short_labels[col])
        lowers.append(low)
        uppers.append(up)
        means_list.append(df[col].dropna().mean())

if labels:
    plt.figure()
    x = np.arange(len(labels))
    means_arr = np.array(means_list)
    yerr = np.vstack((means_arr - np.array(lowers), np.array(uppers) - means_arr))
    plt.errorbar(x, means_arr, yerr=yerr, fmt='o')
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title("Means with 95% CI")
    plt.ylabel("Mean")
    plt.tight_layout()

# ---- 12.4 Correlation heatmap ----
if not corr_matrix.empty:
    plt.figure()
    plt.imshow(corr_matrix, interpolation='nearest')
    plt.title("Correlation Heatmap")
    plt.colorbar()
    tick_marks = np.arange(len(NUMERIC_COLS))
    short_names = [short_labels[c] for c in NUMERIC_COLS]
    plt.xticks(tick_marks, short_names, rotation=45, ha="right")
    plt.yticks(tick_marks, short_names)
    plt.tight_layout()

# ---- 12.5 Chi-square stacked bar ----
if chi2_ok:
    plt.figure()
    row_sums = contingency.sum(axis=1)
    cont_prop = contingency.div(row_sums, axis=0)
    bottom = np.zeros(len(cont_prop))

    for col_lab in cont_prop.columns:
        plt.bar(cont_prop.index, cont_prop[col_lab], bottom=bottom, label=col_lab)
        bottom += cont_prop[col_lab]

    plt.title("AI_Exp by Major (Proportion)")
    plt.xlabel("Major")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="AI_Exp")
    plt.tight_layout()

# ==============================
# VISUAL GRAPH FOR CHI-SQUARE RESULT (MOST EXPLANATORY)
# ==============================

if chi2_ok:

    # expected frequencies table
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

    # standardized residuals highlight strengths of deviation
    residuals = (contingency - expected_df) / np.sqrt(expected_df)

    plt.figure(figsize=(8,6))
    plt.imshow(residuals, cmap='coolwarm')
    plt.title("Chi-Square Standardized Residual Heatmap")
    plt.colorbar(label="Residual Value")
    plt.xticks(np.arange(len(contingency.columns)), contingency.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(contingency.index)), contingency.index)
    plt.tight_layout()
    plt.show()

    # short summary auto print
    print("\n--- Chi-square Interpretation from Residual Heatmap ---")
    print("• Red = Higher than expected (contributes positively to χ²)")
    print("• Blue = Lower than expected (contributes negatively to χ²)")
    print("• The stronger the color, the stronger the deviation")
    print("• These cells are the reason p = 0.0478 < 0.05 gave significance")


# ---- 12.6 ANOVA Boxplot ----
if anova_ok and len(grouped) > 0:
    plt.figure()
    data_to_plot = [group[COL_GPA].values for _, group in grouped]
    labels_bp = [name for name, _ in grouped]
    plt.boxplot(data_to_plot, labels=labels_bp)
    plt.title("GPA by Major (Boxplot)")
    plt.xlabel("Major")
    plt.ylabel("GPA")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

# ---- 12.7 Regression graphs ----

# GPA vs AI_Hours
if reg_hours_ok:
    plt.figure()
    x = reg_data[COL_AI_HOURS]
    y = reg_data[COL_GPA]
    plt.scatter(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = intercept + slope * xs
    plt.plot(xs, ys)
    plt.title("GPA vs AI_Hours")
    plt.xlabel("AI_Hours")
    plt.ylabel("GPA")
    plt.grid(True)
    plt.tight_layout()

# GPA vs AI_Benefit
if reg_benefit_ok:
    plt.figure()
    x2 = reg_data2[COL_AI_BENEFIT]
    y2 = reg_data2[COL_GPA]
    plt.scatter(x2, y2)
    slope2, intercept2, _, _, _ = stats.linregress(x2, y2)
    xs2 = np.linspace(x2.min(), x2.max(), 100)
    ys2 = intercept2 + slope2 * xs2
    plt.plot(xs2, ys2)
    plt.title("GPA vs AI_Benefit")
    plt.xlabel("AI_Benefit")
    plt.ylabel("GPA")
    plt.grid(True)
    plt.tight_layout()

# ---- 12.8 Z-score histogram ----
if z_df is not None:
    plt.figure()
    z_df["GPA_z_score"].plot(kind="hist", bins=10)
    plt.title("GPA Z-score Distribution")
    plt.xlabel("Z-score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

print(" All graphs created. Close the plot windows to end the program.")
plt.show()
