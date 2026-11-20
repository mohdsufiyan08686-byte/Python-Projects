import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION & VISUAL STYLING
# ==============================================================================
INPUT_CSV = "02_student_performance.csv"
REPORT_DIR = "diagnostic_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Distinct dark-accent & warm-neutral visual palette
plt.style.use("seaborn-v0_8-ticks")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "figure.dpi": 150
})
ACCENT_COLORS = ["#1B4965", "#62B6CB", "#5FA8D3", "#CAE9FF", "#EE6C4D"]


def log_step(step_id: str, heading: str):
    """Prints a clean, minimalist block header for each analytical phase."""
    print(f"\n[{step_id}] {heading}")
    print("-" * (len(step_id) + len(heading) + 4))


# ==============================================================================
# PHASE 1: INGESTION & STRUCTURAL PROFILING (Task 1)
# ==============================================================================
def ingest_and_profile(filepath: str) -> pd.DataFrame:
    log_step("PHASE 01", "Dataset Ingestion & Structural Profiling")
    if not os.path.exists(filepath):
        sys.exit(f"ERROR: Could not locate '{filepath}' in the current directory.")

    raw_df = pd.read_csv(filepath)
    rows, cols = raw_df.shape

    print(f"File Loaded   : {filepath}")
    print(f"Shape         : {rows} records across {cols} fields")
    print(f"Field List    : {', '.join(raw_df.columns)}\n")

    print(">> Head (Rows 1–5):")
    print(raw_df.head(5).to_markdown(index=False))

    print("\n>> Tail (Last 5 Rows):")
    print(raw_df.tail(5).to_markdown(index=False))

    schema_df = pd.DataFrame({
        "Column": raw_df.columns,
        "Dtype": raw_df.dtypes.values,
        "Non_Null": raw_df.notnull().sum().values,
        "Unique_Values": raw_df.nunique().values
    })
    print("\n>> Field Schema & Cardinality:")
    print(schema_df.to_markdown(index=False))

    print("\n>> Numerical & Categorical Summary:")
    print(raw_df.describe(include="all").round(2).to_markdown())

    return raw_df


# ==============================================================================
# PHASE 2: HYGIENE, IMPUTATION & BOUNDARY VALIDATION (Tasks 2, 3, 4)
# ==============================================================================
def sanitize_records(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    log_step("PHASE 02", "Data Hygiene, Median Imputation & Score Boundary Checks")
    clean_df = df.copy()
    diagnostics = {}

    # 1. Missing value audit
    null_series = clean_df.isnull().sum()
    null_table = pd.DataFrame({
        "Field": null_series.index,
        "Null_Count": null_series.values,
        "Null_Share_%": (null_series.values / len(clean_df) * 100).round(2)
    })
    print(">> Missing Value Audit:")
    print(null_table.to_markdown(index=False))

    # 2. Duplicate Student_ID inspection
    dup_count = int(clean_df["Student_ID"].duplicated().sum()) if "Student_ID" in clean_df.columns else 0
    print(f"\n>> Duplicate Student_ID Count: {dup_count}")
    if dup_count > 0:
        clean_df = clean_df.drop_duplicates(subset=["Student_ID"], keep="first").reset_index(drop=True)
        print("   Action Taken: Deduplicated on Student_ID (retained first record).")

    # 3. Median imputation for Attendance
    if "Attendance" in clean_df.columns:
        missing_att = int(clean_df["Attendance"].isnull().sum())
        att_median = float(clean_df["Attendance"].median())
        att_skew = float(clean_df["Attendance"].skew())
        clean_df["Attendance"] = clean_df["Attendance"].fillna(att_median)

        print(f"\n>> Attendance Imputation:")
        print(f"   Missing Entries Filled : {missing_att}")
        print(f"   Imputed Median Value   : {att_median:.2f}% (Distribution Skewness: {att_skew:.2f})")
        print(
            "   Rationale              : Attendance data exhibits skewness due to chronic absenteeism.\n"
            "                            Using the median prevents extreme low-attendance outliers from\n"
            "                            pulling the imputed baseline downward."
        )

    # 4. Examination score range verification [0, 100]
    numeric_fields = clean_df.select_dtypes(include=[np.number]).columns.tolist()
    score_fields = [c for c in numeric_fields if "mark" in c.lower() or "score" in c.lower()]
    if not score_fields:
        non_scores = {"Student_ID", "Attendance", "Study_Hours", "Daily_Study_Hours", "Age"}
        score_fields = [c for c in numeric_fields if c not in non_scores]

    print(f"\n>> Validating Examination Score Columns: {score_fields}")
    out_of_bounds_mask = pd.Series(False, index=clean_df.index)
    for col in score_fields:
        out_of_bounds_mask |= (clean_df[col] < 0) | (clean_df[col] > 100)

    invalid_rows = clean_df[out_of_bounds_mask]
    print(f"   Out-of-Range Records (<0 or >100): {len(invalid_rows)}")
    if not invalid_rows.empty:
        print("\n>> Flagged Invalid Records:")
        print(invalid_rows.to_markdown(index=False))
        for col in score_fields:
            clean_df[col] = clean_df[col].clip(0, 100)
        print("   Action Taken: Clipped invalid entries to [0, 100] bounds.")

    diagnostics["duplicates"] = dup_count
    diagnostics["invalid_scores"] = len(invalid_rows)
    return clean_df, score_fields, diagnostics


# ==============================================================================
# PHASE 3: PERFORMANCE METRICS & GRADE CLASSIFICATION (Tasks 5, 6)
# ==============================================================================
def compute_academics(df: pd.DataFrame, score_fields: list[str]) -> pd.DataFrame:
    log_step("PHASE 03", "Student Scoring & Grade Tier Classification")
    scored_df = df.copy()

    scored_df["Total_Marks"] = scored_df[score_fields].sum(axis=1)
    scored_df["Average_Marks"] = scored_df[score_fields].mean(axis=1).round(2)

    # Grade tier mapping via pd.cut
    grade_bins = [-np.inf, 39.999, 59.999, 69.999, 79.999, 100.0]
    grade_labels = ["F", "D", "C", "B", "A"]
    scored_df["Grade"] = pd.cut(scored_df["Average_Marks"], bins=grade_bins, labels=grade_labels)

    preview_cols = [c for c in ["Student_ID", "Department", "Total_Marks", "Average_Marks", "Grade"] if c in scored_df.columns]
    print(">> Student Performance Preview (First 8 Records):")
    print(scored_df[preview_cols].head(8).to_markdown(index=False))

    grade_dist = (
        scored_df["Grade"]
        .value_counts()
        .reindex(["A", "B", "C", "D", "F"], fill_value=0)
        .reset_index()
    )
    grade_dist.columns = ["Grade_Tier", "Headcount"]
    grade_dist["Cohort_Share_%"] = (grade_dist["Headcount"] / len(scored_df) * 100).round(2)

    print("\n>> Cohort Grade Distribution:")
    print(grade_dist.to_markdown(index=False))

    return scored_df


# ==============================================================================
# PHASE 4: COHORT SEGMENTATION & RISK DISCOVERY (Tasks 7, 8, 9)
# ==============================================================================
def segment_cohort(df: pd.DataFrame) -> dict:
    log_step("PHASE 04", "Departmental Benchmarks, Top Performers & At-Risk Identification")
    insights = {}

    # 1. Department-wise aggregation
    if "Department" in df.columns:
        dept_df = (
            df.groupby("Department")
            .agg(
                Students=("Student_ID", "count"),
                Avg_Marks=("Average_Marks", "mean"),
                Min_Marks=("Average_Marks", "min"),
                Max_Marks=("Average_Marks", "max"),
                Avg_Attendance=("Attendance", "mean")
            )
            .round(2)
            .sort_values("Avg_Marks", ascending=False)
            .reset_index()
        )
        print(">> Department-Wise Academic Comparison:")
        print(dept_df.to_markdown(index=False))

        best_row = dept_df.iloc[0]
        worst_row = dept_df.iloc[-1]
        print(f"\n   Leading Department : {best_row['Department']} ({best_row['Avg_Marks']:.2f} avg)")
        print(f"   Trailing Department: {worst_row['Department']} ({worst_row['Avg_Marks']:.2f} avg)")
        insights["best_dept"] = (best_row["Department"], best_row["Avg_Marks"])
        insights["worst_dept"] = (worst_row["Department"], worst_row["Avg_Marks"])

    # 2. Top 10 Performers
    top10_df = df.sort_values(by=["Average_Marks", "Total_Marks"], ascending=False).head(10)
    top_cols = [c for c in ["Student_ID", "Department", "Attendance", "Total_Marks", "Average_Marks", "Grade"] if c in df.columns]
    print("\n>> Top 10 Academic Performers:")
    print(top10_df[top_cols].to_markdown(index=False))

    # 3. Academic Support / At-Risk Cohort (< 40%)
    support_df = df[df["Average_Marks"] < 40].sort_values("Average_Marks", ascending=True)
    req_cols = [c for c in ["Student_ID", "Department", "Attendance", "Average_Marks"] if c in df.columns]
    print(f"\n>> Students Requiring Academic Support (< 40% Average) — Total: {len(support_df)}")
    if not support_df.empty:
        print(support_df[req_cols].to_markdown(index=False))
    else:
        print("   Zero students fell below the 40% threshold.")
    insights["support_count"] = len(support_df)

    return insights


# ==============================================================================
# PHASE 5: BEHAVIORAL & INFRASTRUCTURE DRIVERS (Tasks 10, 11, 12)
# ==============================================================================
def analyze_behavioral_drivers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log_step("PHASE 05", "Attendance Bands, Study Habits & Internet Access Drivers")
    work_df = df.copy()
    driver_stats = {}

    # 1. Four Attendance Bands
    att_bins = [-np.inf, 59.999, 74.999, 89.999, 100.0]
    att_labels = ["Below 60%", "60–74%", "75–89%", "90–100%"]
    work_df["Attendance_Band"] = pd.cut(work_df["Attendance"], bins=att_bins, labels=att_labels)

    att_summary = (
        work_df.groupby("Attendance_Band", observed=False)
        .agg(
            Students=("Student_ID", "count"),
            Mean_Marks=("Average_Marks", "mean"),
            Median_Marks=("Average_Marks", "median")
        )
        .round(2)
        .reset_index()
    )
    print(">> Performance by Attendance Band:")
    print(att_summary.to_markdown(index=False))

    # 2. Study Hours Correlation
    study_col = next((c for c in work_df.columns if "study" in c.lower()), None)
    if study_col:
        r_val = float(work_df[study_col].corr(work_df["Average_Marks"]))
        print(f"\n>> Study Hours ({study_col}) vs. Average Marks:")
        print(f"   Pearson Correlation (r): {r_val:.4f}")
        driver_stats["study_col"] = study_col
        driver_stats["study_r"] = r_val

    # 3. Internet Access Comparison
    net_col = next((c for c in work_df.columns if "internet" in c.lower()), None)
    if net_col:
        net_df = (
            work_df.groupby(net_col)
            .agg(
                Students=("Student_ID", "count"),
                Mean_Marks=("Average_Marks", "mean"),
                Median_Marks=("Average_Marks", "median"),
                Std_Marks=("Average_Marks", "std")
            )
            .round(2)
            .reset_index()
        )
        print(f"\n>> Internet Accessibility ({net_col}) Comparison:")
        print(net_df.to_markdown(index=False))
        driver_stats["net_col"] = net_col
        driver_stats["net_table"] = net_df

    return work_df, driver_stats


# ==============================================================================
# PHASE 6: NUMPY & IQR OUTLIER DIAGNOSTICS ON PYTHON_MARKS (Tasks 13, 14)
# ==============================================================================
def run_numpy_iqr_audit(df: pd.DataFrame) -> dict:
    log_step("PHASE 06", "NumPy Quartiles & IQR Outlier Audit (Python_Marks)")
    if "Python_Marks" not in df.columns:
        print("WARNING: 'Python_Marks' column missing from dataset.")
        return {}

    arr = df["Python_Marks"].dropna().to_numpy(dtype=float)

    med = float(np.median(arr))
    std = float(np.std(arr, ddof=1))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    stats_table = pd.DataFrame({
        "Metric": ["Median", "Standard Deviation", "Q1 (25th Pct)", "Q3 (75th Pct)", "IQR (Q3 - Q1)", "Lower Fence", "Upper Fence"],
        "Value": [round(med, 2), round(std, 2), round(q1, 2), round(q3, 2), round(iqr, 2), round(lower_fence, 2), round(upper_fence, 2)]
    })
    print(">> NumPy Statistical & IQR Summary for Python_Marks:")
    print(stats_table.to_markdown(index=False))

    outlier_df = df[(df["Python_Marks"] < lower_fence) | (df["Python_Marks"] > upper_fence)]
    print(f"\n>> Potential Python_Marks Outliers Detected: {len(outlier_df)}")
    if not outlier_df.empty:
        cols = [c for c in ["Student_ID", "Department", "Python_Marks", "Average_Marks"] if c in df.columns]
        print(outlier_df[cols].to_markdown(index=False))

    return {"q1": q1, "q3": q3, "med": med, "lower": lower_fence, "upper": upper_fence, "outlier_count": len(outlier_df)}


# ==============================================================================
# PHASE 7: MULTI-PANEL DASHBOARD & INDIVIDUAL CHARTS (Task 15)
# ==============================================================================
def build_visual_suite(df: pd.DataFrame, driver_stats: dict, py_stats: dict):
    log_step("PHASE 07", "Rendering Executive Dashboard & Individual Diagnostic Charts")

    study_col = driver_stats.get("study_col")
    net_col = driver_stats.get("net_col")

    # --- A. Master 2x3 Executive Diagnostic Dashboard ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Student Academic Performance — Diagnostic Analytics Suite", fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: Horizontal Bar Chart for Grade Distribution
    grade_counts = df["Grade"].value_counts().reindex(["A", "B", "C", "D", "F"], fill_value=0)
    axes[0, 0].barh(grade_counts.index[::-1], grade_counts.values[::-1], color="#1B4965", height=0.6)
    axes[0, 0].set_title("1. Grade Tier Headcount", fontweight="bold")
    axes[0, 0].set_xlabel("Students")
    for idx, val in enumerate(grade_counts.values[::-1]):
        axes[0, 0].text(val + 0.3, idx, str(val), va="center", fontweight="bold")

    # Panel 2: Lollipop Chart for Department-Wise Average Marks
    if "Department" in df.columns:
        dept_means = df.groupby("Department")["Average_Marks"].mean().sort_values()
        axes[0, 1].hlines(y=dept_means.index, xmin=0, xmax=dept_means.values, color="#62B6CB", linewidth=3)
        axes[0, 1].plot(dept_means.values, dept_means.index, "o", markersize=9, color="#EE6C4D")
        axes[0, 1].set_title("2. Department Average Marks", fontweight="bold")
        axes[0, 1].set_xlabel("Mean Marks")
        axes[0, 1].set_xlim(0, 105)

    # Panel 3: Line + Marker Trend across Attendance Bands
    band_means = df.groupby("Attendance_Band", observed=False)["Average_Marks"].mean()
    axes[0, 2].plot(band_means.index.astype(str), band_means.values, marker="s", markersize=8, linewidth=2.5, color="#1B4965")
    axes[0, 2].fill_between(band_means.index.astype(str), band_means.values, alpha=0.15, color="#62B6CB")
    axes[0, 2].set_title("3. Marks Progression by Attendance", fontweight="bold")
    axes[0, 2].set_ylabel("Mean Marks")
    axes[0, 2].tick_params(axis="x", rotation=15)

    # Panel 4: Study Hours vs Average Marks Scatter (colored by Grade)
    if study_col:
        sns.scatterplot(data=df, x=study_col, y="Average_Marks", hue="Grade",
                        hue_order=["A", "B", "C", "D", "F"], palette="viridis", s=55, ax=axes[1, 0])
        axes[1, 0].set_title(f"4. Study Hours vs Marks (r={driver_stats.get('study_r', 0):.2f})", fontweight="bold")
        axes[1, 0].set_xlabel("Daily Study Hours")
        axes[1, 0].set_ylabel("Average Marks")

    # Panel 5: Violin Plot for Internet Access Comparison
    if net_col:
        sns.violinplot(data=df, x=net_col, y="Average_Marks", inner="quart",
                       palette=["#62B6CB", "#EE6C4D"], ax=axes[1, 1])
        axes[1, 1].set_title("5. Marks Density by Internet Access", fontweight="bold")
        axes[1, 1].set_xlabel("Internet Access")
        axes[1, 1].set_ylabel("Average Marks")

    # Panel 6: Python Marks Distribution with IQR Fences
    if "Python_Marks" in df.columns:
        sns.histplot(df["Python_Marks"], bins=14, kde=True, color="#1B4965", ax=axes[1, 2])
        if py_stats:
            axes[1, 2].axvline(py_stats["q1"], color="#EE6C4D", linestyle=":", linewidth=1.8, label=f"Q1 ({py_stats['q1']:.1f})")
            axes[1, 2].axvline(py_stats["q3"], color="#2A9D8F", linestyle=":", linewidth=1.8, label=f"Q3 ({py_stats['q3']:.1f})")
            axes[1, 2].legend(fontsize=8)
        axes[1, 2].set_title("6. Python Marks & Quartile Spread", fontweight="bold")
        axes[1, 2].set_xlabel("Python Marks")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    dashboard_path = os.path.join(REPORT_DIR, "executive_summary_dashboard.png")
    plt.savefig(dashboard_path)
    plt.close()

    # --- B. Save Standalone Individual Charts ---
    individual_specs = [
        ("chart_1_grades.png", lambda: sns.countplot(data=df, x="Grade", order=["A", "B", "C", "D", "F"], palette="crest")),
        ("chart_2_departments.png", lambda: sns.barplot(data=df, y="Department", x="Average_Marks", palette="mako", errorbar=None) if "Department" in df.columns else None),
        ("chart_3_attendance.png", lambda: sns.pointplot(data=df, x="Attendance_Band", y="Average_Marks", color="#1B4965")),
        ("chart_4_study_scatter.png", lambda: sns.scatterplot(data=df, x=study_col, y="Average_Marks", color="#EE6C4D", s=65) if study_col else None),
        ("chart_5_internet_violin.png", lambda: sns.violinplot(data=df, x=net_col, y="Average_Marks", inner="box", palette="Set2") if net_col else None),
        ("chart_6_python_dist.png", lambda: sns.histplot(df["Python_Marks"], kde=True, color="#1B4965") if "Python_Marks" in df.columns else None),
    ]

    for fname, plot_fn in individual_specs:
        plt.figure(figsize=(7.5, 4.8))
        res = plot_fn()
        if res is not None:
            plt.title(fname.replace(".png", "").replace("_", " ").title(), fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(REPORT_DIR, fname))
        plt.close()

    print(f"Exported Master Dashboard : {dashboard_path}")
    print(f"Exported 6 Individual PNGs: {REPORT_DIR}/chart_*.png")


# ==============================================================================
# PHASE 8: ACTIONABLE ACADEMIC RECOMMENDATIONS (Task 16)
# ==============================================================================
def print_action_plan(insights: dict, driver_stats: dict):
    log_step("PHASE 08", "Five Evidence-Based Academic Recommendations")

    best_dept, best_avg = insights.get("best_dept", ("N/A", 0.0))
    worst_dept, worst_avg = insights.get("worst_dept", ("N/A", 0.0))
    at_risk_n = insights.get("support_count", 0)
    r_study = driver_stats.get("study_r", 0.0)

    plan = [
        ("Targeted Remedial Cohorting",
         f"Assign the {at_risk_n} students averaging below 40% to small-group tutoring sessions focused on core foundational modules."),
        ("75% Attendance Gateways",
         "Introduce bi-weekly attendance check-ins; cohort averages drop measurably once attendance falls out of the 75–100% tiers."),
        ("Departmental Curriculum Calibration",
         f"Bridge the {best_avg - worst_avg:.2f}-point gap between {best_dept} ({best_avg:.2f}) and {worst_dept} ({worst_avg:.2f}) by standardizing lab exercises and practice assessments."),
        ("Daily Study-Block Scheduling",
         f"Leverage the study-hour relationship (r = {r_study:.2f}) by incorporating structured 90-minute evening study blocks into student timetables."),
        ("Offline Resource & Lab Access",
         "Equip students without home internet access with downloadable offline course bundles and priority evening access to campus computer labs.")
    ]

    for idx, (title, detail) in enumerate(plan, start=1):
        print(f"{idx}. **{title}**: {detail}")


# ==============================================================================
# MAIN WORKFLOW EXECUTION
# ==============================================================================
def main():
    raw_data = ingest_and_profile(INPUT_CSV)
    clean_data, score_cols, _ = sanitize_records(raw_data)
    scored_data = compute_academics(clean_data, score_cols)
    cohort_insights = segment_cohort(scored_data)
    enriched_data, driver_metrics = analyze_behavioral_drivers(scored_data)
    python_metrics = run_numpy_iqr_audit(enriched_data)
    build_visual_suite(enriched_data, driver_metrics, python_metrics)
    print_action_plan(cohort_insights, driver_metrics)

    final_csv = os.path.join(REPORT_DIR, "student_performance_analyzed.csv")
    enriched_data.to_csv(final_csv, index=False)
    print(f"\n[COMPLETE] Final analyzed dataset saved to '{final_csv}'.")


if __name__ == "__main__":
    main()