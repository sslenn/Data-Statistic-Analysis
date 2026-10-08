# %% [markdown]
# # Lab 01 — Descriptive Statistics in Python
# ## ✅ FULL SOLUTION (Instructor Version)
# **Department of Applied Mathematics and Statistics (AMS), Institute of Technology of Cambodia**
# Asst. Prof. Dr. PHAUK Sokkhey
#
# ---
#
# ### Learning objectives
# By the end of this lab you will be able to:
# 1. Classify variables as **nominal, ordinal, interval, or ratio**.
# 2. Compute and interpret **mean, median, mode, range, variance, standard deviation, IQR, CV, skewness and kurtosis** using built-in functions.
# 3. Build a **five-number summary**, detect outliers with the $1.5\times\text{IQR}$ rule, and draw a **boxplot**.
# 4. Summarise **categorical data** with a frequency table (frequency, percent, cumulative percent).
#
# ### Lab structure
# | Part | Content | Status |
# |---|---|---|
# | A — Exercise 1 | Tensile strength of steel specimens (quantitative data) | Solved |
# | A — Exercise 2 | Salaries of Company A vs B + marital-status table | Solved |
# | B — Practice 1 | Test scores in English, Math, Physics | Solved |
# | B — Practice 2 | Defect types and customer ratings on a production line | Solved |

# %% [markdown]
# ### Formula reference (from Lecture 01)
#
# | Measure | Population | Sample | Python |
# |---|---|---|---|
# | Mean | $\mu=\dfrac{\sum x_i}{N}$ | $\bar x=\dfrac{\sum x_i}{n}$ | `s.mean()` |
# | Variance | $\sigma^2=\dfrac{\sum (x_i-\mu)^2}{N}$ | $s^2=\dfrac{\sum (x_i-\bar x)^2}{n-1}$ | `s.var(ddof=0)` / `s.var()` |
# | Std. deviation | $\sigma=\sqrt{\sigma^2}$ | $s=\sqrt{s^2}$ | `s.std(ddof=0)` / `s.std()` |
# | Coefficient of variation | | $CV=\dfrac{s}{\bar x}\times100\%$ | `s.std()/s.mean()*100` |
# | IQR | | $Q_3-Q_1$ | `s.quantile(.75)-s.quantile(.25)` |
# | Pearson skewness | | $Sk=\dfrac{3(\bar x-\text{median})}{s}$ | — |
# | Kurtosis | | normal $=3$ (excess $=0$) | `stats.kurtosis(x, fisher=False)` |
#
# > ⚠️ **Watch the defaults:** `pandas` uses `ddof=1` (sample), `numpy` uses `ddof=0` (population). `pandas.Series.kurt()` returns **excess** kurtosis (normal $=0$), not the lecture's scale (normal $=3$).

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option("display.precision", 3)
plt.rcParams["figure.figsize"] = (8, 4)

# %% [markdown]
# ---
# # Part A — Worked Exercises (with Solutions)
#
# ## Exercise 1 — Tensile Strength of Steel Specimens
# A materials lab tested 20 steel rods. Tensile strength (MPa) is recorded below.
# The last specimen was mis-recorded during data entry (an **outlier**).
#
# **Tasks**
# 1. Identify the level of measurement of `strength`.
# 2. Compute mean, median, mode.
# 3. Compute range, sample variance, sample SD, CV, and IQR.
# 4. Build the five-number summary, find outliers with the $1.5\times\text{IQR}$ rule, and draw a boxplot.
# 5. Compute skewness and kurtosis and interpret the shape.
# 6. Remove the outlier and compare: which measures are **robust**?

# %%
strength = pd.Series([412, 425, 418, 430, 421, 415, 428, 419, 433, 422,
                      417, 426, 420, 424, 431, 418, 423, 427, 416, 610],
                     name="strength_MPa")
strength.describe()

# %% [markdown]
# **1. Level of measurement.** Tensile strength has a true zero (0 MPa = no strength) and ratios are meaningful (400 MPa is twice 200 MPa) → **ratio**, quantitative, continuous.
#
# **2. Measures of central tendency**

# %%
center = pd.Series({
    "Mean":   strength.mean(),
    "Median": strength.median(),
    "Mode":   strength.mode().tolist(),
})
center

# %% [markdown]
# The mean (≈ 431.8) is pulled above the median (422.5) by the 610 MPa outlier. The mode (418) is not very informative for continuous data.
#
# **3. Measures of variability**

# %%
Q1, Q3 = strength.quantile([0.25, 0.75])
spread = pd.Series({
    "Range":            strength.max() - strength.min(),
    "Variance (s^2)":   strength.var(),           # ddof=1 → sample
    "Variance (σ^2)":   strength.var(ddof=0),     # population
    "SD (s)":           strength.std(),
    "CV (%)":           strength.std() / strength.mean() * 100,
    "IQR":              Q3 - Q1,
})
spread

# %% [markdown]
# **4. Five-number summary, outlier fences, boxplot**
#
# Lower fence $=Q_1-1.5\,\text{IQR}$, Upper fence $=Q_3+1.5\,\text{IQR}$.

# %%
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

five_num = pd.Series({"Min": strength.min(), "Q1": Q1, "Median": strength.median(),
                      "Q3": Q3, "Max": strength.max()})
print(five_num, f"\nFences: [{lower:.2f}, {upper:.2f}]")
print("Outliers:", strength[(strength < lower) | (strength > upper)].tolist())

strength.plot.box(vert=False, title="Tensile strength (MPa)")
plt.axvline(upper, ls="--", c="red", label="Upper fence"); plt.legend(); plt.show()

# %% [markdown]
# **5. Shape: skewness and kurtosis**

# %%
shape = pd.Series({
    "Pearson skew 3(mean-median)/s": 3 * (strength.mean() - strength.median()) / strength.std(),
    "Moment skew (scipy)":           stats.skew(strength),
    "Kurtosis (normal = 3)":         stats.kurtosis(strength, fisher=False),
    "Excess kurtosis (pandas)":      strength.kurt(),
})
shape

# %% [markdown]
# **Interpretation.** Moment skewness ≈ 4 $>1$ → **highly right-skewed**. Kurtosis ≈ 17 $\gg 3$ → **leptokurtic** (heavy tail), driven entirely by one extreme value.
#
# > Note that Pearson's formula gives only ≈ 0.66 here: it uses the median, which ignores the extreme value, so it *understates* skewness caused by a single outlier. Software (`scipy`, `pandas`, Excel `SKEW`) uses the moment-based definition.
#
# **6. Effect of removing the outlier**

# %%
clean = strength[strength.between(lower, upper)]
compare = pd.DataFrame({
    "With outlier":    [strength.mean(), strength.median(), strength.std(), Q3 - Q1, stats.skew(strength)],
    "Without outlier": [clean.mean(), clean.median(), clean.std(),
                        clean.quantile(.75) - clean.quantile(.25), stats.skew(clean)],
}, index=["Mean", "Median", "SD", "IQR", "Skewness"])
compare["Change"] = compare["Without outlier"] - compare["With outlier"]
compare

# %% [markdown]
# **Conclusion.** One bad value moved the mean by ≈ 9.4 MPa and inflated the SD about seven-fold (5.8 → 42.3), while the **median and IQR barely changed** — they are *robust* measures. For engineering QC data with possible recording errors, report the median and IQR alongside the mean and SD.

# %% [markdown]
# ---
# ## Exercise 2 — "The Deceptive Average" + Categorical Summary
#
# ### 2a. Company A vs Company B
# Both companies report a mean salary of \$70,000 (Lecture slide *Central Tendency*).
# - Company A: 50 employees each earn \$70,000.
# - Company B: 25 earn \$35,000 and 25 earn \$105,000.
#
# **Tasks:** Compare mean, median, SD, CV and range. Which company would you join?

# %%
salary = pd.DataFrame({
    "Company A": np.repeat(70_000, 50),
    "Company B": np.repeat([35_000, 105_000], 25),
})

summary = salary.agg(["mean", "median", "std", "min", "max"]).T
summary["CV (%)"] = summary["std"] / summary["mean"] * 100
summary["Range"]  = summary["max"] - summary["min"]
summary

# %%
salary.plot.hist(bins=20, alpha=0.7, title="Salary distribution"); plt.xlabel("USD"); plt.show()

# %% [markdown]
# **Interpretation.** Same mean and median, but Company A has SD = 0 while Company B has SD ≈ \$35,355 (CV ≈ 50%). A new employee at B has a 50% chance of earning only half the "average". Note also that B's median (\$70,000) is the average of the two middle values — **no one at B actually earns it**. **A measure of center is incomplete without a measure of spread.**
#
# ### 2b. Frequency table for a nominal variable
# Reproduce the lecture's marital-status table for a sample of 80 people.

# %%
marital = pd.Series(["Single"] * 44 + ["Married"] * 29 + ["Other"] * 7, name="Marital status")

freq = marital.value_counts().to_frame("Frequency")
freq["Percent"]            = marital.value_counts(normalize=True) * 100
freq["Cumulative Percent"] = freq["Percent"].cumsum()
freq.loc["Total"] = [freq["Frequency"].sum(), 100, np.nan]
freq.astype({"Frequency": int}).round(1)

# %%
print("Mode:", marital.mode()[0])
marital.value_counts().plot.bar(rot=0, title="Marital status (n = 80)"); plt.ylabel("Frequency"); plt.show()

# %% [markdown]
# **Note.** For a **nominal** variable, the mean and median are meaningless; the **mode** and the frequency/percent table are the appropriate summaries.

# %% [markdown]
# ---
# # Part B — Practice Exercises — Solutions
#
# ## Practice 1 — Test Scores in Three Subjects
# This mirrors the lecture slide *Standard Deviation Interpretation*. Scores for 300 students are generated below (do **not** change the seed).
#
# **Tasks**
# 1. Compute mean, median, SD, CV, IQR, skewness and kurtosis (normal = 3) for each subject in **one table**.
# 2. Draw side-by-side boxplots and histograms.
# 3. List the outliers in each subject using the $1.5\times\text{IQR}$ rule.
# 4. Answer: *What does the SD tell you about students' performance in each test?*

# %%
rng = np.random.default_rng(2026)
scores = pd.DataFrame({
    "English": rng.normal(100, 7.5, 300).round(1),
    "Math":    rng.normal(100, 15.3, 300).round(1),
    "Physics": rng.normal(100, 30.8, 300).round(1),
})
scores.head()

# %%

# SOLUTION 1: summary table, one row per subject
q = scores.quantile([0.25, 0.75])
summary_scores = pd.DataFrame({
    "Mean":                  scores.mean(),
    "Median":                scores.median(),
    "SD":                    scores.std(),
    "CV (%)":                scores.std() / scores.mean() * 100,
    "IQR":                   q.loc[0.75] - q.loc[0.25],
    "Skewness":              scores.skew(),
    "Kurtosis (normal = 3)": scores.apply(stats.kurtosis, fisher=False),
})
summary_scores.round(2)

# %%

# SOLUTION 2: boxplots and overlaid histograms
scores.plot.box(title="Test scores by subject"); plt.ylabel("Score"); plt.show()

scores.plot.hist(bins=40, alpha=0.5, title="Distribution of test scores")
plt.xlabel("Score"); plt.show()

# %%

# SOLUTION 3: 1.5×IQR outliers per subject
for col in scores.columns:
    Q1, Q3 = scores[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    out = scores[col][(scores[col] < lo) | (scores[col] > hi)]
    print(f"{col:8s} fences [{lo:6.2f}, {hi:6.2f}]  outliers ({len(out)}): {sorted(out.tolist())}")

# %% [markdown]
#
# **SOLUTION 4 — Interpretation**
#
# All three tests have almost the same center (mean ≈ median ≈ 100), skewness close to 0 and kurtosis close to 3, so each is roughly **symmetric and normal-shaped**. They differ only in **spread**: SD ≈ 8 (English), ≈ 16 (Math) and ≈ 31 (Physics), and the CV and IQR grow in the same proportion.
# - **English:** scores are tightly clustered; most students perform similarly, so the test separates students only weakly.
# - **Math:** moderate spread; a typical student scores within about ±16 points of the average.
# - **Physics:** very wide spread; performance is highly uneven, with some students far ahead and others far behind. An "average of 100" describes the Physics class much less well than the English class.
#
# The handful of 1.5×IQR outliers in each subject (2–5 values out of 300) is about what a normal distribution produces by chance, so they are **not** data errors and should not be removed.
#

# %% [markdown]
# ---
# ## Practice 2 — Quality Control on a Production Line
# An inspector recorded each defective unit's **defect type** (nominal) and the customer's **rating** of the replacement service (ordinal: *Bad < Good < Excellent*).
#
# **Tasks**
# 1. State the level of measurement of each variable.
# 2. Build a frequency table for `defect` with Frequency, Percent and Cumulative Percent, **sorted from most to least frequent** (a Pareto table).
# 3. Find the mode of each variable and draw a bar chart of `defect`.
# 4. Convert `rating` to an **ordered categorical** and find its median category.
# 5. Answer: *Why can we report a median for `rating` but not for `defect`?*

# %%
rng = np.random.default_rng(15)
qc = pd.DataFrame({
    "defect": rng.choice(["Scratch", "Crack", "Misalignment", "Porosity", "Dent"],
                         size=120, p=[0.35, 0.10, 0.25, 0.20, 0.10]),
    "rating": rng.choice(["Bad", "Good", "Excellent"], size=120, p=[0.2, 0.5, 0.3]),
})
qc.head()

# %% [markdown]
#
# **SOLUTION 1 — Level of measurement**
# - `defect`: **nominal** — categories with names only, no natural order.
# - `rating`: **ordinal** — ordered categories (*Bad < Good < Excellent*), but the gaps between them are not measurable.
#

# %%

# SOLUTION 2: Pareto table (value_counts already sorts most → least frequent)
pareto = qc["defect"].value_counts().to_frame("Frequency")
pareto["Percent"]            = qc["defect"].value_counts(normalize=True) * 100
pareto["Cumulative Percent"] = pareto["Percent"].cumsum()
pareto.round(1)

# %%

# SOLUTION 3: modes and bar chart
print("Mode of defect:", qc["defect"].mode()[0])
print("Mode of rating:", qc["rating"].mode()[0])

qc["defect"].value_counts().plot.bar(rot=0, title="Defect types (n = 120)")
plt.ylabel("Frequency"); plt.show()

# %%

# SOLUTION 4: ordered categorical and its median
qc["rating"] = pd.Categorical(qc["rating"], categories=["Bad", "Good", "Excellent"], ordered=True)

median_code = int(np.median(qc["rating"].cat.codes))
print("Counts:\n", qc["rating"].value_counts(sort=False), sep="")
print("Median rating:", qc["rating"].cat.categories[median_code])
print("Min / Max:", qc["rating"].min(), "/", qc["rating"].max())   # valid because the category is ordered

# %% [markdown]
#
# **SOLUTION 5 — Interpretation**
#
# The median is the value in the **middle after sorting**, so it needs an order. `rating` has a natural order (*Bad < Good < Excellent*), so we can sort the 120 ratings and read off the middle one: **Good**. `defect` has no order (is a "Crack" more or less than a "Dent"?), so sorting is arbitrary and any "median" would change with the alphabet. For nominal data the only meaningful center is the **mode** (here, *Scratch*).
#
# **Pareto reading:** Scratch, Misalignment and Porosity together account for ≈ 82% of all defects, so QC improvement effort should target those three first.
#
