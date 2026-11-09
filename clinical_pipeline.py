import logging
import numpy as np
import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_synthetic_clinical_data(n_samples=500, random_seed=42):
    """Generates a realistic synthetic clinical dataset with noise and missing values."""
    np.random.seed(random_seed)

    patient_ids = [f"PAT_{i:04d}" for i in range(1, n_samples + 1)]
    age = np.random.normal(loc=58, scale=12, size=n_samples).astype(int)
    sex = np.random.choice(["M", "F"], size=n_samples, p=[0.52, 0.48])
    bmi = np.random.normal(loc=26.5, scale=4.5, size=n_samples)
    sys_bp = np.random.normal(loc=130, scale=15, size=n_samples)
    biomarker_alpha = np.random.exponential(scale=15.0, size=n_samples)
    treatment_group = np.random.choice(["Control", "Treatment_A", "Treatment_B"], size=n_samples)
    readmission_30d = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "sys_bp": sys_bp,
        "biomarker_alpha": biomarker_alpha,
        "treatment_group": treatment_group,
        "readmission_30d": readmission_30d
    })

    # Introduce random missingness (MCAR)
    mask_bmi = np.random.rand(n_samples) < 0.08
    mask_bp = np.random.rand(n_samples) < 0.05
    mask_bio = np.random.rand(n_samples) < 0.12

    df.loc[mask_bmi, "bmi"] = np.nan
    df.loc[mask_bp, "sys_bp"] = np.nan
    df.loc[mask_bio, "biomarker_alpha"] = np.nan

    # Introduce explicit outliers
    df.loc[12, "sys_bp"] = 310.0  # Unrealistic measurement error
    df.loc[45, "biomarker_alpha"] = 350.0  # Extreme biological outlier

    return df

class ClinicalDataCleaner:
    def __init__(self, df):
        self.raw_data = df.copy()
        self.data = df.copy()
        self.qc_summary = {}

    def inspect_missingness(self):
        """Calculates missing value rates across columns."""
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        missing_df = pd.DataFrame({"Missing_Count": missing, "Missing_Percentage": missing_pct})
        self.qc_summary["missingness"] = missing_df
        logging.info("Missingness audit completed.")
        return missing_df

    def handle_missing_values(self):
        """Imputes missing numerical values using median grouped by sex/age group."""
        # Create age brackets for stratified imputation
        self.data["age_group"] = pd.cut(self.data["age"], bins=[0, 45, 65, 100], labels=["<45", "45-65", ">65"])

        num_cols = ["bmi", "sys_bp", "biomarker_alpha"]
        for col in num_cols:
            self.data[col] = self.data.groupby(["sex", "age_group"], observed=False)[col].transform(
                lambda x: x.fillna(x.median())
            )
        logging.info("Stratified median imputation executed.")

    def detect_and_cap_outliers(self, cols, iqr_multiplier=2.5):
        """Detects outliers using IQR and caps them (Winsorization) to preserve sample size."""
        for col in cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (iqr_multiplier * IQR)
            upper_bound = Q3 + (iqr_multiplier * IQR)

            outliers_cnt = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
            logging.info(f"Feature '{col}': {outliers_cnt} outliers detected (IQR threshold: {iqr_multiplier}). Capping values.")

            self.data[col] = np.clip(self.data[col], lower_bound, upper_bound)

    def normalize_features(self, cols):
        """Applies Log1p transformation to skewed biomarker distributions and standard scaling."""
        # Log transformation for right-skewed biomarker data
        self.data["biomarker_alpha_log"] = np.log1p(self.data["biomarker_alpha"])

        # Z-score normalization
        for col in cols + ["biomarker_alpha_log"]:
            mean_val = self.data[col].mean()
            std_val = self.data[col].std()
            self.data[f"{col}_zscore"] = (self.data[col] - mean_val) / std_val

        logging.info("Normalization and Z-score standardization complete.")

    def run_pipeline(self):
        """Executes full quality control pipeline."""
        logging.info("Starting Clinical Data Preprocessing Pipeline...")
        self.inspect_missingness()
        self.handle_missing_values()
        self.detect_and_cap_outliers(cols=["sys_bp", "biomarker_alpha"])
        self.normalize_features(cols=["bmi", "sys_bp"])

        # Drop temporary helper column
        self.data.drop(columns=["age_group"], inplace=True)
        logging.info("Pipeline execution finished successfully.")
        return self.data

if __name__ == "__main__":
    # Generate raw synthetic data
    raw_df = generate_synthetic_clinical_data()
    raw_df.to_csv("raw_clinical_data_sample.csv", index=False)

    # Run Cleaner Pipeline
    cleaner = ClinicalDataCleaner(raw_df)
    processed_df = cleaner.run_pipeline()
    processed_df.to_csv("cleaned_clinical_data.csv", index=False)

    print("\nProcessed Data Sample (Head):")
    print(processed_df.head())
