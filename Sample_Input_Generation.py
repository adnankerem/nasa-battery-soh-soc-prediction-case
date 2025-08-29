import pandas as pd
import joblib

FEATURE_LIST = joblib.load("feature_list.pkl")

# Use your rich features CSV!
df = pd.read_csv("./data/processed/all_cycles_metadata_core.csv")

# Check which features are missing (strict validation)
missing_cols = set(FEATURE_LIST) - set(df.columns)
if missing_cols:
    print("ERROR: These features are missing from your CSV:", missing_cols)
    # (Optional: exit or adapt code as needed)

# Now sample as before (if all columns exist)
N = 5
sample_df = df[FEATURE_LIST].dropna().sample(N, random_state=42)
sample_df.to_csv("sample_input_ReadyTo_Streamlit_Upload.csv", index=False)
print("Saved: sample_input_ReadyTo_Streamlit_Upload.csv (ready for Streamlit upload)")
