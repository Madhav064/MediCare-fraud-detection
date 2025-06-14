import pandas as pd

print("🔄 Loading test data...")

# Load test datasets
test_ip = pd.read_csv("Test_Inpatientdata-1542969243754.csv")
test_op = pd.read_csv("Test_Outpatientdata-1542969243754.csv")
test_bene = pd.read_csv("Test_Beneficiarydata-1542969243754.csv")
# Convert dates
test_ip['AdmissionDt'] = pd.to_datetime(test_ip['AdmissionDt'], errors='coerce')
test_ip['DischargeDt'] = pd.to_datetime(test_ip['DischargeDt'], errors='coerce')
test_ip['Admit_Duration'] = (test_ip['DischargeDt'] - test_ip['AdmissionDt']).dt.days

test_op['ClaimStartDt'] = pd.to_datetime(test_op['ClaimStartDt'], errors='coerce')
test_op['ClaimEndDt'] = pd.to_datetime(test_op['ClaimEndDt'], errors='coerce')
test_op['Claim_Duration'] = (test_op['ClaimEndDt'] - test_op['ClaimStartDt']).dt.days

# Combine inpatient & outpatient
claims = pd.concat([
    test_ip[['Provider', 'BeneID', 'Admit_Duration']],
    test_op[['Provider', 'BeneID', 'Claim_Duration']]
], ignore_index=True)

# Merge with beneficiary for State info
claims = claims.merge(test_bene[['BeneID', 'State']], on='BeneID', how='left')

# Group by Provider and State
agg = claims.groupby(['Provider', 'State']).agg({
    'Admit_Duration': ['mean', 'count', 'sum'],
    'Claim_Duration': ['mean', 'count', 'sum']
}).reset_index()

# Rename columns
agg.columns = ['Provider', 'State',
               'IP_mean_admit', 'IP_claim_count', 'IP_total_admit',
               'OP_mean_claim', 'OP_claim_count', 'OP_total_claim']

# Fill NAs
agg.fillna(0, inplace=True)

# Save
agg.to_csv("prediction_input.csv", index=False)
print("✅ Saved prediction_input.csv with State column.")
