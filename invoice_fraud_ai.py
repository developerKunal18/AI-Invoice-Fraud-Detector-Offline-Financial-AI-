import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Sample invoice dataset
data = {
    "amount": [500, 2000, 300, 10000, 250, 9000],
    "tax": [50, 200, 30, 1200, 20, 1100],
    "vendor_score": [8, 9, 7, 2, 8, 3],
    "fraud": [0, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["amount", "tax", "vendor_score"]]
y = df["fraud"]

model = RandomForestClassifier()
model.fit(X, y)

print("🧾 Invoice Fraud Detector \n")

amt = float(input("Invoice amount: "))
tax = float(input("Tax amount: "))
vendor = float(input("Vendor trust score (1-10): "))

pred = model.predict([[amt, tax, vendor]])[0]

if pred == 1:
    print("\n🚨 FRAUDULENT invoice detected")
else:
    print("\n✅ Legitimate invoice")
