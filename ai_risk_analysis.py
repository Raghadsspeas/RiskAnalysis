import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# تحميل البيانات الافتراضية
file_path = "legal_risk_database.xlsx"
xls = pd.ExcelFile(file_path)

# قراءة بيانات المخاطر القانونية
legal_risks = pd.read_excel(xls, "Legal_Risks")
compliance_status = pd.read_excel(xls, "Compliance_Status")

# تحويل البيانات الفئوية إلى أرقام باستخدام التشفير الرقمي
le = LabelEncoder()
legal_risks["Risk_Level"] = le.fit_transform(legal_risks["Risk_Level"])
legal_risks["Department"] = le.fit_transform(legal_risks["Department"])

# تجهيز بيانات التدريب
X = legal_risks[["Department"]]
y = legal_risks["Risk_Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب نموذج الذكاء الاصطناعي
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# اختبار النموذج
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))

# تحليل الامتثال القانوني
compliance_status["Compliance_Risk"] = np.where(compliance_status["Compliance_Score"] < 70, "High Risk", "Low Risk")
print(compliance_status[["Department", "Compliance_Score", "Compliance_Risk"]])
