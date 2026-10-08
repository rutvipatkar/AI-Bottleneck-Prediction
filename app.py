import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -----------------------------
# Load ML training data
# -----------------------------
data = pd.read_csv("hospital_data.csv")
waiting_data = pd.read_csv("waiting_time_data.csv")

X = data[["patients", "beds", "staff", "waiting_time", "emergency_cases"]]
y = data["bottleneck"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Train waiting-time model
waiting_X = waiting_data[
    ["patients", "beds", "staff", "emergency_cases"]
]

waiting_y = waiting_data["waiting_time"]

waiting_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

waiting_model.fit(waiting_X, waiting_y)

# -----------------------------
# Load patient records
# -----------------------------
patients_file = "patients.csv"

patients = pd.read_csv(patients_file)

# -----------------------------
# Hospital information
# -----------------------------

DEPARTMENT_RESOURCES = {
    "ICU": {
        "beds": 10,
        "staff": 8
    },
    "Emergency": {
        "beds": 20,
        "staff": 10
    },
    "OPD": {
        "beds": 20,
        "staff": 12
    }
}

# -----------------------------
# Dashboard
# -----------------------------
st.title("🏥 Hospital Bottleneck Prediction System")

st.header("Patient Registration")

# Create a new form version when needed
if "form_version" not in st.session_state:
    st.session_state.form_version = 0

# -----------------------------
# Patient Registration Form
# -----------------------------

with st.form(key=f"patient_form_{st.session_state.form_version}"):

    patient_id = st.text_input("Patient ID")

    name = st.text_input("Patient Name")

    department = st.selectbox(
        "Department",
        ["ICU", "Emergency", "OPD"]
    )

    emergency = st.selectbox(
        "Emergency Case?",
        ["Yes", "No"]
    )

    submitted = st.form_submit_button("➕ Admit Patient")


# -----------------------------
# Admit patient
# -----------------------------

if submitted:

    if patient_id == "" or name == "":
        st.error("Please enter Patient ID and Patient Name.")

    elif patient_id in patients["patient_id"].astype(str).values:
        st.error("⚠️ This Patient ID already exists.")

    else:

        new_patient = pd.DataFrame({
            "patient_id": [patient_id],
            "name": [name],
            "department": [department],
            "emergency": [emergency],
            "status": ["Admitted"]
        })

        patients = pd.concat(
            [patients, new_patient],
            ignore_index=True
        )

        patients.to_csv(patients_file, index=False)

        st.success("✅ Patient admitted successfully!")

        # Create a fresh form
        st.session_state.form_version += 1

        st.rerun()
# -----------------------------
# Current hospital statistics
# -----------------------------

admitted_patients = patients[
    patients["status"] == "Admitted"
]

# -----------------------------
# Display statistics
# -----------------------------

st.header("🏥 Current Hospital Status")

for dept in ["ICU", "Emergency", "OPD"]:

    dept_patients = admitted_patients[
        admitted_patients["department"] == dept
    ]

    patient_count = len(dept_patients)

    total_beds = DEPARTMENT_RESOURCES[dept]["beds"]
    total_staff = DEPARTMENT_RESOURCES[dept]["staff"]

    available_beds = total_beds - patient_count

    emergency_cases = len(
        dept_patients[
            dept_patients["emergency"] == "Yes"
        ]
    )

    st.subheader(f"🏥 {dept}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Patients", patient_count)
    col2.metric("Total Beds", total_beds)
    col3.metric("Available Beds", available_beds)
    col4.metric("Emergency Cases", emergency_cases)


st.write("### Current Patients")

st.dataframe(admitted_patients)

# -----------------------------
# Discharge Patient
# -----------------------------

st.subheader("🚪 Discharge Patient")

if len(admitted_patients) > 0:

    patient_to_discharge = st.selectbox(
        "Select Patient",
        admitted_patients["patient_id"].astype(str).tolist()
    )

    if st.button("🚪 Discharge Patient"):

        patients.loc[
            patients["patient_id"].astype(str) == patient_to_discharge,
            "status"
        ] = "Discharged"

        patients.to_csv(
            patients_file,
            index=False
        )

        st.success(
            f"Patient {patient_to_discharge} discharged successfully!"
        )

        st.rerun()

else:

    st.info("No admitted patients to discharge.")

# -----------------------------
# ML Predictions
# -----------------------------

st.header("🤖 AI Predictions")

for dept in ["ICU", "Emergency", "OPD"]:

    dept_patients = admitted_patients[
        admitted_patients["department"] == dept
    ]

    patient_count = len(dept_patients)

    total_beds = DEPARTMENT_RESOURCES[dept]["beds"]
    total_staff = DEPARTMENT_RESOURCES[dept]["staff"]

    available_beds = total_beds - patient_count

    emergency_cases = len(
        dept_patients[
            dept_patients["emergency"] == "Yes"
        ]
    )

    # Only predict when department has patients
    if patient_count > 0:

        # -----------------------------
        # Predict waiting time
        # -----------------------------

        waiting_input = pd.DataFrame(
            [[
                patient_count,
                available_beds,
                total_staff,
                emergency_cases
            ]],
            columns=[
                "patients",
                "beds",
                "staff",
                "emergency_cases"
            ]
        )

        predicted_waiting_time = waiting_model.predict(
            waiting_input
        )[0]

        # -----------------------------
        # Predict bottleneck
        # -----------------------------

        bottleneck_input = pd.DataFrame(
            [[
                patient_count,
                available_beds,
                total_staff,
                predicted_waiting_time,
                emergency_cases
            ]],
            columns=[
                "patients",
                "beds",
                "staff",
                "waiting_time",
                "emergency_cases"
            ]
        )

        prediction = model.predict(
            bottleneck_input
        )

        probability = model.predict_proba(
            bottleneck_input
        )

        risk = probability[0][1] * 100

        # -----------------------------
        # Risk level
        # -----------------------------

        if risk >= 70:
            risk_level = "HIGH"

        elif risk >= 40:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        # -----------------------------
        # Display prediction
        # -----------------------------

        st.subheader(f"🏥 {dept}")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Estimated Waiting Time",
            f"{round(predicted_waiting_time, 1)} min"
        )

        col2.metric(
            "Bottleneck Probability",
            f"{round(risk, 1)}%"
        )

        col3.metric(
            "Risk Level",
            risk_level
        )

        if prediction[0] == 1:

            st.error(
                "⚠️ Bottleneck Predicted"
            )

        else:

            st.success(
                "✅ No Bottleneck Predicted"
            )

        # -----------------------------
        # Recommendation
        # -----------------------------

        if risk_level == "HIGH":

            st.warning(
                "Recommendation: Increase staff and available beds immediately."
            )

        elif risk_level == "MEDIUM":

            st.info(
                "Recommendation: Monitor the department and prepare additional resources."
            )

        else:

            st.success(
                "Recommendation: Current resources are sufficient."
            )