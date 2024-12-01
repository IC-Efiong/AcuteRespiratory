import streamlit as st 
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fcmeans import FCM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your trained model
model = pickle.load(open('App/trained_model.pkl', 'rb'))
# Load pre-trained FCM model
fcm_model = pickle.load(open('App/fcm_model.pkl', 'rb'))


diagnosis_mapping = {
    0: "ACUTE ASTHMA",
    1: "ACUTE URTI",
    2: "AIRWAY OBSTRUCTION",
    3: "ASTHMA",
    4: "BPN",
    5: "BRONCHOPNEUMONIA",
    6: "CARDIAC CASE",
    7: "CHRONIC KIDNEY DISEASE (CKD)",
    8: "COUGH",
    9: "DEHYDRATION",
    10: "DELAYED SPEECH",
    11: "DIARRHEA",
    12: "E-FEVER",
    13: "ECZEMA",
    14: "FERNATAL ASPYHXIA",
    15: "FUMCULOSIS",
    16: "GET SEPSIS",
    17: "HEAMORRHAGIC DISEASE",
    18: "HEUGRAGHIC TONGUE",
    19: "HYDROCELL",
    20: "HYPERPYREXIA",
    21: "HYPERTENSION",
    22: "HYPERTENSIVE HEART DISEASE",
    23: "INGUINAL HERNIA",
    24: "MACROSOMU BABY",
    25: "MALARIA",
    26: "MALARIA WITH PEPSIS",
    27: "MALARIAL",
    28: "NNS",
    29: "OBSTRUCTIVE ADENOID",
    30: "OBSTRUCTIVE AXIS",
    31: "ORAL THRUSH",
    32: "PELVIC INFLAMMATORY DISEASE",
    33: "PEPTIC ULCER D2",
    34: "PERINATAL ASPHYXIA",
    35: "PHERYGITIS",
    36: "PID",
    37: "PNEUMONIA",
    38: "POORLY TREATED MALARIA",
    39: "PYELONEPHRITIS",
    40: "RIGHT VENTRICULAR DYSPLASIA (RVD)",
    41: "RVD-RETRO VIRUS DISEASE",
    42: "SALMONELLOSIS",
    43: "SEPSIS",
    44: "SEVERE ANEMIA",
    45: "SEVERE MALARIA",
    46: "TONSILLITIS",
    47: "UNCOMPLICATED MALARIA",
    48: "UNTREATED MALARIA",
    49: "URTI",
    50: "UTI"
}

# Function to predict the cluster
def predict_cluster(input_data, fcm):
    try:
        # Fit a scaler on the input data (for the web app's new data)
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(np.array(input_data).reshape(1, -1))

        # Predict cluster membership using the FCM model
        membership = fcm.predict(input_scaled)
        
        # Find the predicted cluster (the one with the highest membership value)
        predicted_cluster = np.argmax(membership)
        return predicted_cluster, membership
    except Exception as e:
        return None, str(e)

# App title
st.title("Respiratory Prediction System")

# Sidebar for navigation
menu = st.sidebar.selectbox("Menu", ["Prediction App", "Fuzzy C-mean Cluster"])

if menu == "Prediction App":
    st.header("Prediction App")

    # Input columns for user features
    col1, col2, col3 = st.columns(3)
    
    # Numeric inputs for Age and Duration
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)
    with col2:
        duration = st.number_input("Duration (in days)", min_value=0, max_value=365, value=10, step=1)

    # Dropdown for Gender
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_mapped = 1 if gender == "Male" else 0

    # Other feature inputs using dropdowns
    features = [
        "cough", "night_fever", "catarrh", "cold", "tenderness", "headache", 
        "black_stool", "jerking_movement_of_the_hands", "unable_to_sit_unsupported",
        "throat_fever", "fast_breathing", "body_rashes", "cough_sills", 
        "frequent_stooling", "chest_pain", "frequent_vomiting", 
        "sleep_disturbance_at_night", "nasal_stuffiness", "excessive_sweating", 
        "chronic_cough", "profuse_nasal_discharge", "difficulty_in_breathing", 
        "itching_ear", "frequent_urination", "swelling_of_pubic_area_right_sided", 
        "generalized_body_rashes", "painful_swallowing", "passage_of_freight", 
        "poor_vision", "frequent_micturition", "watery_of_the_tongue", 
        "poor_appetite", "noisy_breathing", "pepsis_swells_up_since_birth", 
        "body_weakness", "flame_pain", "swollen_leg", "abdominal_pain", 
        "excessive_salivation", "crawling_sensation", "weight_loss", "g.b_pain", 
        "stomach_ache", "tysing_of_the_right_ear", "refusal_of_food", 
        "waist_pain", "snoring", "knee_pain", "dizziness", "scrotal_swelling", 
        "vaginal_discharge", "vaginal_itching", "stooling", "running_nose", 
        "club_pain", "bleeding_from_the_nose", "unable_to_speak", 
        "mouth_breathing", "sleep_apneas"
    ]

    # Initialize a list for storing inputs
    inputs = [gender_mapped, age, duration]

    # Create dropdowns for remaining features
    for i, feature in enumerate(features):
        with [col1, col2, col3][i % 3]:  # Distribute inputs across columns
            value = st.selectbox(f"{feature.replace('_', ' ').capitalize()}", ["Yes", "No"], index=1)
            inputs.append(1 if value == "Yes" else 0)

    # Convert inputs to numpy array for prediction
    input_data = np.asarray(inputs).reshape(1, -1)

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        diagnosis = diagnosis_mapping.get(prediction, "Unknown Diagnosis")
        st.success(f"The predicted condition is: {diagnosis}")

elif menu == "Fuzzy C-mean Cluster":
    st.header("Fuzzy C-mean Cluster")
    
    # Input new data for cluster prediction
    st.subheader("Input New Data Point for Cluster Prediction")
    
    # Create three columns for inputs
    col1, col2, col3 = st.columns(3)
    
    features = [
        "cough", "night_fever", "catarrh", "cold", "tenderness", "headache", 
        "black_stool", "jerking_movement_of_the_hands", "unable_to_sit_unsupported",
        "throat_fever", "fast_breathing", "body_rashes", "cough_sills", 
        "frequent_stooling", "chest_pain", "frequent_vomiting", 
        "sleep_disturbance_at_night", "nasal_stuffiness", "excessive_sweating", 
        "chronic_cough", "profuse_nasal_discharge", "difficulty_in_breathing", 
        "itching_ear", "frequent_urination", "swelling_of_pubic_area_right_sided", 
        "generalized_body_rashes", "painful_swallowing", "passage_of_freight", 
        "poor_vision", "frequent_micturition", "watery_of_the_tongue", 
        "poor_appetite", "noisy_breathing", "pepsis_swells_up_since_birth", 
        "body_weakness", "flame_pain", "swollen_leg", "abdominal_pain", 
        "excessive_salivation", "crawling_sensation", "weight_loss", "g.b_pain", 
        "stomach_ache", "tysing_of_the_right_ear", "refusal_of_food", 
        "waist_pain", "snoring", "knee_pain", "dizziness", "scrotal_swelling", 
        "vaginal_discharge", "vaginal_itching", "stooling", "running_nose", 
        "club_pain", "bleeding_from_the_nose", "unable_to_speak", 
        "mouth_breathing", "sleep_apneas"
    ]

    # Define input fields for the new data point, matching the features
    new_data_point = []
    new_data_point.append(st.number_input("Age", min_value=0, max_value=120, value=25, step=1))
    new_data_point.append(st.number_input("Duration (in days)", min_value=0, max_value=365, value=10, step=1))
    gender = st.selectbox("Gender", ["Male", "Female"])
    new_data_point.append(1 if gender == "Male" else 0)

    # Collect other features for the new data point
    for feature in features:
        value = st.selectbox(f"{feature.replace('_', ' ').capitalize()}", ["Yes", "No"], index=1)
        new_data_point.append(1 if value == "Yes" else 0)
    
    # Button to predict the cluster
    if st.button("Predict Cluster"):
        cluster, membership = predict_cluster(new_data_point, fcm_model)
        if cluster is not None:
        # Check if membership is a 1D array or a single value
            if isinstance(membership, np.ndarray) and membership.ndim == 2:
            # Flatten the array to make sure it's iterable
                membership_percentages = [f"{score * 100:.2f}%" for score in membership[0]]
            else:
             # If it's a single value (cluster assignment), no need to format percentages
                membership_percentages = [f"Cluster {cluster + 1}: 100.00%"]

            # Create a more readable output for the cluster and membership scores
            st.success(f"\nThe new data point belongs to Cluster {cluster + 1}.")
            st.write("\nMembership Scores:")
    
            # Print each membership score with its corresponding cluster
            for i, score in enumerate(membership_percentages):
                st.write(f"{score}")
        else:
            st.error(f"Error in prediction: {membership}")