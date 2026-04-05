import streamlit as st
import pandas as pd
import pickle

# Load files
model = pickle.load(open("reg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
train_cols = pickle.load(open("columns.pkl", "rb"))

st.title("🏡 Real Estate Investment Advisor")

# Inputs
price = st.number_input("Price (Lakhs)", 1.0)
size = st.number_input("Size (SqFt)", 100.0)
bhk = st.selectbox("BHK", [1,2,3,4,5])

schools = st.slider("Schools", 0, 10)
hospitals = st.slider("Hospitals", 0, 10)
transport = st.slider("Transport", 0, 10)

year = st.number_input("Year Built", 1990, 2026)

property_type = st.selectbox("Type", ["Apartment","Villa","Independent House"])
furnished = st.selectbox("Furnished", ["Furnished","Semi-Furnished","Unfurnished"])
facing = st.selectbox("Facing", ["North","South","East","West"])
owner = st.selectbox("Owner", ["Owner","Dealer","Builder"])
availability = st.selectbox("Availability", ["Ready","Under Construction"])

# Features
price_sqft = (price * 100000) / size
amenity = schools + hospitals + transport
age = 2026 - year

# Base data
data = {
    "Price_in_Lakhs": price,
    "Size_in_SqFt": size,
    "BHK": bhk,
    "Nearby_Schools": schools,
    "Nearby_Hospitals": hospitals,
    "Public_Transport_Accessibility": transport,
    "Price_per_SqFt": price_sqft,
    "Amenity_Score": amenity,
    "Age_of_Property": age
}

df = pd.DataFrame([data])

# Add categorical (dummy style)
df["Property_Type_" + property_type] = 1
df["Furnished_Status_" + furnished] = 1
df["Facing_" + facing] = 1
df["Owner_Type_" + owner] = 1
df["Availability_Status_" + availability] = 1

# Match training columns
df = df.reindex(columns=train_cols, fill_value=0)

# Scale
df_scaled = scaler.transform(df)

# Predict
if st.button("Predict"):
    pred = model.predict(df_scaled)[0]
    st.success(f"₹ {round(pred,2)} Lakhs (after 5 years)")

    if price_sqft < 5000 and amenity >= 10 and age <= 10:
        st.success("✅ Yes, it is a GOOD investment 👍")
    else:
        st.error("❌ No, it is NOT a good investment")