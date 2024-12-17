import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading .sav files
from PIL import Image
import altair as alt

# Display the logo
image = Image.open("logo.png")
image = image.resize((500, 100))  # Resize image
st.image(image, use_column_width=True)

# Load the model from the .sav file
loaded_model = joblib.load('Logistic Regression.sav')  # replace with your .sav file path

# Title and headers for the app
st.title(':red[OvarianAI]')
st.title(':yellow[Ovarian AI - The Multidisciplinary Project Making a Difference]')
st.header(':orange[Revolutionize PCOS diagnosis with OvarianAI]')
st.subheader('AI-powered solution to detect Polycystic Ovary Syndrome (PCOS) with high precision!')
st.subheader("OvarianAI's machine learning models have been validated on a large dataset to provide accurate results.")

# Age vs Share of Respondents chart
st.subheader(':red[Age VS Share of Respondents]')
df = pd.DataFrame({
    'Age group': ['<19', '20-29', '30-44', '45-59', '60>'],
    'Percentage': [3.8, 16.81, 11.58, 1.44, 0.55]
})

chart = alt.Chart(df).mark_bar(color='#FFA07A').encode(
    x=alt.X('Age group', title='Age group'),
    y=alt.Y('Percentage', title='Percentage'),
    text=alt.Text('Percentage', format='.1f'),
    color=alt.Color('Age group',)
).configure_axis(
    grid=False
).configure_view(
    strokeWidth=0
)

chart = chart.properties(width=alt.Step(40))  # Adjust the bar width
st.altair_chart(chart, use_container_width=True)

# Start the diagnosis section
st.title(":red[Start your diagnosis]")

# Selected features from Mutual Information (MI) rankings
fields = ['PCOS (Y/N)', 'Follicle No. (R)', 'Follicle No. (L)', 
                     'Skin darkening (Y/N)', 'hair growth(Y/N)', 'Weight gain(Y/N)', 
                     'Cycle length(days)', 'AMH(ng/mL)', 'Fast food (Y/N)', 
                     'Cycle(R/I)', 'FSH/LH', 'PRL(ng/mL)', 'Pimples(Y/N)', 
                     'Age (yrs)', 'BMI']

# Collect input for each selected field
follicle_r = st.number_input('Follicle No. (R)', min_value=0, max_value=100, step=1, format="%d")
follicle_l = st.number_input('Follicle No. (L)', min_value=0, max_value=100, step=1, format="%d")
skin_darkening = st.selectbox('Skin darkening (Y/N)', ['No', 'Yes'])
hair_growth = st.selectbox('Hair growth (Y/N)', ['No', 'Yes'])
weight_gain = st.selectbox('Weight gain (Y/N)', ['No', 'Yes'])
cycle_length = st.number_input('Cycle length (days)', min_value=1, max_value=50, step=1, format="%d")
amh = st.number_input('AMH (ng/mL)', min_value=0.0, format="%.2f")
fast_food = st.selectbox('Fast food (Y/N)', ['No', 'Yes'])
cycle_irregular = st.selectbox('Cycle (R/I)', ['2 (Regular)', '4 (Irregular)'])
fsh_lh = st.number_input('FSH/LH', min_value=0.0, format="%.2f")
prl = st.number_input('PRL (ng/mL)', min_value=0.0, format="%.2f")
pimples = st.selectbox('Pimples (Y/N)', ['No', 'Yes'])

# Age, BMI, Weight, and Height Inputs
age = st.number_input('Age (yrs)', min_value=1, max_value=100, step=1, format="%d")
bmi = st.number_input('BMI', min_value=1.0, step=0.1, format="%.1f")


# Convert categorical inputs to binary (Yes=1, No=0)
binary_features = {
    'No': 0,
    'Yes': 1,
    '2 (Regular)': 2,
    '4 (Irregular)': 4
}

skin_darkening = binary_features[skin_darkening]
hair_growth = binary_features[hair_growth]
weight_gain = binary_features[weight_gain]
fast_food = binary_features[fast_food]
pimples = binary_features[pimples]
cycle_irregular = binary_features[cycle_irregular]

# Create an array of all inputs
inputs_arr = np.array([
    follicle_r, follicle_l, skin_darkening, hair_growth, weight_gain, 
    cycle_length, amh, fast_food, cycle_irregular, fsh_lh, prl, pimples,
    age, bmi
]).reshape(1, -1)

# Submit button and prediction
if st.button("Submit"):
    value = loaded_model.predict(inputs_arr)
    if value == 1:
        st.success("Diagnosis completed ✅")
        st.success("You have a probability of Polycystic Ovary Syndrome")
    else:
        st.success("Diagnosis completed ✅")
        st.success("You are not prone to Polycystic Ovary Syndrome")

# Developer credits
st.text("Developed By")
st.text("Surbi")
st.text("B.Tech Computer Science and Technology")
st.text("Department of Software engineering")
st.text("Nepal College of Information Technology")
