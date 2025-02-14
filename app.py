import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the data from Google Drive
file_id = '1-uHj-17MQA6_DuNT9DKKU17opybldZpJ'  # Replace with your actual file ID
data_url = f'https://drive.google.com/uc?id={file_id}'
data2 = pd.read_csv(data_url)

# Drop the specified columns
columns_to_drop = ['Complaint number', 'Confirmed complaint', 'Keywords', 'Others involved']
data1 = data2.drop(columns=columns_to_drop)

# Remove rows with null values
data1 = data1.dropna().reset_index(drop=True)

data = data1.iloc[:50000]

# Clean the 'Complaint filed against' column by converting all text to uppercase
data['Complaint filed against'] = data['Complaint filed against'].str.upper()

# Clean the 'Reason complaint filed' column by removing anything after the first semicolon
data['Reason complaint filed'] = data['Reason complaint filed'].apply(lambda x: x.split(';')[0] if pd.notnull(x) else x)

# Convert date columns to datetime with the correct format
data['Received date'] = pd.to_datetime(data['Received date'], format='%m/%d/%Y')
data['Closed date'] = pd.to_datetime(data['Closed date'], format='%m/%d/%Y')

# Create a new feature for the difference between received and closed dates
data['Days to Close'] = (data['Closed date'] - data['Received date']).dt.days

# Extract month from 'Received date' for analysis
data['Month'] = data['Received date'].dt.month

# Display the title of the app
st.title('The Claim Game')
st.write('### Sample data from Original File:')
st.write(data2.head(20))

st.write('### Shape of Data before data cleaning:')
st.write(data2.shape)

st.write('### Shape of data after data cleaning:')
st.write(data.shape)
# Display the basic stats
#st.write('Basic Statistics:')
#st.write(data.describe())

# Draw the first two graphs side by side
#col1, col2 = st.columns(2)

#with col1:
# Draw a graph for complaints filed against a specific column (assuming 'Complaint filed against' column exists)
if 'Complaint filed against' in data.columns:
        st.write('### Agencies with the Most Complaints Filed Against Them:')
        top_10_complaints = data['Complaint filed against'].value_counts().nlargest(10).index
        filtered_data = data[data['Complaint filed against'].isin(top_10_complaints)]
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.countplot(y='Complaint filed against', data=filtered_data, ax=ax, order=top_10_complaints, palette='viridis')
        ax.set_title('Top 10 Most Complained-About Insurance Agencies', fontsize=20)
        ax.set_xlabel('Number of Complaints', fontsize=18)
        ax.set_ylabel('Complaint Filed Against', fontsize=18)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 1, p.get_y() + p.get_height() / 2, f'{int(width)}', ha='center', va='center', fontsize=16)
        st.pyplot(fig)
else:
        st.write('The column "Complaint filed against" does not exist in the dataset.')

#with col2:
# Draw a pie chart for complaints filed by a specific column (assuming 'Complaint filed by' column exists)
if 'Complaint filed by' in data.columns:
        st.write('### Pie chart representing Complaints Filed By:')
        top_5_complaints_by = data['Complaint filed by'].value_counts().nlargest(5)
        
        def func(pct, allvals):
            absolute = int(pct/100.*sum(allvals))
            return f"{pct:.1f}%\n({absolute:d})"
        
        fig, ax = plt.subplots(figsize=(3, 3))  # Reduced the size of the pie chart
        wedges, texts, autotexts = ax.pie(top_5_complaints_by, labels=top_5_complaints_by.index, autopct=lambda pct: func(pct, top_5_complaints_by), startangle=90, colors=sns.color_palette('viridis', len(top_5_complaints_by)))
        ax.set_title('Top 5 Categories of Complaints Filed By', fontsize=6)  # Reduced the font size of the title
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        for text in texts:
            text.set_fontsize(6)  # Reduce the font size of the labels around the pie chart
        for autotext in autotexts:
            autotext.set_fontsize(5)  # Reduce the font size of the numbers inside the pie chart
        ax.legend(wedges, top_5_complaints_by.index, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        st.pyplot(fig)
else:
        st.write('The column "Complaint filed by" does not exist in the dataset.')

# Draw a grouped bar chart for the column 'Reason complaint filed' (assuming it exists)
if 'Complaint type' in data.columns:
    st.write('### Reasons for Complaints Filed:')
    top_10_reasons = data['Complaint type'].value_counts().nlargest(7).index
    filtered_data = data[data['Complaint type'].isin(top_10_reasons)]
    reason_complaint_pivot = filtered_data.pivot_table(index='Complaint type', columns='Complaint filed against', aggfunc='size', fill_value=0)
    reason_complaint_pivot = reason_complaint_pivot[top_10_complaints]  # Ensure the columns are in the same order as the top 10 complaints
    reason_complaint_pivot = reason_complaint_pivot.loc[top_10_reasons]  # Ensure the index is in the same order as the top 10 reasons
    fig, ax = plt.subplots(figsize=(14, 11))
    reason_complaint_pivot.plot(kind='bar', stacked=True, ax=ax, color=sns.color_palette('viridis', len(top_10_complaints)))
    ax.set_title('Grouped Bar Chart of Reasons for Complaints Filed', fontsize=16)
    ax.set_xlabel('Reason Complaint Filed', fontsize=18)  # Increased font size
    ax.set_ylabel('Number of Complaints', fontsize=18)  # Increased font size
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.legend(title='Complaint Filed Against', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
else:
    st.write('The column "Reason complaint filed" does not exist in the dataset.')

# Analyze the relationship between Complaint type and Coverage level
if 'Complaint type' in data.columns and 'Coverage level' in data.columns:
    st.write('### Relationship between Complaint Type and Coverage Level:')
    top_10_coverage_levels = data['Coverage level'].value_counts().nlargest(10).index
    filtered_data = data[data['Coverage level'].isin(top_10_coverage_levels)]
    complaint_coverage_pivot = filtered_data.pivot_table(index='Complaint type', columns='Coverage level', aggfunc='size', fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(complaint_coverage_pivot, annot=True, fmt='d', cmap='viridis', ax=ax)
    ax.set_title('Heatmap of Complaint Type vs Coverage Level', fontsize=16)
    ax.set_xlabel('Coverage Level', fontsize=14)
    ax.set_ylabel('Complaint Type', fontsize=14)
    st.pyplot(fig)
else:
    st.write('The columns "Complaint type" and/or "Coverage level" do not exist in the dataset.')

# Draw a bar chart for complaints received by month
if 'Received date' in data.columns:
    st.write('### Complaints Received by Month:')
    data['Received date'] = pd.to_datetime(data['Received date'], format='%m/%d/%Y')
    data['Month'] = data['Received date'].dt.month
    complaints_by_month = data['Month'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette('husl', 12)  # Use a different color for each bar
    bars = ax.bar(complaints_by_month.index, complaints_by_month.values, color=colors)
    ax.set_title('Complaints Received by Month', fontsize=16)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Number of Complaints', fontsize=14)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    #ax.bar_label(bars, fontsize=12)  # Add count on the bars with specified font size
    st.pyplot(fig)
else:
    st.write('The column "Received date" does not exist in the dataset.')

# Convert date columns to ordinal for model training
data['Received date'] = data['Received date'].map(pd.Timestamp.toordinal)
data['Closed date'] = data['Closed date'].map(pd.Timestamp.toordinal)

# Predictive model
#st.write('### Predictive Model')

# Encode categorical variables
label_encoders = {}
for column in ['Complaint filed against', 'Complaint filed by', 'Reason complaint filed', 'Complaint type', 'Coverage type', 'Coverage level']:
    if column in data.columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Prepare the data for training
X = data[['Complaint filed against', 'Complaint filed by', 'Reason complaint filed', 'Received date', 'Complaint type', 'Coverage type', 'Coverage level']]
y = data['Days to Close']  # Predicting 'Days to Close'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#st.write('### Model Evaluation')
#st.write(f'Mean Absolute Error (MAE): {mae}')
#st.write(f'Mean Squared Error (MSE): {mse}')
#st.write(f'R-squared (R²) Score: {r2}')

# Input form for prediction
st.write('### Model to Predict the Days to Close a Claim:')
complaint_filed_against = st.selectbox('Complaint filed against', label_encoders['Complaint filed against'].classes_)
complaint_filed_by = st.selectbox('Complaint filed by', label_encoders['Complaint filed by'].classes_)
reason_complaint_filed = st.selectbox('Reason complaint filed', label_encoders['Reason complaint filed'].classes_)
received_date = st.date_input('Received date')
complaint_type = st.selectbox('Complaint type', label_encoders['Complaint type'].classes_)
coverage_type = st.selectbox('Coverage type', label_encoders['Coverage type'].classes_)
coverage_level = st.selectbox('Coverage level', label_encoders['Coverage level'].classes_)

# Encode the input values
encoded_complaint_filed_against = label_encoders['Complaint filed against'].transform([complaint_filed_against])[0]
encoded_complaint_filed_by = label_encoders['Complaint filed by'].transform([complaint_filed_by])[0]
encoded_reason_complaint_filed = label_encoders['Reason complaint filed'].transform([reason_complaint_filed])[0]
encoded_complaint_type = label_encoders['Complaint type'].transform([complaint_type])[0]
encoded_coverage_type = label_encoders['Coverage type'].transform([coverage_type])[0]
encoded_coverage_level = label_encoders['Coverage level'].transform([coverage_level])[0]

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'Complaint filed against': [encoded_complaint_filed_against],
    'Complaint filed by': [encoded_complaint_filed_by],
    'Reason complaint filed': [encoded_reason_complaint_filed],
    'Received date': [received_date.toordinal()],
    'Complaint type': [encoded_complaint_type],
    'Coverage type': [encoded_coverage_type],
    'Coverage level': [encoded_coverage_level]
})

# Predict the outcome
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'The predicted days to close is: {int(prediction[0])} days')
