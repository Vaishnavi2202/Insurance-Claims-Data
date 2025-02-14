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
# Clean the 'Reason complaint filed' and 'How resolved' columns by removing anything after the first semicolon
data1['Reason complaint filed'] = data1['Reason complaint filed'].apply(lambda x: x.split(';')[0] if pd.notnull(x) else x)
data1['How resolved'] = data1['How resolved'].apply(lambda x: x.split(';')[0] if pd.notnull(x) else x)
# Clean the 'Complaint filed against' column by converting all text to uppercase
data1['Complaint filed against'] = data1['Complaint filed against'].str.upper()
# Remove rows where 'Complaint filed against' is 'UNKNOWN'
data1 = data1[data1['Complaint filed against'] != 'UNKNOWN']
#Copy of data.
data = data1.iloc[-50000:]

# Clean the 'Complaint filed against' column by converting all text to uppercase
data['Complaint filed against'] = data['Complaint filed against'].str.upper()


# Convert date columns to datetime with the correct format
data['Received date'] = pd.to_datetime(data['Received date'], format='%m/%d/%Y')
data['Closed date'] = pd.to_datetime(data['Closed date'], format='%m/%d/%Y')

# Create a new feature for the difference between received and closed dates
data['Days to Close'] = (data['Closed date'] - data['Received date']).dt.days

# Extract month from 'Received date' for analysis
data['Month'] = data['Received date'].dt.month

# Display the title of the app
st.title("TDI Complaint Data Analysis: Exploring Trends and Patterns")
st.markdown('<p style="font-size:18px;">The data analyzed is sourced from Data.Gov.</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;">The Texas Department of Insurance (TDI) handles complaints against people and organizations licensed by TDI, such as companies, agents, and adjusters.</p>', unsafe_allow_html=True)

st.write('### Sample data from Original File:')
st.write(data2.tail(20))
# Display the number of "Yes" and "No" in the "Confirmed complaint" column
if 'Confirmed complaint' in data2.columns:
    st.write('### Confirmed Complaints counts:')
    confirmed_complaint_counts = data2['Confirmed complaint'].value_counts()
    st.write(confirmed_complaint_counts)
else:
    st.write('The column "Confirmed complaint" does not exist in the dataset.')

st.markdown('<p style="font-size:18px;">Notes:</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;">1. We observe unclean data in columns How Resolved and Reason for Complaint.</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;">2. The categories Others in Complaint Filed by and Miscellaneous in Coverage type lack specificity, making the data less precise and harder to analyze.</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;">3. Confirmed complaints are fewer than unconfirmed complaints.</p>', unsafe_allow_html=True)

st.write('### Size of Raw Data:')
st.write(data2.shape)

st.write('### Data size after cleaning:')
st.write(data.shape)

st.markdown('<p style="font-size:18px;">Columns were eliminated if they contained more null values than valid data points. Additionally, rows with any null values were also removed.</p>', unsafe_allow_html=True)

# Display the number of unique categories in each column
st.write('### Number of Unique Categories in Each Column:')
unique_counts = data1.nunique().reset_index()
unique_counts.columns = ['Field','Count']
st.write(unique_counts)

st.markdown('<p style="font-size:18px;">If the data had a smaller number of categories for Reason Complaint filed, it would be easier to translate the analysis into actionable insights and targeted improvements.</p>', unsafe_allow_html=True)

# Draw a graph for complaints filed against a specific column (assuming 'Complaint filed against' column exists)
if 'Complaint filed against' in data.columns:
        st.write('### Companies with the Most Complaints Filed Against Them:')
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

st.markdown('<p style="font-size:18px;">We see that Progressive County Mutual Insurance Company followed by State Farm Mutual Automobile Insurance Company have highest complaints.</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;">We also see considerable number of complaints against Blue Cross and Blue Shield of Texas. </p>', unsafe_allow_html=True)


# Draw a grouped bar chart for the top 10 highest "Complaints filed against" vs top 5 "Complaints filed by"
if 'Complaint filed against' in data.columns and 'Complaint filed by' in data.columns:
    st.write('### Top 10 Highest Complaints Filed Against vs Top 5 Complaints Filed By:')
    top_10_complaints = data['Complaint filed against'].value_counts().nlargest(10).index
    top_5_complaints_by = data['Complaint filed by'].value_counts().nlargest(5).index
    filtered_data = data[data['Complaint filed against'].isin(top_10_complaints) & data['Complaint filed by'].isin(top_5_complaints_by)]
    complaint_pivot = filtered_data.pivot_table(index='Complaint filed against', columns='Complaint filed by', aggfunc='size', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 9))
    complaint_pivot.plot(kind='bar', stacked=True, ax=ax, color=sns.color_palette('viridis', len(top_5_complaints_by)))
    #ax.set_title('Top 10 Highest Complaints Filed Against vs Top 5 Complaints Filed By', fontsize=16)
    ax.set_xlabel('Complaint Filed Against', fontsize=14)
    ax.set_ylabel('Number of Complaints', fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.legend(title='Complaint Filed By', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
else:
    st.write('The columns "Complaint filed against" and/or "Complaint filed by" do not exist in the dataset.')

st.markdown('<p style="font-size:18px;">Notes:</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;">1. Insured and Third Party are consistently significant sources of complaints across most companies. The dark blue (Insured) and light green (Third Party) segments of the bars are generally quite prominent for almost all companies, indicating these two groups are major drivers of complaints.</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;">2. Attorney and Other categories appear to be the smallest contributors to complaints.</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;">3. Blue Cross and Blue Shield of Texas: Seems to have a particularly high proportion of Provider complaints compared to some other companies, suggesting potential issues related to provider relationships or claims processing.</p>', unsafe_allow_html=True)

# Draw a grouped bar chart for the column 'Complaint Type' (assuming it exists)
if 'Complaint type' in data.columns:
    st.write('### Complaints filed against vs Complaint Type:')
    top_10_reasons = data['Complaint type'].value_counts().nlargest(7).index
    filtered_data = data[data['Complaint type'].isin(top_10_reasons)]
    reason_complaint_pivot = filtered_data.pivot_table(index='Complaint type', columns='Complaint filed against', aggfunc='size', fill_value=0)
    reason_complaint_pivot = reason_complaint_pivot[top_10_complaints]  # Ensure the columns are in the same order as the top 10 complaints
    reason_complaint_pivot = reason_complaint_pivot.loc[top_10_reasons]  # Ensure the index is in the same order as the top 10 reasons
    fig, ax = plt.subplots(figsize=(14, 11))
    reason_complaint_pivot.plot(kind='bar', stacked=True, ax=ax, color=sns.color_palette('viridis', len(top_10_complaints)))
    #ax.set_title('Grouped Bar Chart of Reasons for Complaints Filed', fontsize=16)
    ax.set_xlabel('Complaint Type', fontsize=18)  # Increased font size
    ax.set_ylabel('Number of Complaints', fontsize=18)  # Increased font size
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.legend(title='Complaint Filed Against', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
else:
    st.write('The column "Reason complaint filed" does not exist in the dataset.')

# Display the top 10 highest "Complaints filed against" and the average number of days between received and closed dates
st.write('### Top 10 Highest Complaints Filed Against and Average Days to Close:')
top_10_complaints = data['Complaint filed against'].value_counts().nlargest(10).index
complaint_counts = data['Complaint filed against'].value_counts().reset_index()
complaint_counts.columns = ['Complaint Filed Against', 'Number of Complaints']
avg_days_to_close = data[data['Complaint filed against'].isin(top_10_complaints)].groupby('Complaint filed against')['Days to Close'].mean().reset_index()
avg_days_to_close.columns = ['Complaint Filed Against', 'Average Days to Close']
merged_data = pd.merge(complaint_counts, avg_days_to_close, on='Complaint Filed Against')
sorted_data = merged_data.sort_values(by='Number of Complaints', ascending=False)
st.write(sorted_data)

# Draw a grouped bar chart for the column 'Complaint filed against' vs 'Coverage type'
if 'Complaint filed against' in data.columns and 'Coverage type' in data.columns:
    st.write('### Complaints Filed Against vs Coverage Type:')
    top_10_complaints = data['Complaint filed against'].value_counts().nlargest(10).index
    filtered_data = data[data['Complaint filed against'].isin(top_10_complaints)]
    complaint_coverage_pivot = filtered_data.pivot_table(index='Complaint filed against', columns='Coverage type', aggfunc='size', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 9))
    complaint_coverage_pivot.plot(kind='bar', stacked=True, ax=ax, color=sns.color_palette('viridis', len(filtered_data['Coverage type'].unique())))
    #ax.set_title('Complaints Filed Against vs Coverage Type', fontsize=16)
    ax.set_xlabel('Complaint Filed Against', fontsize=14)
    ax.set_ylabel('Number of Complaints', fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.legend(title='Coverage Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
else:
    st.write('The columns "Complaint filed against" and/or "Coverage type" do not exist in the dataset.')


# Analyze the relationship between Complaint type and Coverage level
if 'Complaint type' in data.columns and 'Coverage level' in data.columns:
    st.write('### Heatmap to understand relation between Complaint Type and Coverage Level:')
    top_10_coverage_levels = data['Coverage level'].value_counts().nlargest(10).index
    filtered_data = data[data['Coverage level'].isin(top_10_coverage_levels)]
    complaint_coverage_pivot = filtered_data.pivot_table(index='Complaint type', columns='Coverage level', aggfunc='size', fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(complaint_coverage_pivot, annot=True, fmt='d', cmap='viridis', ax=ax)
    #ax.set_title('Heatmap of Complaint Type vs Coverage Level', fontsize=16)
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
    #ax.set_title('Complaints Received by Month', fontsize=16)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Number of Complaints', fontsize=14)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    #ax.bar_label(bars, fontsize=12)  # Add count on the bars with specified font size
    st.pyplot(fig)
else:
    st.write('The column "Received date" does not exist in the dataset.')

# List the top 5 highest "Respondent Role"
if 'Respondent Role' in data.columns:
    st.write('### Top 5 Respondent Roles:')
    top_5_respondent_role = data['Respondent Role'].value_counts().nlargest(5)
    st.write(top_5_respondent_role)
else:
    st.write('The column "Respondent Role" does not exist in the dataset.')


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
