import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Add this import

# Load the data
raw_data = pd.read_csv('mail_data.csv')

# Replace null values with an empty string
mail_data = raw_data.where((pd.notnull(raw_data)), '')

# Replace 'spam' with 0 and 'ham' with 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separating the texts and labels
X = mail_data['Message']
Y = mail_data['Category']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Transform the text data to feature vectors
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Make predictions on the test set
Y_test_pred = model.predict(X_test_features)

# Calculate accuracy on the test data
accuracy_on_test_data = accuracy_score(Y_test, Y_test_pred)

# Streamlit App
st.title("Spam or Ham Classifier")

# User input
user_input = st.text_area("Enter the text of the email:")

# Make predictions on user input
if user_input:
    input_mail = [user_input]
    input_data_features = feature_extraction.transform(input_mail)
    pred = model.predict(input_data_features)

    # Display the prediction
    if pred[0] == 1:
        st.write('Valid Mail')
    else:
        st.write('Spam Mail')

# Display accuracy on the test data
#st.write('Accuracy on Test Data:', accuracy_on_test_data)
