import pandas as pd
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.preprocessing import LabelEncoder

# Load the preprocessed data
df = pd.read_csv('ailments_symptoms.csv')

# Encode the target variable as numeric labels
le = LabelEncoder()
df['ailments'] = le.fit_transform(df['ailments'])

# Create a tokenizer and fit it to the preprocessed data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['symptoms'])

# Convert the text data to sequences
sequences = tokenizer.texts_to_sequences(df['symptoms'])

# Pad the sequences to a fixed length
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import tensorflow.keras.backend as K
K.clear_session()

# Train the model
model.fit(padded_sequences, df['ailments'], batch_size=32, epochs=10, validation_split=0.2)

# Define a function to classify patient ailments based on their symptoms
def classify_ailment(symptoms):
    # Convert the symptoms to a sequence
    sequence = tokenizer.texts_to_sequences([symptoms])
    
    # Pad the sequence to a fixed length
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Predict the ailment and get confidence level
    prediction = model.predict(padded_sequence)[0][0]
    
    # Decode the numeric label to get the ailment name
    ailment = le.inverse_transform([int(round(prediction))])[0]
    
    # Return the predicted ailment and confidence level
    return ailment, prediction


# Define a function to get input from user
def get_input():
    print("Enter your symptoms:")
    text_input = input()
    return text_input

# Define a function to generate feedback based on the confidence level
def generate_feedback(confidence_level):
    if confidence_level >= 0.9:
        return "The predicted the ailment with a very high level of confidence."
    elif confidence_level >= 0.7:
        return " The predicted the ailment with a decent level of confidence, but there may be room for improvement. Consider your parameters to see if you can achieve better results."
    elif confidence_level >= 0.5:
        return " The predicted of the ailment with a relatively low level of confidence. This may be due : insufficient data. Consider inputing sufficient data to improve your model to achieve better results."
    else:
        return "THe prediction of the ailment with a very low level of confidence. Consider seeking additional guidance or resources to improve your results and condition."
    

# Define a function to send email notifications
def send_email_notification(ailment, confidence_level, recipient):
    sender_email = 'naibrian44@gmail.com'
    sender_password = 'your_password'
    receiver_email = recipient
    message = MIMEMultipart()
    message['Subject'] = 'Predicted Ailment and Confidence Level'
    message['From'] = sender_email
    message['To'] = receiver_email
    body = f"The predicted ailment is {ailment} with a confidence level of {confidence_level}."
    message.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        print("Email notification sent successfully!")
    except Exception as e:
        print(f"An error occurred while sending email notification: {e}")
    finally:
        server.quit()

# Example usage:
symptoms = get_input()
ailment, confidence_level = classify_ailment(symptoms)
feedback = generate_feedback(confidence_level)
print(f"The predicted ailment is {ailment}.")
print(feedback)
