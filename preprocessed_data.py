import pandas as pd
import random

# Define the possible ailments and symptoms
ailments = ['cold', 'flu', 'headache', 'stomachache', 'ear infection', 'bronchitis', 'pneumonia', 'sinus infection', 'tonsillitis', 'migraine', 'toothache', 'back pain', 'sprain', 'fracture', 'arthritis', 'ulcer', 'diabetes', 'hypertension', 'asthma', 'allergies', 'rash', 'eczema', 'psoriasis', 'acne', 'conjunctivitis']
symptoms = ['runny nose', 'cough', 'fever', 'nausea', 'vomiting', 'headache', 'ear pain', 'throat pain', 'chest pain', 'shortness of breath', 'wheezing', 'fatigue', 'muscle aches', 'joint pain', 'stiffness', 'swelling', 'numbness', 'tingling', 'dizziness', 'blurred vision', 'redness', 'itching', 'dryness', 'scaling', 'tearing']

# Generate the data
data = []
for i in range(100):
    ailment = random.choice(ailments)
    symptom_count = random.randint(1, 4)
    symptom_indices = random.sample(range(len(symptoms)), symptom_count)
    symptom_list = [symptoms[i] for i in symptom_indices]
    symptom_str = ', '.join(symptom_list)
    data.append({'ailments': ailment, 'symptoms': symptom_str})

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('ailments_symptoms.csv',index=False)