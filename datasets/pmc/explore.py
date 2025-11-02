import pandas as pd
import random

# Read the CSV file
df = pd.read_csv('datasets/PMC/raw/PMC-Patients.csv')

# Display basic information about the dataset
print("\nDataset Overview:")
print(f"Total number of patients: {len(df)}")
print("\nSample of 3 patient records:")
sample_patients = df.sample(n=3, random_state=42)

for _, patient in sample_patients.iterrows():
    print("\n" + "="*80)
    print(f"Patient ID: {patient['patient_id']}")
    print(f"Patient UID: {patient['patient_uid']}")
    print(f"Gender: {patient['gender']}")
    
    # Format age display
    age_list = eval(patient['age']) if isinstance(patient['age'], str) else patient['age']
    age_str = ", ".join([f"{value} {unit}s" for value, unit in age_list])
    print(f"Age: {age_str}")
    
    # Show a snippet of the patient summary
    summary = patient['patient']
    print(f"\nPatient Summary (first 200 chars):\n{summary[:200]}...")
    
    # Show some relevant articles
    rel_articles = eval(patient['relevant_articles']) if isinstance(patient['relevant_articles'], str) else patient['relevant_articles']
    print("\nRelevant Articles:")
    for pmid, score in list(rel_articles.items())[:3]:  # Show first 3
        print(f"- PMID: {pmid}, Relevance Score: {score}")
    
    # Show similar patients
    sim_patients = eval(patient['similar_patients']) if isinstance(patient['similar_patients'], str) else patient['similar_patients']
    print("\nSimilar Patients:")
    for pat_uid, score in list(sim_patients.items())[:3]:  # Show first 3
        print(f"- Patient UID: {pat_uid}, Similarity Score: {score}")
