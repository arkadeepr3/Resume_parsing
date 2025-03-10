import os
import pandas as pd
from utils.extract_text import extract_text_from_pdf, extract_text_from_docx
from utils.extract_info import extract_resume_data

#  Folder Paths
INPUT_FOLDER = "resumes/"
OUTPUT_FOLDER = "output/"
CSV_FILE = os.path.join(OUTPUT_FOLDER, "extracted_resumes.csv")

#  Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#  Extract data from all resumes (overwrite CSV every time)
print(" Extracting all resumes...")

#  Get all resumes from input folder
resume_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith((".pdf", ".docx"))]

extracted_data = []
for resume in resume_files:
    resume_path = os.path.join(INPUT_FOLDER, resume)

    #  Extract text based on file type
    if resume.endswith(".pdf"):
        text = extract_text_from_pdf(resume_path)
    elif resume.endswith(".docx"):
        text = extract_text_from_docx(resume_path)
    else:
        continue  # Skip unsupported formats

    #  Extract structured data
    resume_data = extract_resume_data(text)
    resume_data["Filename"] = resume  #  Store filename for tracking
    extracted_data.append(resume_data)

#  Save new extracted data to CSV (overwrite old CSV)
df = pd.DataFrame(extracted_data)
df.to_csv(CSV_FILE, index=False)

print(f" Resume data updated in {CSV_FILE}")

#  Load extracted resume data
df = pd.read_csv(CSV_FILE)

#  Function to filter resumes based on user input
def filter_resumes(location=None, min_experience=None, skills=None, degree=None, pin_code=None):
    """Filters resumes dynamically based on user criteria."""
    filtered_df = df.copy()

    #  Filter by location
    if location:
        filtered_df = filtered_df[filtered_df["Location"].str.contains(location, case=False, na=False)]

    #  Filter by minimum experience (convert to numeric)
    if min_experience:
        filtered_df["Experience"] = filtered_df["Experience"].astype(str).str.extract(r'(\d+)').astype(float)
        filtered_df = filtered_df[filtered_df["Experience"] >= min_experience]

    #  Filter by skills (must match at least one)
    if skills:
        skill_pattern = "|".join(skills)
        filtered_df = filtered_df[filtered_df["Skills"].str.contains(skill_pattern, case=False, na=False)]

    #  Filter by degree
    if degree:
        filtered_df = filtered_df[filtered_df["Education"].str.contains(degree, case=False, na=False)]

    #  Filter by PIN Code (optional)
    if pin_code:
        filtered_df = filtered_df[filtered_df["PIN Code"].astype(str).str.contains(pin_code, case=False, na=False)]

    return filtered_df

#  Get user input
print("\n🔎 Resume Filtering System")
user_location = input("Enter location (or press Enter to skip): ").strip()
user_experience = input("Enter minimum experience (or press Enter to skip): ").strip()
user_skills = input("Enter skills (comma-separated, or press Enter to skip): ").strip()
user_degree = input("Enter degree (or press Enter to skip): ").strip()
user_pin_code = input("Enter PIN Code (or press Enter to skip): ").strip()

#  Convert inputs
min_experience = int(user_experience) if user_experience.isdigit() else None
skills = [s.strip() for s in user_skills.split(",")] if user_skills else None
degree = user_degree if user_degree else None
pin_code = user_pin_code if user_pin_code else None

#  Apply filtering
filtered_data = filter_resumes(location=user_location, min_experience=min_experience, skills=skills, degree=degree, pin_code=pin_code)

#  Display results
if not filtered_data.empty:
    print("\n Matching Resumes:")
    for _, row in filtered_data.iterrows():
        #  Extract and format skills properly
        skills_list = row["Skills"].strip("[]").replace("'", "").split(", ") if row["Skills"] else []
        languages_list = row["Languages Known"].strip("[]").replace("'", "").split(", ") if row["Languages Known"] else []

        print(f" Name: {row['Name']}")
        print(f" Email: {row['Email']}")
        print(f" Phone: {row['Phone']}")
        print(f" Location: {row['Location']}")
        print(f" PIN Code: {row['PIN Code'] if row['PIN Code'] else 'Not Available'}")
        print(f" Skills: {', '.join(skills_list) if skills_list else 'Not Mentioned'}")
        print(f" Languages Known: {', '.join(languages_list) if languages_list else 'Not Mentioned'}") 
        print(f" Gender: {row['Gender'] if 'Gender' in row and row['Gender'] else 'Not Mentioned'}")
        print(f" Experience: {row['Experience'] if 'Experience' in row and row['Experience'] else 'Not Mentioned'}")
        print("-" * 40)
else:
    print("\n No matching resumes found.")
