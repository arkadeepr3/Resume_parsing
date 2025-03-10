import re
import os
import spacy
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import gender_guesser.detector


#  Load spaCy model
nlp = spacy.load("en_core_web_sm")

#  Try loading BERT NER model (if available)
try:
    model_path = "models/bert-ner"  
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    else:
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")
except Exception:
    ner_pipeline = None  # Use spaCy if BERT fails

#  Predefined skill set for better matching
SKILL_KEYWORDS = {
    # Software Architecture & Design
    "Microservices", "Microservice Architecture", "Microservices Pattern", "Monolithic Architecture",
    "Distributed Systems", "SOA", "Serverless", "Event-Driven Architecture",

    # Programming Languages
    "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript", "Swift", "Kotlin",
    "Go", "Rust", "Dart", "PHP", "Ruby", "Perl", "Lua", "R", "Objective-C",

    # Data Structures & Algorithms
    "Data Structures", "Data Structure", "DSA", "Algorithm", "Algorithms",

    # Object-Oriented Programming
    "OOP", "OOPs", "OOPS Concepts", "Object-Oriented Programming", "Encapsulation", "Polymorphism", "Abstraction", "Inheritance",

    # Web & Frontend Technologies
    "HTML", "CSS", "React", "Vue.js", "Angular", "Node.js", "Express.js", "Django", "Flask", "FastAPI",

    # Databases & Query Languages
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "GraphQL", "Cassandra", "Redis",

    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "CI/CD", "Linux",

    # Machine Learning & AI
    "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch", "Keras",

    #  Cybersecurity
    "Penetration Testing", "Ethical Hacking", "Cryptography", "Network Security", "SOC Analysis",

    # Software Testing & Automation
    "Selenium", "JUnit", "PyTest", "Cypress", "Appium", "Postman", "API Testing", "Manual Testing",
    "Automation Testing", "Performance Testing", "Load Testing", "Integration Testing", "Regression Testing",

    # Soft Skills
    "Communication", "Leadership", "Problem Solving", "Critical Thinking", "Teamwork",
    "Adaptability", "Decision Making", "Public Speaking", "Project Management"
}

#  Extract Email & Phone Number
def extract_email_mobile(text):
    """Extracts email and mobile number from resume text."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"\+?\d{1,4}[-.\s]?\(?\d{2,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{4,6}"

    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)

    valid_phones = [p for p in phones if len(re.sub(r"\D", "", p)) in range(10, 15)]  # 10-14 digits

    return emails[0] if emails else None, valid_phones[0] if valid_phones else None  # Return only the first match

#  Extract Name
JOB_TITLES = {"developer", "engineer", "manager", "programmer", "scientist", "consultant","software", "data", "architect", "intern", "administrator", "specialist", "executive"}

#  Extract Name from First Line + spaCy Fallback
def extract_name(text):
    """
    Extracts the candidate's name by returning the first valid line.
    Ignores job titles, emails, and numbers.
    """

    # Step 1: Get the first non-empty line
    lines = [line.strip() for line in text.split("\n") if line.strip()]  # Remove empty lines
    if not lines:
        return None  #  If no text, return None

    first_line = lines[0]  #  Get the first line

    #  Ignore lines with emails, numbers, or common job titles
    if (
        "@" in first_line  #  Avoid extracting email
        or any(char.isdigit() for char in first_line)  #  Avoid numbers
        or re.search(r"\b(developer|engineer|manager|consultant|scientist|architect|intern|specialist|executive)\b", first_line, re.IGNORECASE)  # ✅ Avoid job titles
    ):
        return None  #  Not a valid name

    return first_line
#  Define Known Locations
KNOWN_LOCATIONS = {
    #  Major Indian Cities
    "Bangalore", "Mumbai", "Delhi", "Pune", "Chennai", "Hyderabad", "Kolkata",
    "Ahmedabad", "Gurgaon", "Noida", "Indore", "Jaipur", "Lucknow", "Chandigarh",
    "Bhopal", "Patna", "Bhubaneswar", "Thiruvananthapuram", "Kochi", "Visakhapatnam",
    "Coimbatore", "Vadodara", "Nagpur", "Surat", "Kanpur", "Varanasi", "Ludhiana",
    "Madurai", "Raipur", "Nashik", "Jodhpur", "Ranchi", "Agra", "Dehradun", "Meerut",
    "Gwalior", "Jabalpur", "Mysore", "Guwahati", "Amritsar", "Vijayawada","Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal","Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli", "Daman and Diu",
    "Lakshadweep", "Delhi", "Puducherry", "Ladakh", "Jammu and Kashmir","North India", "South India", "East India", "West India", "Central India","Northeast India"
}

#  Regex pattern to detect location phrases
CONTACT_LOCATION_PATTERN = re.compile(
    r"(?:\bLocation\b|\bLives in\b|\bBased in\b|\bResiding at\b|\bWorks in\b|\bCurrently in\b)[:\s]*([A-Za-z\s]+)", 
    re.IGNORECASE
)

#  Define common false positives to avoid
FALSE_POSITIVES = {"analysis, design", "skills, expertise", "data, science", "technology, innovation"}

def extract_location_from_contact_section(text):
    """Extracts the most probable location from the 3 lines after the name in the resume."""

    # Step 1: Split text into lines
    lines = text.strip().split("\n")

    #  Step 2: Find the first non-empty line (assumed to be the Name)
    name_line_index = None
    for i, line in enumerate(lines):
        if line.strip():
            name_line_index = i
            break

    if name_line_index is None:  # No valid lines found
        return "Not Found"

    # Step 3: Get the next 3 lines after the name
    contact_lines = lines[name_line_index + 1:name_line_index + 4]

    #  Step 4: Try regex-based location extraction
    for line in contact_lines:
        match = CONTACT_LOCATION_PATTERN.search(line)
        if match:
            extracted_location = match.group(1).strip()

            # Ensure the extracted location is valid
            if extracted_location.lower() not in FALSE_POSITIVES:
                return extracted_location

    #  Step 5: Search for known locations in the 3 contact lines
    for line in contact_lines:
        for city in KNOWN_LOCATIONS:
            if re.search(rf"\b{city}\b", line, re.IGNORECASE):
                return city

    return "Not Found"

#  Extract Education
DEGREE_PATTERNS = [
    r"(Bachelor|Master|B\.?E\.?|M\.?E\.?|B\.?Tech|M\.?Tech|B\.?Sc|M\.?Sc|MBA|PhD|Diploma|BCA|MCA)\b"
]

def extract_education(text):
    """Extracts degree names only (B.Tech, MCA, MBA) without extra details."""
    education = set()
    for pattern in DEGREE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            education.add(match.strip())  
    return list(education)

#  Extract Years of Experience
def extract_years_of_experience(text):
    """Extracts years of experience from resume text."""
    experience_pattern = r'(\d+\+?\s*(years?|yrs?)(?:\s*of\s*experience)?)|(\d+\s*-\s*\d+\s*(years?|yrs?))'
    matches = re.findall(experience_pattern, text, re.IGNORECASE)

    for match in matches:
        if match[0]:
            return match[0].strip()
        elif match[2]:
            return match[2].strip()

    return None

#  Define common spoken languages
COMMON_LANGUAGES = [
    "English", "Hindi", "French", "German", "Spanish", "Chinese", "Japanese", "Korean",
    "Arabic", "Portuguese", "Russian", "Italian", "Dutch", "Bengali", "Tamil", "Telugu",
    "Marathi", "Gujarati", "Punjabi", "Urdu", "Malayalam", "Kannada", "Odia"
]

#  Define programming languages to exclude
PROGRAMMING_LANGUAGES = {"Python", "Java", "C++", "C#", "JavaScript", "SQL", "R", "PHP", "Swift", "Go", "Rust"}

def extract_languages(text):
    """
    Extracts spoken languages known from a resume text.
    Ignores programming languages.
    """
    detected_languages = set()

    #  Search for each language in the text
    for language in COMMON_LANGUAGES:
        if re.search(rf"\b{language}\b", text, re.IGNORECASE):  #  Match whole word
            detected_languages.add(language)

    #  Remove programming languages if detected accidentally
    detected_languages -= PROGRAMMING_LANGUAGES

    return list(detected_languages) if detected_languages else ["Not Mentioned"]

def extract_skills(text):
    """Extracts both technical and soft skills from resume text, ensuring whole-word matching."""
    detected_skills = set()
    text_lower = text.lower()

    for skill in SKILL_KEYWORDS:
        #   regex with word boundaries (\b) for accurate matching
        pattern = rf"\b{re.escape(skill.lower())}\b"
        if re.search(pattern, text_lower, re.IGNORECASE):
            detected_skills.add(skill)

    return list(detected_skills) if detected_skills else ["Not Mentioned"]

def extract_pin_code(text):
    """Extracts the first valid PIN code (postal code) from the resume."""
    pin_patterns = [
        r"\b\d{6}\b",           # Indian PIN codes (6 digits) - e.g., 560078
        r"\b\d{5}(?:-\d{4})?\b" # US ZIP codes (5 or 9 digits) - e.g., 10001 or 10001-2345
    ]

    lines = text.strip().split("\n")[:10]

    for line in lines:
        for pattern in pin_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group()

    return "Not Available"

# ✅ Initialize Gender Detector
detector = gender_guesser.detector.Detector()

def extract_gender(text, name=""):
    """Infers gender based on pronouns, titles, and name matching."""
    male_pronouns = [" he ", " him ", " his ", " mr."]
    female_pronouns = [" she ", " her ", " hers ", " ms.", " mrs.", " miss."]
    
    text_lower = " " + text.lower() + " "  # Add spaces for word boundaries
    
    # ✅ Step 1: Check for gendered pronouns/titles
    for pronoun in male_pronouns:
        if pronoun in text_lower:
            return "Male"
    
    for pronoun in female_pronouns:
        if pronoun in text_lower:
            return "Female"

    # ✅ Step 2: Infer Gender Using First Name (Fallback)
    if name:
        first_name = name.split()[0]  # Extract first name
        guessed_gender = detector.get_gender(first_name)
        
        if guessed_gender in ["male", "mostly_male"]:
            return "Male"
        elif guessed_gender in ["female", "mostly_female"]:
            return "Female"

    return "Not Mentioned"


#  Extract All Resume Data
def extract_resume_data(text):
    """Extracts structured resume data from text."""
    email, phone = extract_email_mobile(text)

    return {
        "Name": extract_name(text),
        "Email": email,
        "Phone": phone,
        "Education": extract_education(text),
        "Experience": extract_years_of_experience(text),
        "Skills": extract_skills(text),
        "Location": extract_location_from_contact_section(text),
        "PIN Code": extract_pin_code(text),
        "Languages Known": extract_languages(text),
        "Gender": extract_gender(text),
    }
