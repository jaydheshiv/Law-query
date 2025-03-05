import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string
import streamlit as st
import spacy
from spacy.matcher import Matcher
from textblob import TextBlob
from rank_bm25 import BM25Okapi

# Add custom CSS
st.markdown("""
    <style>
        /* General styling */
        body {
            background-color: #F0F0F5; /* Set the background color */
            color: #333333; /* Set the text color */
            font-family: 'Helvetica', sans-serif; /* Font family */
        }

        /* Header styling */
        .stApp h1, .stApp h2, .stApp h3 {
            color: #FF6F61; /* Header color */
        }

        /* Button styling */
        .stButton > button {
            background-color: #FF6F61; /* Button color */
            color: white; /* Text color on buttons */
            border: none;
            border-radius: 5px; /* Rounded corners */
            padding: 10px 15px; /* Button padding */
            cursor: pointer; /* Pointer cursor on hover */
        }

        .stButton > button:hover {
            background-color: #E03E31; /* Darker shade on hover */
        }

        /* Sidebar styling */
        .stSidebar {
            background-color: #E3E4E8; /* Sidebar background color */
            color: #333333; /* Sidebar text color */
        }

        /* Title styling in sidebar */
        .stSidebar h2 {
            color: #FF6F61; /* Sidebar header color */
        }

        /* Input box styling */
        .stTextInput input {
            border: 2px solid #FF6F61; /* Input border color */
            border-radius: 5px; /* Rounded corners */
            padding: 10px; /* Padding inside input */
        }

        /* Subheader styling */
        .stSubheader {
            color: #FF6F61; /* Subheader color */
        }

        /* Customizing markdown text */
        .stMarkdown {
            font-size: 16px; /* Markdown font size */
            line-height: 1.6; /* Line height for readability */
        }

        /* Add any additional styling here */
    </style>
""", unsafe_allow_html=True)
# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load NLTK stopwords
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load spaCy model for NER and Dependency Parsing
nlp = spacy.load("en_core_web_sm")

# Rule-based entity matching for general legal terms
matcher = Matcher(nlp.vocab)
legal_terms = ["divorce", "contract", "agreement", "injury", "liability", "warranty", "negligence", "custody", "tort", 
    "damages", "injunction", "breach of contract", "nuisance", "trespass", "defamation", "alimony", "adoption", 
    "paternity", "traffic violation", "license suspension", "insurance", "DUI", "registration", "appeal", 
    "summons", "judgment", "helmet law", "seatbelt law", "road safety","license", "penalty", "fine", "speeding","Article"]
for term in legal_terms:
    pattern = [{"LOWER": term}]
    matcher.add(term, [pattern])

# Load datasets
civil_law_path = 'D:\\sem-5\\NLP\\civil.csv'
family_law_path = 'D:\\sem-5\\NLP\\family.csv'
motor_vehicles_act_path = 'D:\\sem-5\\NLP\\Motor_Vehicles_Act_Serial_1988.csv'

# Read the datasets
civil_law_df = pd.read_csv(civil_law_path, on_bad_lines='skip')
family_law_df = pd.read_csv(family_law_path, on_bad_lines='skip')
motor_vehicles_act_df = pd.read_csv(motor_vehicles_act_path, on_bad_lines='skip')

# Add a 'Law Type' column to each dataset
civil_law_df['Law Type'] = 'Civil Law'
family_law_df['Law Type'] = 'Family Law'
motor_vehicles_act_df['Law Type'] = 'Motor Vehicles Act'

# Select relevant columns from all datasets
civil_law_cleaned_df = civil_law_df[['Section', 'Description', 'Law Type']].copy()
family_law_cleaned_df = family_law_df[['Section', 'Description', 'Law Type']].copy()
motor_vehicles_act_cleaned_df = motor_vehicles_act_df[['Section', 'Description', 'Law Type']].copy()

# Combine the datasets into one DataFrame
combined_laws_df = pd.concat([civil_law_cleaned_df, family_law_cleaned_df, motor_vehicles_act_cleaned_df])
combined_laws_df.dropna(inplace=True)  # Ensure there are no missing values

# Define a general text preprocessing function
def preprocess_text(text):
    tokens = text.lower().split()
    tokens = [word.strip(string.punctuation) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the descriptions
combined_laws_df['Processed_Description'] = combined_laws_df['Description'].apply(preprocess_text)

# Split the data into training and testing sets
X = combined_laws_df['Processed_Description']
y = combined_laws_df['Law Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a TF-IDF Vectorizer and a Logistic Regression classifier
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

# Train the classifier
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to classify the law type based on the user's query
def classify_law_type(query, model):
    corrected_query = str(TextBlob(query).correct())
    prediction = model.predict([corrected_query])[0]
    return prediction

# Initialize BM25 for retrieval
def initialize_bm25(descriptions):
    tokenized_corpus = [desc.split() for desc in descriptions]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

bm25_model = initialize_bm25(combined_laws_df['Processed_Description'].tolist())

# Function to retrieve the most relevant law section based on a query using BM25
def retrieve_law_bm25(query, bm25_model, combined_laws_df):
    processed_query = preprocess_text(query).split()
    scores = bm25_model.get_scores(processed_query)
    
    # Check if the query contains specific phrases
    if "driving with a license" in query.lower():
        return None, "This query pertains to driving with a license; no penalties are applicable."
    
    # Get the index of the most similar law description
    best_match_idx = scores.argmax()
    
    # Retrieve the best matching law section and description
    best_match_section = combined_laws_df.iloc[best_match_idx]['Section']
    best_match_description = combined_laws_df.iloc[best_match_idx]['Description']
    
    return best_match_section, best_match_description

# Function to analyze syntax, semantics, tagging concepts, and chunk relationships
def analyze_syntax_and_semantics(query):
    # Parse the query with spaCy
    doc = nlp(query)
    
    # Named Entities (Semantics)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Rule-based matching for legal terms
    matches = matcher(doc)
    matched_legal_terms = [doc[start:end].text for match_id, start, end in matches]
    
    # Part-of-speech tagging
    pos_tags = [(token.text, token.pos_) for token in doc]
    
    # Dependency Parsing (Parse Tree)
    parse_tree = [(token.text, token.dep_, token.head.text) for token in doc]
    
    # Noun phrase chunking (Chunk relationship)
    noun_chunks = [(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text) for chunk in doc.noun_chunks]
    
    return entities, pos_tags, parse_tree, noun_chunks, matched_legal_terms

# Function to perform semantic analysis and verification
def semantic_analysis_and_verification(query):
    # Using TextBlob for simple sentiment analysis
    blob = TextBlob(query)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Verifying the sentiment
    sentiment = "Neutral"
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"

    return sentiment, polarity, subjectivity

# Function to detect negation in a query
def detect_negation(query):
    negation_terms = ['not', 'no', 'never', 'none', 'neither', 'nor', 'nobody', 'nothing', 'nowhere', 'without']
    query_tokens = query.lower().split()
    negations = [term for term in negation_terms if term in query_tokens]
    return len(negations) > 0, negation_terms

# Streamlit application UI
st.title("Generalized Law Query Retrieval and Classification System")

# Query input
user_query = st.text_input("Enter your legal query:")

# Initialize containers for results
if user_query:
    # Classify law type and show result directly
    law_type = classify_law_type(user_query, model)
    st.subheader("Predicted Law Type:")
    st.write(law_type)

    # Retrieve the law section and show result directly
    best_section, best_description = retrieve_law_bm25(user_query, bm25_model, combined_laws_df)
    st.subheader("Best Matching Section:")
    if best_section is None:
        st.write(best_description)
    else:
        st.write(best_section)
        st.subheader("Description:")
        st.write(best_description)

    # Detect negation
    has_negation, negation_terms = detect_negation(user_query)
    
    if has_negation:
        st.subheader("Negation Detected!")
        

    # Perform syntax and semantics analysis
    entities, pos_tags, parse_tree, noun_chunks, matched_legal_terms = analyze_syntax_and_semantics(user_query)

    # Sidebar for analysis buttons
    st.sidebar.header("Analysis Options")
    
    if st.sidebar.button("Show Part-of-Speech Tags (Syntax)"):
        st.subheader("Part-of-Speech Tags (Syntax):")
        st.write(pos_tags)

    if st.sidebar.button("Show Parse Tree (Dependency)"):
        st.subheader("Parse Tree (Dependency):")
        st.write(parse_tree)

    if st.sidebar.button("Show Noun Phrases and Relationships (Chunk)"):
        st.subheader("Noun Phrases and Relationships (Chunk):")
        st.write(noun_chunks)

    if st.sidebar.button("Show Named Entities (Semantics)"):
        st.subheader("Named Entities (Semantics):")
        st.write(entities)

    if st.sidebar.button("Show Matched Legal Terms"):
        st.subheader("Matched Legal Terms:")
        st.write(matched_legal_terms)

    if st.sidebar.button("Show Sentiment Analysis"):
        sentiment, polarity, subjectivity = semantic_analysis_and_verification(user_query)
        st.subheader("Sentiment Analysis Result:")
        st.write(f"Sentiment: {sentiment}, Polarity: {polarity}, Subjectivity: {subjectivity}")
