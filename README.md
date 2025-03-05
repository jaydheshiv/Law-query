Generalized Law Query Retrieval and Classification System
This is a Natural Language Processing (NLP) based legal query system that classifies user queries into different law categories and retrieves the most relevant legal sections based on the input. The system also provides syntactic, semantic, and sentiment analysis of user queries.

Features
Law Classification: Uses TF-IDF + Logistic Regression to classify queries into Civil Law, Family Law, or Motor Vehicles Act.
Legal Information Retrieval: Implements BM25 ranking algorithm to find the most relevant legal sections.
Syntax & Semantics Analysis:
Named Entity Recognition (NER)
Part-of-Speech (POS) tagging
Dependency Parsing
Noun Phrase Chunking
Sentiment Analysis: Uses TextBlob to determine the sentiment (positive, negative, or neutral) of the query.
Negation Detection: Identifies if negation words like not, never, no are present in the query.
Streamlit UI: Interactive user interface for entering queries and analyzing legal information.
Installation
1. Clone the Repository
sh
Copy
Edit
git clone https://github.com/jaydheshiv/Law-Query.git
cd Law-Query
2. Install Dependencies
Make sure you have Python 3.8+ installed. Then, install the required packages:

sh
Copy
Edit
pip install -r requirements.txt
Usage
Run the Streamlit app:

sh
Copy
Edit
streamlit run app.py
Then, enter your legal query in the text box to classify the law type and retrieve relevant legal information.

File Structure
graphql
Copy
Edit
📂 Law-Query/
│── 📄 app.py  # Main Streamlit application
│── 📂 data/
│   ├── civil.csv  # Civil Law dataset
│   ├── family.csv  # Family Law dataset
│   ├── Motor_Vehicles.csv  # Motor Vehicles Law dataset
Technologies Used
Python
Streamlit (for UI)
scikit-learn (for ML model)
spaCy (for NLP)
BM25Okapi (for legal text retrieval)
TextBlob (for sentiment analysis)
NLTK (for preprocessing)
