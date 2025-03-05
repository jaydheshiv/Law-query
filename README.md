 ⚖️ Generalized Law Query Retrieval and Classification System

 📌 Project Overview
The **Generalized Law Query Retrieval and Classification System** is an **AI-powered legal document retrieval system** that helps users classify legal queries and retrieve relevant law sections based on their input. The system utilizes **TF-IDF + Logistic Regression** for law classification and **BM25** for legal document retrieval, alongside **syntax & semantics analysis, named entity recognition (NER), sentiment analysis, and negation detection**.

 🎯 Features
- **Legal Query Classification** – Identifies whether a legal query falls under **Civil Law, Family Law, or the Motor Vehicles Act**.
- **AI-powered Legal Information Retrieval** – Uses **BM25 ranking** to fetch the most relevant legal sections.
- **Text Preprocessing & NLP** – Performs **tokenization, stopword removal, lemmatization**.
- **Named Entity Recognition (NER)** – Identifies legal terms and key entities within queries.
- **Syntax & Semantics Analysis** – Provides **POS tagging, dependency parsing, noun phrase chunking**, and **rule-based entity matching**.
- **Sentiment Analysis** – Determines sentiment polarity using **TextBlob**.
- **Negation Detection** – Identifies negations within legal queries.
- **User-Friendly UI** – Built using **Streamlit** for an intuitive and interactive user experience.

 🚀 Installation & Setup
 Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **pip**
- **Virtual Environment (Optional but Recommended)**

 Clone the Repository
```sh
git clone https://github.com/your-username/law-query.git
cd law-query
```

 Create & Activate a Virtual Environment
```sh
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate     # On Windows
```

 Install Dependencies
```sh
pip install -r requirements.txt
```

 Download NLP Models
```sh
python -m spacy download en_core_web_sm
```

 Start the Application
```sh
streamlit run app.py
```

 📁 Project Structure
```
law-query/
│-- data/
│   │-- civil.csv
│   │-- family.csv
│   │-- Motor_Vehicles_Act_Serial_1988.csv
│-- app.py                  # Main Streamlit application
│-- requirements.txt
│-- README.md
```

 📍 Key Functionalities & Implementation
 1️⃣ **Legal Query Classification**
- Uses **TF-IDF + Logistic Regression** for classification.
- Categorizes queries into **Civil Law, Family Law, or the Motor Vehicles Act**.

 2️⃣ **Legal Document Retrieval**
- Implements **BM25 ranking** for retrieving the most relevant legal sections.
- Uses preprocessed legal texts for improved matching.

 3️⃣ **Named Entity Recognition (NER)**
- Identifies key legal terms in queries.
- Extracts **persons, organizations, laws, and legal actions**.

 4️⃣ **Syntax & Semantics Analysis**
- Provides:
  - **POS tagging**
  - **Dependency parsing**
  - **Noun phrase chunking**
  - **Rule-based entity matching**

 5️⃣ **Sentiment & Negation Analysis**
- Uses **TextBlob** to determine sentiment polarity.
- Detects negations to enhance query understanding.

 📸 Screenshot
