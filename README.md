# 🚀 Gen AI Orchestrator for Email and Document Triage/Routing

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
Gatekeeper application processes and classifies files such as emails, PDFs and images. It uses ML models and external APIs to classify the content of these files into predefined request and sub-request types. This also checks for any duplicate requests using Faiss index. 

## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  gaied-pioneersofpe/artifacts/demo
🖼️ Screenshots:

![image](https://github.com/user-attachments/assets/f527b4ef-5f4b-4c10-b4e0-dd086785e953)


## ⚙️ What It Does
Our application classifies the emails, documents and any attachments based on the best available dataset. It make use of LLM and with the best available score, files/content gets classified into request/sub request types for further processing.

## 🛠️ How We Built It
We made use of VS code to run the python flask application. API from postman is used to process the files from the local storage to vector db and then content gets classified accordingly using scoring mechanism.

## 🏃 How to Run

Refer Demo video & guide in the below location :  gaied-pioneersofpe/artifacts/demo

1. Clone the repository  
   ```sh
   git clone https://github.com/ewfx/gaied-pioneersofpe.git
   ```
2. Install dependencies  
   ```sh
   pip install -r requirements.txt (for Python)
   ```
3. Run the project  
   ```sh
   python code/src/3_restService.py
   ```


**Api request in postman:**
Access the API request as shown in the below image. pass folder location as header.
![image](https://github.com/user-attachments/assets/f527b4ef-5f4b-4c10-b4e0-dd086785e953)



## 🏗️ Tech Stack
Programming Language: Python,
Web Framework: FastAPI,
Machine Learning Libraries:
Transformers (Hugging Face)
PyTorch
scikit-learn
PIL (Python Imaging Library)
pytesseract
faiss
OCR Library: pytesseract
Email Processing: email
PDF Processing: PyPDF2
HTTP Client: requests
JSON Processing: json
HTML Parsing: BeautifulSoup
Environment Management: dotenv
Logging: logging

## 👥 Team

- G S Venkatesh Bollam
- Subbarao Bhagavathula
- Santosh Ganapathi Varma Indukuri
- Pranay Kumar Samala
- Kalathur Ashok Kumar
