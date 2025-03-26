# 🚀 Gen AI Orchestrator for Email and Document Triage/Routing

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
Gatekeeper application processes and classifies files such as emails, PDFs and images. It uses ML models and external APIs to classify the content of these files into predefined request and sub-request types. This also checks for any duplicate requests using Faiss index. 

## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)


## ⚙️ What It Does
Our application classifies the emails, documents and any attachments based on the best available dataset. It make use of LLM and with the best available score, files/content gets classified into request/sub request types for further processing.

## 🛠️ How We Built It
We made use of VS code to run the flask application. API from postman is used to process the files from the local storage to vector db and then content gets classified accordingly using scoring mechanism.

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/your-repo.git
   ```
2. Install dependencies  
   ```sh
   npm install  # or pip install -r requirements.txt (for Python)
   ```
3. Run the project  
   ```sh
   npm start  # or python app.py
   ```

## 🏗️ Tech Stack
- 🔹 Database: Vector DB
- 🔹 Other: Postman API / VS code / Python

## 👥 Team
- Venkatesh Bollam
- Subbarao Bhagavathula
- Santosh Ganapathi Varma Indukuri
- Pranay Kumar Samala
