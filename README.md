# XNL-21BCI0273-LLM-4
LLM TASK 4: REAL‑TIME FINANCIAL ASSISTANT CHATBOT WITH MULTI‑MODAL CONTEXT RETRIEVAL



## Overview

This repository contains a real-time financial assistant chatbot that leverages multi-modal context retrieval to answer financial queries. The system integrates multiple data sources (PDFs, images, CSV files, and real-time financial APIs) into a retrieval-augmented generation (RAG) pipeline powered by an open-source LLM. The solution is designed to be cloud-native, scalable, and secure.

The project is structured into multiple phases:

- **Phase 1: Inception & Strategic Planning**  
  Define core features, user interaction flows, architecture design, and cloud strategy.

- **Phase 2: Data Collection, Preprocessing & Integration**  
  Integrate financial data APIs, open banking APIs, and historical data. Preprocess multi-modal inputs (PDF, images via OCR, CSV) for retrieval.

- **Phase 3: Model Selection, Fine-Tuning & AI Agent Integration**  
  Choose and fine-tune an open-source LLM (e.g., GPT-Neo), and implement a RAG module for generating responses using retrieved context.

- **Phase 4: Real-Time Chatbot Interaction & Frontend Integration**  
  Build a Streamlit-based web interface for interactive user queries and display conversation history.

- **Phase 5: Testing, Validation & Performance Monitoring**  
  Write unit, integration, and end-to-end tests. Implement load testing and model drift detection using tools like pytest, Cypress, Artillery, and MLflow.

- **Phase 6: Multi-Cloud Deployment, Scalability & Maintenance**  
  Containerize the application with Docker, deploy with Kubernetes, set up CI/CD pipelines, and monitor performance with Prometheus/Grafana.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)


## Features

- **Real-Time Financial Q&A:** Answer queries related to stocks, personal finance, investment advice, and cryptocurrencies.
- **Multi-Modal Data Integration:** Processes PDFs, images (using OCR), CSV files, and integrates live data APIs.
- **Retrieval-Augmented Generation (RAG):** Combines document retrieval (via vector embeddings stored in Chroma) with an open-source LLM (e.g., GPT-Neo) for generating responses.
- **User-Friendly Interface:** A Streamlit-based chat interface for real-time interaction.
- **Scalable Cloud-Native Architecture:** Designed for containerized deployment with Kubernetes, CI/CD integration, and performance monitoring.

## Project Structure

D:\genAI
├── app.py # Main Streamlit application ├── Dockerfile # Dockerfile for containerizing the app ├── requirements.txt # Python dependencies ├── utils.py # Utility functions (e.g., PDF, image, CSV extraction) ├── data_processing.py # Text splitting, normalization, etc. ├── embeddings.py # Functions to create vector DB with Chroma ├── retrieval_qa.py # Setup and configuration for the QA chain (RAG) ├── tests
│ ├── init.py # (Optional) Make tests a package │ └── test_core.py # Unit tests for core functionalities └── README.md # This file

#Architecture Diagram

![image](https://github.com/user-attachments/assets/0d70075e-795a-4661-bd82-8e25f9d67ade)


## Installation

### Prerequisites

- Python 3.9 or later (recommended)
- Git

### Clone the Repository

bash
git clone https://github.com/yourusername/XNL-21BCI0273-LLM-4.git
cd XNL-21BCI0273-LLM-4

Set Up the Environment

Create a Conda environment and install dependencies:

conda create --name finchat python=3.9
conda activate finchat
pip install -r requirements.txt

Note: If you haven't created a requirements.txt yet, include the following packages:

streamlit
yfinance
PyPDF2
langchain
transformers
chromadb
sentence-transformers
pytest

Usage
Running the Chatbot Locally

To start the chatbot via Streamlit, run:

streamlit run app.py

Open your web browser and navigate to the URL provided (typically http://localhost:8501).
File Upload & Local Data

    Uploads: Use the sidebar to upload PDFs, images, or CSV files containing financial data.
    Local Fallback: If no file is uploaded, the app loads a default document from D:\genAI\data\document.pdf.

Querying

Type your financial query in the input field. For queries mentioning "stock price," the app will attempt to fetch live data using yfinance.
Testing
Unit Tests

Run unit tests using Pytest:

pytest

This will run tests in the tests directory to validate functions such as text extraction and CSV loading.
End-to-End Testing

(Optional) Set up end-to-end tests with Playwright or Cypress to simulate user interactions with the Streamlit app.
Load Testing

(Optional) Use Artillery or JMeter for load testing to simulate high traffic and monitor performance.
Deployment
Docker Deployment

Create a Docker image using the provided Dockerfile:

# Dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]

Build and run the Docker image:

docker build -t chatbot-app .
docker run -p 8501:8501 chatbot-app

Kubernetes Deployment

For scalable cloud-native deployment, create a Kubernetes deployment and service YAML file. For example, to use Kubernetes with auto-scaling:

apiVersion: apps/v1
kind: Deployment
metadata:
  name: chatbot-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chatbot
  template:
    metadata:
      labels:
        app: chatbot
    spec:
      containers:
      - name: chatbot
        image: chatbot-app:latest
        ports:
        - containerPort: 8501
---
apiVersion: v1
kind: Service
metadata:
  name: chatbot-service
spec:
  selector:
    app: chatbot
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer

Apply the configuration with:

kubectl apply -f deployment.yaml

Continuous Monitoring & Maintenance

    Monitoring: Use Prometheus and Grafana (or Datadog) to monitor application performance.
    Model Monitoring: Implement MLflow or EvidentlyAI to monitor model drift and response times.
    Automated Maintenance: Set up Kubernetes CronJobs for scheduled tasks like model retraining and backups.

Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
License


##Contact

For any questions or feedback, please reach out at aaaditya199@gmail.com


