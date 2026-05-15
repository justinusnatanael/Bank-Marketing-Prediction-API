# Bank Marketing Term Deposit Prediction with FastAPI

## Overview
This project delivers an end-to-end solution for predicting whether a client will subscribe to a term deposit following a bank marketing campaign. By leveraging a Logistic Regression classifier, the system analyzes customer demographic data and previous campaign interactions to provide accurate subscription probabilities. The project uniquely bridges the gap between machine learning research and production by serving the model through a high-performance **FastAPI** REST interface, enabling real-time integration for banking systems.

## Goals
The primary goal of this project is to showcase a complete Machine Learning lifecycle, transitioning from data exploration to API deployment:
- **Predictive Classification:** Utilizing Logistic Regression to accurately categorize customer responses to marketing efforts.
- **Advanced Preprocessing:** Implementing standardized scaling and categorical encoding pipelines to ensure high model integrity.
- **Model Serving:** Developing a robust, production-ready API using **FastAPI** and **Pydantic** for automated data validation and schema enforcement.
- **System Architecture:** Demonstrating the importance of serialized artifacts (Pickle) to maintain consistency between training and inference environments.

## Data
The analysis is conducted on a bank marketing dataset consisting of historical campaign data. The features used for prediction include:
- **Client Demographics:** Age, occupation, marital status, and educational background.
- **Financial Indicators:** Credit default history, housing loans, and personal loans.
- **Campaign Interaction:** Communication type, month, day of the week, and call duration.
- **Social/Economic Context:** Results from previous marketing outcomes (`poutcome`).
- **Target Variable:** `y` (Term deposit subscription status: Yes/No).

## Methodology
The project is divided into two distinct phases: development and deployment.

### 1. Data Preprocessing & Training
- **Data Normalization:** A `StandardScaler` was applied to numerical features to ensure that variables like 'Age' and 'Duration' contribute equally to the Logistic Regression optimizer.
- **Categorical Transformation:** Qualitative data was transformed using One-Hot Encoding (`get_dummies`) to convert categorical labels into a machine-readable binary format.
- **Train-Test Validation:** The dataset was split to validate the model's predictive power on unseen data, ensuring the model generalizes well.

---

# Modelling

## 1. Logistic Regression Classifier
The core of the prediction engine is built on Logistic Regression. This model was selected for several key reasons:
- **Interpretability:** It allows for a clear understanding of which features (e.g., call duration or job type) are the strongest drivers for customer subscription.
- **Inference Speed:** The low computational overhead makes it ideal for real-time applications where low-latency API responses are critical.
- **Reliability:** It provides a statistically sound baseline for binary classification tasks in the financial sector.

---

## 2. API Implementation (FastAPI)
To transform the model into a usable service, a REST API was developed using the FastAPI framework.

Key implementation features include:
- **Pydantic Data Models:** Strict typing via Pydantic classes to validate incoming client data before it reaches the model.
- **Automated Documentation:** Built-in Swagger UI (accessible via `/docs`) for interactive API testing and documentation.
- **Seamless Integration:** An internal processing pipeline that automatically scales and encodes raw API inputs to match the model’s expected feature vector.

---

## Results & Performance
The resulting system provides a highly efficient pipeline for bank marketing analysis.
- **Precision & Recall:** The model achieves a balanced performance, allowing the bank to focus marketing resources on customers with the highest probability of conversion.
- **Scalability:** The FastAPI implementation is capable of handling multiple concurrent requests, making it suitable for integration with larger banking CRM systems.
- **Inference Accuracy:** By using a pre-fitted `StandardScaler` during deployment, the system maintains the exact same data distribution as used during the training phase.

## Conclusion
This project successfully demonstrates how a statistical machine learning model can be operationalized. By moving from a Jupyter Notebook research environment to a structured FastAPI application, the project provides a tangible tool for banking institutions to optimize their marketing efficiency, reduce operational costs, and improve customer acquisition rates through data-driven decisions.

How to run:
uvicorn main:app --reload
