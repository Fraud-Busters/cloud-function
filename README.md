# Fraud Busters Cloud Function

Fraud Busters is a fraud detection system consisting of three components: frontend, backend, and a cloud function for executing a machine learning model. The system is designed to detect fraudulent activities in uploaded CSV files. The machine learning model is deployed using a cloud function triggered by the "Object Finalized" event in a cloud storage service. The frontend handles file uploads, which are then processed by the backend and stored in the cloud storage. The cloud function, running on Python 3.11.1 runtime, checks if the uploaded file's name ends with "\_in.csv" and triggers the machine learning model to analyze and predict the data. Additionally, the function updates relevant data status in a MongoDB database.

## Table of Contents

- [Fraud Busters Cloud Function](#fraud-busters-cloud-function)
  - [Table of Contents](#table-of-contents)
  - [System Architecture](#system-architecture)
  - [Components](#components)
    - [Frontend](#frontend)
    - [Backend](#backend)
    - [Cloud Function](#cloud-function)
  - [Installation and Deployment](#installation-and-deployment)
  - [Configuration](#configuration)
  - [Usage](#usage)

## System Architecture

The Fraud Busters system consists of three main components: the frontend, backend, and cloud function. The architecture can be visualized as follows:

```
        +----------------+
        |                |
        |    Frontend    |
        |                |
        +-------+--------+
                |
                | File Upload
                |
        +-------v--------+
        |                |
        |    Backend     |
        |                |
        +-------+--------+
                |
                | Store File in Cloud Storage
                |
        +-------v--------+
        |                |
        | Cloud Function |
        |                |
        +-------+--------+
                |
                | Triggered by "Object Finalized" Event
                |
        +-------v--------+
        |                |
        |  Machine       |
        |  Learning      |
        |  Model         |
        |                |
        +----------------+
```

## Components

### Frontend

The frontend component is responsible for providing a user interface for uploading CSV files. Users can interact with the frontend to select and upload files to the system. It communicates with the backend to process the uploaded files.

### Backend

The backend component handles file uploads from the frontend and performs necessary data processing tasks. It receives the uploaded file, saves it in a cloud storage service, and updates the MongoDB database with relevant data status. The backend also communicates with the cloud function to trigger the machine learning model for fraud detection.

### Cloud Function

The cloud function component is responsible for executing the machine learning model when triggered by the "Object Finalized" event in the cloud storage service. The function checks if the uploaded file's name ends with "\_in.csv" and proceeds to analyze and predict the data using the machine learning model. It also updates the relevant data status in the MongoDB database.

## Installation and Deployment

To install and deploy the Fraud Busters system, follow these steps:

1. Set up the frontend:

   - Clone the frontend repository from [GitHub](https://github.com/Fraud-Busters/frontend.git).
   - Install the required dependencies.
   - Configure the backend API endpoint.

2. Set up the backend:

   - Clone the backend repository from [GitHub](https://github.com/Fraud-Busters/backend.git).
   - Install the required dependencies.
   - Configure the cloud storage service and MongoDB connection details.

3. Set up the cloud function:
   - Create a new cloud function in your cloud provider's environment.
   - Add MONGO_URI as environment variable in your configuration.
   - Use Python 3.11.1 as the runtime for the function.
   - Upload this repo as zip file.

- Install the required dependencies.
- Configure the cloud storage service, MongoDB connection details, and machine learning model integration.

1. Deploy the frontend, backend, and cloud function to their respective environments.
   - Ensure that the backend API endpoint is correctly configured in the frontend.
   - Verify that the cloud function is set up to trigger on the "Object Finalized" event in the cloud storage service.

## Configuration

The Fraud Busters system requires the following configurations:

- Frontend:

  - Backend API endpoint: Set the endpoint URL in the frontend configuration to communicate with the backend.

- Backend:

  - Cloud storage service: Configure the backend to connect to the appropriate cloud storage service (e.g., Amazon S3, Google Cloud Storage) and provide the necessary credentials.
  - MongoDB connection: Set the MongoDB connection details (hostname, port, username, password) in the backend configuration.

- Cloud Function:
  - Cloud storage service: Configure the cloud function to connect to the same cloud storage service used by the backend and provide the necessary credentials.
  - MongoDB connection: Set the MongoDB connection details (hostname, port, username, password) in the cloud function configuration.
  - Machine learning model integration: Integrate the machine learning model into the cloud function and configure it to analyze and predict the data.

## Usage

To use the Fraud Busters system, follow these steps:

1. Access the frontend user interface.
2. Upload a CSV file containing the data to be analyzed for fraud detection.
3. Monitor the backend for file processing and database updates.
4. Once the file upload is complete, the cloud function will be triggered automatically.
5. The cloud function will analyze and predict the data using the machine learning model.
6. The MongoDB database will be updated with the results of the analysis.
7. Access the MongoDB database to view the updated data status and any detected fraudulent activities.
