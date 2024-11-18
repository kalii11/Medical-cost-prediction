

# Medical Cost Prediction App

Welcome to the **Medical Cost Prediction App**! This application leverages machine learning to predict medical insurance costs based on user input. It is designed for healthcare professionals and individuals seeking to understand potential medical expenses. The app features an intuitive interface for estimating charges based on factors such as age, sex, BMI, smoking status, number of children, and region.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

## Features

- **User Input:** Simple form for entering details like age, sex, BMI, smoking status, number of children, and region.
- **Prediction:** Instant medical cost estimates based on user input.
- **Interactive Visualizations:** Graphs that display prediction distributions.
- **Responsive Design:** Works seamlessly on both desktop and mobile devices.
- **Educational Insights:** Additional information on the factors influencing medical costs.

## Technologies Used

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask (for serving the machine learning model)
- **Machine Learning Libraries:** Scikit-learn, Pandas, NumPy
- **Data Visualization:** Plotly, Matplotlib
- **Deployment:** Docker (`medicalcost` image), Heroku (or other cloud services)
- **Version Control:** Git, GitHub

## Installation

To get started with the Medical Cost Prediction App:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/kalii11/Medical-Cost-main.git
    ```

2. **Navigate into the project directory**:
    ```bash
    cd Medical-Cost-main
    ```

3. **Install dependencies**:
    Use a virtual environment for managing dependencies:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    pip install -r libraries.txt
    ```

4. **Run the application**:
    ```bash
    python app.py
    ```

## Docker Setup

To run the app using Docker:

1. **Build the Docker image**:
    ```bash
    docker build -t medicalcost .
    ```

2. **Run the Docker container**:
    ```bash
    docker run -p 5555:5555 medicalcost
    ```

## Usage

1. **Input Data**: Enter age, sex, BMI, smoking status, number of children, and region.
2. **Get Prediction**: Click the "Predict" button to receive an estimate of medical charges.
3. **View Results**: The app displays the predicted medical cost and associated graphs.

## How It Works

The Medical Cost Prediction App is based on a machine learning model trained with data containing age, sex, BMI, smoking status, number of children, region, and insurance charges. The user input is processed to generate a medical cost prediction.

## Model Training

The training process includes:

1. **Data Preprocessing**: Cleaning and encoding input features.
2. **Model Selection**: Training regression models (e.g., Linear Regression, Random Forest).
3. **Evaluation**: Tuning hyperparameters for optimal performance.

## Contributing

To contribute:

1. Fork the repository.
2. Create a branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License.

## Contact Information

For inquiries, please contact:

- **Name:** IMLOUL DOUAE
- **Email:** douaeimloul@gmail.com

- **Name:** RACHIDI WIDAD
- **Email:** widadrachidi438@gmail.com

