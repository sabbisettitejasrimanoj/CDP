
# 🌿 Crop Disease Prediction System

This project is a Crop Disease Prediction Web Application developed using Python and HTML.
The system helps identify crop diseases by analyzing crop leaf images using a trained machine learning model.

Farmers or users can upload an image of a crop leaf, and the system will predict the disease and display the result on the webpage.

## 🚀 Features

- Upload crop leaf images
- Predict crop diseases using Machine Learning
- Simple and user-friendly interface
- Fast prediction results
- Web-based application

## 🛠️ Technologies Used

- Python
- HTML
- CSS
- Flask
- Machine Learning

## 📁 Project Structure

crop-disease-prediction
│
├── static
│   └── style.css
│
├── templates
│   ├── index.html
│   └── result.html
│
├── model
│   └── crop_disease_model.pkl
│
├── app.py
├── requirements.txt
└── README.md

## ⚙️ Installation

1. Clone the repository

git clone https://github.com/your-username/crop-disease-prediction.git
cd crop-disease-prediction

2. Create a virtual environment

python -m venv venv

3. Activate the virtual environment

Windows:
venv\Scripts\activate

Linux / Mac:
source venv/bin/activate

4. Install required libraries

pip install -r requirements.txt

## ▶️ Run the Application

python app.py

Then open your browser and go to:

http://127.0.0.1:5000

## 📸 How It Works

1. User uploads a crop leaf image
2. Image is sent to the Python backend
3. The machine learning model analyzes the image
4. The system predicts the disease
5. The result is displayed on the webpage

## 🌾 Applications

- Smart farming
- Agriculture monitoring
- Disease detection in crops
- Farmer assistance systems

## 👨‍💻 Author

Manoj

## 📜 License

This project is licensed under the MIT License.
