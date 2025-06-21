# MediCare Fraud Detection System

Medicare Fraud Detection System is an AI-powered tool that identifies fraudulent healthcare providers by analyzing Medicare claim patterns using machine learning (XGBoost). It features an interactive Streamlit interface for data upload, fraud prediction, and visualization. The project leverages real-world datasets from Kaggle.

## Features

- **AI-Powered Fraud Detection:** Utilizes XGBoost to identify suspicious and fraudulent activity.
- **User-Friendly Interface:** Streamlit app for easy interaction, data upload, and predictions.
- **Data Visualization:** Visualizes claim patterns and fraud predictions.
- **Real-World Data:** Uses authentic Medicare claim datasets from Kaggle.
- **Customizable & Extensible:** Built with Python, easy for data scientists and developers to extend.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Madhav064/MediCare-fraud-detection.git
   cd MediCare-fraud-detection
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Medicare datasets:**
   - Obtain the required datasets from Kaggle ([Link to dataset](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data)).
   - Place the datasets in the appropriate directory (e.g., `data/`).

### Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
2. **Follow the on-screen instructions** to upload your data and view fraud predictions.

## Project Structure

- `app.py`: Main Streamlit application.
- `model/`: Contains machine learning models and training scripts.
- `data/`: Directory for datasets.
- `utils/`: Helper functions and utilities.
- `requirements.txt`: List of Python dependencies.

## How It Works

1. **Data Ingestion:** User uploads Medicare claim data via the web interface.
2. **Prediction:** The XGBoost model analyzes the data and predicts potential fraud.
3. **Visualization:** Results and key metrics are displayed in interactive charts.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for suggestions or improvements.

## License

This project is currently unlicensed. Please add a license if you intend to share or distribute.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for the Medicare datasets.
- Open source Python libraries: Streamlit, XGBoost, Pandas, NumPy, Matplotlib, etc.

---

*For any questions or support, please create an issue on the [GitHub repo](https://github.com/Madhav064/MediCare-fraud-detection/issues).*
