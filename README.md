# Wellness AI

This project generates synthetic wellness data and uses it to train an XGBoost model for machine learning applications. The data is designed to be realistic and can be used to train models for various wellness-related tasks.

## Project Structure

- `ml_wellness_xgboost.ipynb`: The main Jupyter notebook for data analysis, model training, and evaluation.
- `synthetic_data_generator.py`: A script to generate the synthetic wellness data used in the notebook.
- `synthetic-health-data/`: Directory containing the generated data (`synthetic_wellness.csv`).
- `requirements.txt`: A list of Python dependencies for the project.

## Getting Started

### Prerequisites

- Python 3.8+
- `venv` for virtual environment management

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd wellness-ai
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv wellness-env
    source wellness-env/bin/activate
    ```
    *On Windows, use `wellness-env\Scripts\activate`*

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Generate Data (Optional):**
    If `synthetic-health-data/synthetic_wellness.csv` does not exist, you can generate it by running the data generator script:
    ```bash
    python synthetic_data_generator.py
    ```

2.  **Run the Jupyter Notebook:**
    Launch Jupyter Lab to explore the analysis and model:
    ```bash
    jupyter lab
    ```
    Then, open and run the `ml_wellness_xgboost.ipynb` notebook.

## Contributing

We welcome contributions to this project. If you have an idea for a new feature or a bug fix, please open an issue on the GitHub repository.

## License

This project is licensed under the MIT License.
