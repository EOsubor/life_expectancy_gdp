
# Life Expectancy and GDP Analysis

## Overview

This project aims to explore the relationship between GDP and life expectancy at birth, and to provide a tool for predicting life expectancy based on GDP input. The analysis is visualized through an interactive Streamlit app that allows users to view data, understand model performance, and make predictions.

## Features

- **Data Visualization:** Scatter plots of GDP vs. Life Expectancy for different countries.
- **Statistical Summary:** Summary statistics of the dataset.
- **Model Performance:** Display of Mean Squared Error and R-squared metrics for the linear regression model.
- **Interactive Prediction:** Users can input GDP values to get predicted life expectancy.
- **User-Friendly Number Formatting:** Large GDP values are formatted with commas for better readability.

## Installation

1. **Clone the repository:**

   \`\`\`bash
   git clone https://github.com/your-username/life-expectancy-gdp.git
   cd life-expectancy-gdp
   \`\`\`

2. **Create a virtual environment:**

   \`\`\`bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use \`venv\Scripts\activate\`
   \`\`\`

3. **Install the dependencies:**

   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## Usage

1. **Ensure the dataset is in the correct location:**

   The dataset \`all_data.csv\` should be in the same directory as the \`life_expectancy_gdp_streamlit.py\` file.

2. **Run the Streamlit app:**

   \`\`\`bash
   streamlit run life_expectancy_gdp_streamlit.py
   \`\`\`

3. **Interact with the app:**

   Open the provided URL (usually \`http://localhost:8501\`) in your web browser to interact with the app.

## Project Structure

- \`life_expectancy_gdp_streamlit.py\`: The main Python script that runs the Streamlit app.
- \`all_data.csv\`: The dataset containing GDP and life expectancy information.
- \`requirements.txt\`: A list of required Python packages.

## Dataset

The dataset used in this project (\`all_data.csv\`) includes the following columns:

- \`Country\`: The name of the country.
- \`Year\`: The year of the data.
- \`Life expectancy at birth (years)\`: The average life expectancy at birth.
- \`GDP\`: The Gross Domestic Product in USD.

## App Functionality

### Dataset Display

The app displays the first few rows of the dataset and provides summary statistics, giving users an initial understanding of the data.

### Data Visualization

Scatter plots are used to visualize the relationship between GDP and life expectancy, with different colors representing different countries.

### Model Training

A linear regression model is trained using GDP as the predictor and life expectancy as the response variable. The model is evaluated using Mean Squared Error and R-squared metrics.

### Predictive Analysis

Users can input a GDP value to predict the corresponding life expectancy. The input GDP value is formatted for readability, and the predicted life expectancy is displayed.

## Example Usage

1. **View Dataset and Statistics:**
   ![Dataset and Statistics](screenshot_dataset_statistics.png)

2. **View Scatter Plot:**
   ![Scatter Plot](screenshot_scatter_plot.png)

3. **Model Performance:**
   ![Model Performance](screenshot_model_performance.png)

4. **Predict Life Expectancy:**
   ![Prediction](screenshot_prediction.png)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.

## Contact

For any questions or inquiries, please contact [your-email@example.com].
