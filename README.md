# TSForecasting_InvoiceGPT

A comprehensive time series forecasting system for invoice data, featuring AI-powered synthetic data generation and advanced forecasting models.

## Overview

**TSForecasting_InvoiceGPT** is a project that combines synthetic invoice data generation using Large Language Models (LLMs) with time series forecasting capabilities. The system uses Google Gemini AI to generate realistic invoice data and applies various deep learning models (LSTM, N-BEATS, Ensemble methods) to forecast future invoice trends.

## Features

### 🤖 AI-Powered Synthetic Data Generation
- **LangGraph-based workflow**: Orchestrates complex data generation pipelines
- **Parallel processing**: Generates multiple invoice samples concurrently for efficiency
- **Category-specific patterns**: Generates invoices for 37+ expense categories with realistic frequency patterns
- **Context-aware generation**: Creates invoices tailored for upper-middle class Indian context (Chennai, India)
- **Rate limiting**: Built-in rate limiting for API calls to manage costs

### 📊 Time Series Forecasting
- **Multiple models**: Implements various forecasting approaches:
  - Naive forecasting (baseline)
  - LSTM (Long Short-Term Memory) networks
  - N-BEATS (Neural Basis Expansion Analysis for Time Series)
  - Ensemble methods
- **Model comparison**: Comprehensive evaluation and visualization
- **Prediction intervals**: Provides uncertainty quantification for forecasts

### 🔄 Workflow Orchestration
- **LangGraph integration**: Visual workflow graphs for data generation
- **State management**: Robust state handling for complex multi-step processes
- **Error handling**: Comprehensive error tracking and recovery

## Installation

### Prerequisites
- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YohanV1/InvoiceGPT-TS-Forecasting.git
   cd InvoiceGPT-TS-Forecasting
   ```

2. **Install dependencies using uv**
   ```bash
   uv pip install -e .
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_FLASH_MODEL_NAME=gemini-2.0-flash
   GEMINI_TEMPERATURE=1.0
   OUTPUT_DIR=generated_data
   SAMPLES_PER_CATEGORY=10
   GEMINI_FLASH_RATE_LIMIT=1000
   ```

## Configuration

The project uses environment variables for configuration. Key settings in `config.py`:

- **GEMINI_API_KEY**: Your Google Gemini API key (required)
- **GEMINI_FLASH_MODEL_NAME**: Model name (default: `gemini-2.0-flash`)
- **GEMINI_TEMPERATURE**: Model temperature (default: `1.0`)
- **OUTPUT_DIR**: Directory for generated CSV files (default: `generated_data`)
- **SAMPLES_PER_CATEGORY**: Number of samples per expense category (default: `10`)
- **GEMINI_FLASH_RATE_LIMIT**: API rate limit in requests per minute (default: `1000`)

## Usage

### 1. Generate Synthetic Invoice Data

```python
from synthetic_data import SyntheticDataGenerator

# Initialize the generator
generator = SyntheticDataGenerator()

# Generate data for specific categories
categories = ["Groceries", "Housing", "Transportation"]
results = generator.generate_for_categories(categories, samples_per_category=50)

# Or generate for all categories
results = generator.generate_all(samples_per_category=50)

# Save graph visualizations
generator.save_graph_visualization()
```

### 2. Run Time Series Forecasting

Open the Jupyter notebooks:
- `TSForecasting_InvoiceGPT.ipynb`: Main forecasting notebook
- `10_time_series_forecasting_in_tensorflow.ipynb`: TensorFlow-based forecasting

The notebooks include:
- Data loading and preprocessing
- Model training and evaluation
- Visualization of results
- Model comparison metrics

### 3. Command Line Usage

```bash
# Run synthetic data generation
python synthetic_data.py
```

## Project Structure

```
TSForecasting_InvoiceGPT/
├── synthetic_data.py          # Main synthetic data generation module
├── chat_model.py               # LLM manager with rate limiting
├── config.py                   # Configuration management
├── TSForecasting_InvoiceGPT.ipynb    # Main forecasting notebook
├── 10_time_series_forecasting_in_tensorflow.ipynb  # TensorFlow notebook
├── generated_data/            # Output directory for CSV files
│   └── Groceries.csv          # Example generated data
├── graph_visualizations/      # LangGraph workflow visualizations
│   ├── expense_graph.png
│   └── batch_graph.png
├── pyproject.toml             # Project dependencies
├── .env                       # Environment variables (create this)
└── README.md                  # This file
```

## Key Components

### SyntheticDataGenerator
The main class for generating synthetic invoice data:
- **LangGraph workflows**: Uses state graphs for orchestration
- **Parallel processing**: ThreadPoolExecutor for concurrent generation
- **Frequency patterns**: Category-specific expense frequency modeling
- **Date generation**: Realistic date sequences based on expense patterns

### LLMManager
Singleton manager for LLM instances:
- **Rate limiting**: InMemoryRateLimiter for API call management
- **Model configuration**: Centralized model settings
- **Error handling**: Robust error management

### Expense Categories
The system supports 37+ expense categories including:
- Groceries, Housing, Dining/Restaurant
- Transportation, Healthcare, Entertainment
- Software/Subscriptions, Professional Services
- And many more...

Each category has defined frequency patterns (daily, weekly, monthly, quarterly).

## Dependencies

- **google-genai** (>=1.3.0): Google Gemini AI integration
- **langchain** (>=0.3.19): LLM framework
- **langchain-google-genai** (>=2.0.11): Google Gemini integration for LangChain
- **langgraph** (>=0.3.1): Workflow orchestration
- **pandas** (>=2.2.3): Data manipulation
- **python-dotenv** (>=1.0.1): Environment variable management

For time series forecasting (in notebooks):
- TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn

## Generated Data Format

The generated CSV files contain the following invoice fields:
- `invoice_number`: Unique invoice identifier
- `invoice_date`: Invoice issue date (YYYY-MM-DD)
- `due_date`: Payment due date
- `seller_information`: Seller details
- `buyer_information`: Buyer details
- `products_services`: Comma-separated items
- `quantities`: Comma-separated quantities
- `unit_prices`: Comma-separated unit prices
- `subtotal`: Sum before taxes
- `service_charges`: Additional charges
- `net_total`: Subtotal + service charges
- `discount`: Applied discounts
- `tax`: Tax amount
- `tax_rate`: Tax percentage
- `shipping_costs`: Delivery charges
- `grand_total`: Final amount
- `currency`: Currency code (INR, USD, etc.)
- `payment_terms`: Payment conditions
- `payment_method`: Accepted methods
- `bank_information`: Bank details
- `invoice_notes`: Additional notes
- `shipping_address`: Delivery address
- `billing_address`: Billing address

## Examples

### Generate Data for Groceries Category

```python
from synthetic_data import SyntheticDataGenerator

generator = SyntheticDataGenerator()
results = generator.generate_for_categories(["Groceries"], samples_per_category=100)

for result in results:
    print(f"Category: {result['expense_category']}")
    print(f"Samples: {result['samples_generated']}")
    print(f"API Calls: {result['api_calls']}")
```

### Visualize Workflow Graphs

```python
generator = SyntheticDataGenerator()
generator.save_graph_visualization()
# Graphs saved to graph_visualizations/
```

## Notes

- The synthetic data generator creates invoices for an upper-middle class Indian context (Chennai, India)
- Date ranges default to 2023-01-01 to 2024-12-31
- Frequency patterns are defined per category to simulate realistic expense timing
- Rate limiting is configured to prevent API quota exhaustion

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Google Gemini AI for LLM capabilities
- LangChain and LangGraph teams for workflow orchestration
- TensorFlow team for deep learning frameworks

## Contact

For questions or issues, please open an issue on the GitHub repository.

