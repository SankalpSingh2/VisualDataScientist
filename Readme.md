# Ollama Plot Generator and Analyzer

This application generates random plots and uses Ollama's VLM models to analyze and describe them or answer questions about the plot.

## Features

- Generate random scatter plots, line plots, and bar charts
- Choose between 1D and 2D data representations
- Connect to a local or remote Ollama server
- Select from available Ollama models
- Convert the plot to a base64 image
- Send user prompt and the converted plot to the Ollama VLM model
- Get AI-generated descriptions of the plots or answers to questions about the plot

## Requirements

- Python 3.10+
- Ollama server
- VLM models such as [MiniCPM-V](https://ollama.com/library/minicpm-v) or [LLaVA](https://ollama.com/library/llava) or any other VLM model that supports image inputs.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ollama-plot-generator.git
   cd ollama-plot-generator
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Pull a few VLM models and make sure Ollama server is running.

2. Run the application:
   ```
   python app.py
   ```

3. Open your web browser and navigate to the URL displayed in the console (usually `http://localhost:7860`).

4. Use the interface to generate plots and get AI-generated descriptions.

## How It Works

1. **Ollama Connection**: The app connects to an Ollama server (default: `http://localhost:11434`) and fetches available models.

2. **Plot Generation**: Users can select the plot type (scatter, line, or bar) and data dimension (1D or 2D). Random data is generated and plotted using Matplotlib.

3. **Image Processing**: The generated plot is saved as an image file and converted to base64 format for API communication.

4. **AI Description**: The app sends the plot image along with a user-provided prompt to the selected Ollama model via API. The model generates a description of the plot.

5. **User Interface**: Gradio is used to create an interactive web interface, allowing users to control plot generation and view results in real-time.

## Assignment
1. Run the repo on your machine with Llava and Minicpm-v models.
2. Modify the code to receive a single column CSV file (`example.csv` provided) and produce qualitative analysis on the time-series data in the CSV by creating plots from smaller segments of the data and identifying whether the segment contains one or more of these patterns using the VLM model:
    - Oscillatory/periodic behavior
    - Rising trend
    - Falling trend
    - Flat
    - Irregular
3. Use the LLM to summarize all observations (the output of part 2) to produce a report of the entire time-series

Bonus:
Instead of using Ollama VLM models, try out other cloud-based multimodal models such as Gemini, GPT-4o, or other models that support image inputs.
