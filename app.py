import shutil

import gradio as gr
import requests
import json
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import os
import base64
import pandas as pd

# Default Ollama URL
DEFAULT_OLLAMA_URL = 'http://localhost:11434'

def check_ollama_connection(ollama_url: str) -> bool:
    try:
        response = requests.get(f'{ollama_url}/api/tags', timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Error connecting to Ollama server: {e}")
        return False

def get_ollama_models(ollama_url: str) -> List[str]:
    try:
        response = requests.get(f'{ollama_url}/api/tags', timeout=5)
        response.raise_for_status()
        models = [model['name'] for model in response.json()['models']]
        return models
    except requests.RequestException as e:
        print(f"Error fetching models from Ollama server: {e}")
        return []

def generate_output(ollama_url: str, model: str, prompt: str, options: Dict[str, Any], images: Optional[List[str]] = None) -> Optional[requests.Response]:
    try:
        data = {
            'model': model,
            'prompt': prompt,
            'options': options
        }
        if images is not None:
            data['images'] = images
        response = requests.post(f'{ollama_url}/api/generate', json=data, stream=True, timeout=300)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

def generate_data_and_plot(plot_type: str, num_points: str):
    uploads_dir = "/Users/sanky/Work/VisualDataScientist/uploads"
    all_files = os.listdir(uploads_dir)
    csv_files = [f for f in all_files if f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the uploads directory.")

    first_csv_file = os.path.join(uploads_dir, csv_files[0])
    data = pd.read_csv(first_csv_file, header=None).squeeze("columns")
    data = pd.to_numeric(data, errors='coerce').dropna()

    num_points2 = int(num_points)
    total_points = len(data)
    if num_points2 > total_points:
        num_points2 = total_points

    x = np.arange(num_points2)
    y = np.interp(np.linspace(0, total_points - 1, num_points2), np.arange(total_points), data)
    num_plots = 8
    points_per_plot = len(x) // num_plots

    file_paths = []
    for i in range(num_plots):
        start = i * points_per_plot
        end = start + points_per_plot
        x_subset = x[start:end]
        y_subset = y[start:end]

        plt.figure(figsize=(5, 3))
        plt.ylim(-4, 4)
        if plot_type == 'scatter-plot':
            plt.scatter(x_subset, y_subset)
        elif plot_type == 'line-plot':
            plt.plot(x_subset, y_subset, marker='o', linestyle='-', color='b')
        elif plot_type == 'bar-chart':
            plt.bar(x_subset, y_subset)
        else:
            plt.plot(x_subset, y_subset, marker='o', linestyle='-', color='b')
        plt.xlabel('Data Point Index (Time)')
        plt.ylabel('Values')

        filename = f'plot_{i + 1}.png'
        save_path = os.path.abspath(filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        file_paths.append(save_path)

    return file_paths, file_paths


def generate_description(ollama_url: str, model: str, user_prompt: str, file_paths: list, temperature: float, top_k: int, top_p: float,):
    images = []
    try:
        for image_path in file_paths:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            images.append([image_base64])  # Each image in its own list as expected by Ollama API
    except Exception as e:
        yield f"Error reading image: {e}"
        return
# test comment
    prompt = user_prompt
    options = {
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'stop': None,
        'num_ctx': 32768
    }

    accumulated_response = ""
    max_lines = 500  # Set a reasonable limit on lines to read

    for idx, image in enumerate(images):
        response = generate_output(ollama_url, model, prompt, options, images=image)
        if response:
            full_response = ''
            for i, line in enumerate(response.iter_lines()):
                if i >= max_lines:  # Break if line limit is reached
                    print("Warning: Maximum line limit reached. Terminating response.")
                    break
                if line:
                    json_response = json.loads(line.decode('utf-8'))
                    if 'response' in json_response:
                        partial_response = json_response['response']
                        full_response += partial_response

            plot_response = f"Plot {idx + 1}: {full_response}"
            accumulated_response += plot_response + "\n\n"
            yield accumulated_response  # Yield accumulated responses so far


def upload_file(file):
    UPLOAD_FOLDER = "/Users/sanky/Work/VisualDataScientist/uploads"
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    for f in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, f)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    dest_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file))
    shutil.copy(file, dest_path)
    return gr.Info("File Uploaded. Now you can query the document.")

def main():
    with gr.Blocks() as demo:
        initial_models: List[str] = get_ollama_models(DEFAULT_OLLAMA_URL)
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("### Controls")
                ollama_url = gr.Textbox(label="🌐 Ollama Server URL", value=DEFAULT_OLLAMA_URL, interactive=True)
                model_dropdown = gr.Dropdown(label="🧠 Ollama Model", choices=initial_models, value=initial_models[0] if initial_models else "", interactive=True)
                plot_type_dropdown = gr.Dropdown(label="📊 Plot Type", choices=['scatter-plot', 'line-plot', 'bar-chart'], value='scatter-plot', interactive=True)
                num_points = gr.Textbox(label="Number of points", value='', interactive=True)
                # 'temperature': 1.0,
                #         'top_k': 40,
                #         'top_p': 1.0,
                #         'stop': None,
                #         'num_ctx': 32768
                Temperature = gr.Slider(label="Temperature", value = '1', minimum=0, maximum=1, step=0.1, interactive=True)
                TopK = gr.Slider(label="Top K", value='50', minimum=0, maximum=100, step=10, interactive=True)
                TopP = gr.Slider(label="Top P", value='1', minimum=0, maximum=1, step=0.1, interactive=True)
                user_prompt = gr.UploadButton("Click to upload a file")
                user_prompt.upload(upload_file, user_prompt)
                generate_data_button = gr.Button("Generate Data")
                pattern_prompt = gr.Textbox(label="📝 Your Prompt (Default is to find patterns)", value="Identify whether the 8 segment in each of the 8 plots in the given gallery contains one of the following patterns:\nOscillatory / periodic behavior\nRising trend\nFalling trend\nFlat\nIrregular")
                generate_description_button = gr.Button("Generate Description")

            with gr.Column(scale=3):
                with gr.Row():
                    plot_gallery = gr.Gallery(label="Generated Plots", height=400)
                    description_output = gr.Textbox(label="🗒️ Model's Description", placeholder="Model's description will appear here...", lines=20)

        image_path_state = gr.State()

        def update_models(ollama_url: str) -> Dict[str, Any]:
            models = get_ollama_models(ollama_url)
            if models:
                first_model = models[0]
            else:
                first_model = ""
            return gr.Dropdown.update(choices=models, value=first_model)

        ollama_url.change(update_models, inputs=[ollama_url], outputs=[model_dropdown])

        generate_data_button.click(generate_data_and_plot, inputs=[plot_type_dropdown, num_points], outputs=[plot_gallery, image_path_state])
        # Update the button click event to use the modified generate_description function
        generate_description_button.click(generate_description,
                                          inputs=[ollama_url, model_dropdown, pattern_prompt, image_path_state, Temperature, TopK, TopP],
                                          outputs=[description_output], queue=True)

    if __name__ == "__main__":
        demo.queue().launch()

main()
