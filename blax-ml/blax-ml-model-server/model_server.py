import argparse
import base64
import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--share",
        action="store_true",
    )
    parser.add_argument(
        "--port",
        default=7860,
        type=int,
    )
    return parser.parse_args()


def _create_gradio_interface(model_dir):
    pipeline, metadata = _load_pipeline_and_metadata(model_dir)

    task_type = metadata["task_type"]
    features = metadata["features"]

    components = []
    for feature in features:
        name = feature["name"]
        feature_type = feature["type"]
        if feature_type == "numeric":
            min_val = feature["min_value"]
            max_val = feature["max_value"]
            component = gr.Textbox(
                placeholder="Leave blank to impute automatically",
                label=name,
                info=f"Training range: {min_val} to {max_val}",
            )
        else:
            choices = feature["possible_values"]
            if choices and len(choices) < 50:
                component = gr.Dropdown(
                    choices=choices,
                    value=choices[0],
                    filterable=True,
                    label=name,
                    info=f"Available options: {len(choices)}",
                )
            else:
                component = gr.Textbox(
                    placeholder="Leave blank to impute automatically",
                    label=name,
                    info="Leave blank to impute",
                )
        components.append(component)

    def predict(*inputs):
        input_dict = {}
        for value, feature in zip(inputs, features):
            name = feature["name"]
            if value is None or str(value).strip() == "":
                input_dict[name] = [None]
            elif feature["type"] == "numeric":
                try:
                    input_dict[name] = [float(value)]
                except (ValueError, TypeError):
                    input_dict[name] = [None]
            else:
                input_dict[name] = [value]

        df = pd.DataFrame(input_dict)

        try:
            if task_type == "classification":
                probabilities = pipeline.predict_proba(df)[0]
                max_probability = probabilities.argmax()
                classes = pipeline.classes_
                return {
                    "predicted": classes[max_probability],
                    "confidence": f"{probabilities[max_probability] * 100:.2f}%",
                    "probabilities": {
                        str(cls): f"{prob * 100:.2f}%"
                        for cls, prob in zip(classes, probabilities)
                    },
                }
            elif task_type == "regression":
                return {"predicted": round(float(pipeline.predict(df)[0]), 4)}
        except Exception as e:
            return {"error": str(e)}

    with gr.Blocks() as app:
        gr.Markdown(
            """
            <div
                style="
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                "
            >
                <div style="display: flex; align-items: center; gap: 15px">
                    <img
                        src="https://github.com/aniketppanchal/mcp-1st-birthday-hackathon-assets/raw/refs/heads/main/blax-ml/logo.png"
                        width="150px"
                    />
                    <div>
                        <h1 style="font-size: 40px; margin: 0">BlaxML</h1>
                        <span>
                            Autonomous Machine Learning, Powered by
                            <a
                                href="https://blaxel.ai/"
                                target="_blank"
                                style="color: #ff8b3d; text-decoration: none"
                            >
                                Blaxel
                            </a>
                        </span>
                    </div>
                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(_generate_model_report_markdown(model_dir, metadata))

            with gr.Column(scale=2):
                gr.Markdown("# 🎯 Make a Prediction")

                with gr.Group():
                    for component in components:
                        component.render()
                submit = gr.Button("🚀 Predict", variant="primary")

                gr.Markdown("## 📊 Result")
                output = gr.JSON()

                submit.click(fn=predict, inputs=components, outputs=output)

    return app


def _load_pipeline_and_metadata(model_dir):
    pipeline_path = model_dir / "pipeline.joblib"
    pipeline_meta_path = model_dir / "pipeline_meta.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Model dir '{model_dir}' is missing required file: 'pipeline.joblib'"
        )
    if not pipeline_meta_path.exists():
        raise FileNotFoundError(
            f"Model dir '{model_dir}' is missing required file: 'pipeline_meta.json'"
        )

    pipeline = joblib.load(pipeline_path)

    with open(pipeline_meta_path, "r") as file:
        metadata = json.load(file)

    return pipeline, metadata


def _generate_model_report_markdown(model_dir, metadata):
    task_type = metadata["task_type"]
    target_column = metadata["target_column"]
    features = metadata["features"]
    metrics = metadata["metrics"]
    args = metadata["args"]

    markdown = "# 📈 Model Performance\n\n"

    markdown += "## 1. Overview\n\n"
    markdown += f"- **Task Type:** {task_type.title()}\n\n"
    markdown += f"- **Target Column:** `{target_column}`\n\n"

    markdown += "## 2. Evaluation Metrics\n\n"
    if task_type == "regression":
        markdown += (
            "| Metric    | Value                                |\n"
            "|-----------|--------------------------------------|\n"
            f"| R² Score | {metrics['r2_score']}                |\n"
            f"| MAE      | {metrics['mean_absolute_error']}     |\n"
            f"| MSE      | {metrics['mean_squared_error']}      |\n"
            f"| RMSE     | {metrics['root_mean_squared_error']} |\n"
            "\n\n"
        )
    elif task_type == "classification":
        markdown += (
            "| Metric     | Value                   |\n"
            "|------------|-------------------------|\n"
            f"| Accuracy  | {metrics['accuracy']}%  |\n"
            f"| Precision | {metrics['precision']}% |\n"
            f"| Recall    | {metrics['recall']}%    |\n"
            f"| F1 Score  | {metrics['f1_score']}%  |\n"
            "\n\n"
        )

    markdown += "## 3. Visualizations\n\n"
    images = []
    if task_type == "regression":
        images.append(("Prediction Error", Path(model_dir / "prediction_error.png")))
    elif task_type == "classification":
        images.append(("Confusion Matrix", Path(model_dir / "confusion_matrix.png")))
    for idx, (title, image) in enumerate(images, start=1):
        if image.exists():
            markdown += f"### {idx}. {title}\n\n"
            b64 = base64.b64encode(image.read_bytes()).decode("utf-8")
            markdown += f"![{title}](data:image/png;base64,{b64})\n\n"

    markdown += "## 4. Features\n\n"
    numeric = [col for col in features if col["type"] == "numeric"]
    categorical = [col for col in features if col["type"] == "categorical"]
    for title, items in [
        ("Numeric", numeric),
        ("Categorical", categorical),
    ]:
        if items:
            markdown += f"**{title} ({len(items)}):**\n\n"
            markdown += "\n".join(f"- `{col['name']}`" for col in items)
            markdown += "\n\n"

    markdown += "## 5. Training Args\n\n"
    markdown += "```json\n"
    markdown += json.dumps(args, indent=2)
    markdown += "\n```\n\n"
    return markdown


if __name__ == "__main__":
    args = _parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir '{model_dir}' does not exist")
    if not any(model_dir.iterdir()):
        raise FileNotFoundError(f"Model dir '{model_dir}' is empty")

    app = _create_gradio_interface(model_dir)
    app.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
