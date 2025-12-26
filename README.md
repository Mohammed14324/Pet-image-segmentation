## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Mohammed14324/Pet-image-segmentation.git
cd Pet-image-segmentation
pip install -r requirements.txt
```

## Usage

Run the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 4000
```

**Note:** `main.py` is in the root folder and contains the FastAPI app object.

Make a prediction by sending a POST request to the `/predict` endpoint:

```bash
curl -X POST "http://localhost:4000/predict" -F "file=@cat_or_dog.jpg" --output result.png
```

The server returns a PNG image with the segmented pet.

## CI Pipeline

This project uses GitHub Actions for Continuous Integration. The workflow automatically installs dependencies and runs tests using pytest on every push or pull request to main. The pipeline file is located at `.github/workflows/pipeline.yaml`.
