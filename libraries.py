import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import uvicorn
from model.functions import combined_loss
from io import BytesIO
from PIL import Image
from fastapi.responses import StreamingResponse
