from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAVDESS_DIR = DATA_DIR / "ravdess_actors"

# Data parameters
SAMPLE_RATE = 44100
DURATION = 4.0  # seconds
N_MELS = 128