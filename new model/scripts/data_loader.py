import pandas as pd
from pathlib import Path
from typing import Union, List

def load_ravdess_dataset(data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the RAVDESS dataset from the specified path.
    
    Args:
        data_path: Path to the RAVDESS dataset directory
        
    Returns:
        DataFrame containing audio file paths and labels
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
    
    # Initialize lists for data collection
    file_paths = []
    emotions = []
    
    # Walk through directory
    for audio_file in data_path.rglob("*.wav"):
        try:
            # Parse filename for labels
            parts = audio_file.stem.split('-')
            emotion = parts[2]
            
            # Map emotion codes to labels
            emotion_map = {
                '01': 'neutral',
                '02': 'calm',
                '03': 'happy',
                '04': 'sad',
                '05': 'angry',
                '06': 'fear',
                '07': 'disgust',
                '08': 'surprise'
            }
            
            if emotion in emotion_map:
                file_paths.append(str(audio_file))
                emotions.append(emotion_map[emotion])
                
        except (IndexError, KeyError) as e:
            print(f"Skipping file {audio_file}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame({
        'path': file_paths,
        'emotion': emotions
    })
    
    return df