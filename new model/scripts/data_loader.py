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


def load_emotionally_dataset(data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the EmotionAlly dataset from the specified path.
    
    Args:
        data_path: Path to the EmotionAlly dataset directory
        
    Returns:
        DataFrame containing audio file paths and labels
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
    
    # Initialize lists for data collection
    file_paths = []
    emotions = []
    
    # Map folder names to standardized emotion labels
    emotion_map = {
        'Angry': 'angry',
        'Disgust': 'disgust', 
        'Fear': 'fear',
        'Happy': 'happy',
        'Neutral': 'neutral',
        'Sad': 'sad',
        'Surprise': 'surprise'
    }
    
    # Walk through emotion directories
    for emotion_folder in data_path.iterdir():
        if emotion_folder.is_dir() and emotion_folder.name in emotion_map:
            emotion_label = emotion_map[emotion_folder.name]
            
            # Get all .wav files in this emotion folder
            for audio_file in emotion_folder.glob("*.wav"):
                file_paths.append(str(audio_file))
                emotions.append(emotion_label)
    
    # Create DataFrame
    df = pd.DataFrame({
        'path': file_paths,
        'emotion': emotions
    })
    
    return df


def load_combined_datasets(ravdess_path: Union[str, Path], emotionally_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and combine both RAVDESS and EmotionAlly datasets.
    
    Args:
        ravdess_path: Path to the RAVDESS dataset directory
        emotionally_path: Path to the EmotionAlly dataset directory
        
    Returns:
        Combined DataFrame containing audio file paths and labels from both datasets
    """
    # Load individual datasets
    ravdess_df = load_ravdess_dataset(ravdess_path)
    emotionally_df = load_emotionally_dataset(emotionally_path)
    
    # Add dataset source column
    ravdess_df['dataset'] = 'RAVDESS'
    emotionally_df['dataset'] = 'EmotionAlly'
    
    # Combine datasets
    combined_df = pd.concat([ravdess_df, emotionally_df], ignore_index=True)
    
    return combined_df