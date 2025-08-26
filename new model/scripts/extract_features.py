import numpy as np
import librosa
import parselmouth


def extract_mfcc(path):
    ''' 
    Load the audio file, convert the audio file into MFCCs and return the MFCCs
    '''
    # Load the audio file and set the sampling rate to 44100
    audio, sr = librosa.load(path, sr=44100, duration=4, mono=True)
    # Pad the audio files that are less than 4 seconds with zeros at the end
    if len(audio) < 4 * sr:
        audio = np.pad(audio, pad_width=(0, 4 * sr - len(audio)), mode='constant')
    # Convert the audio file into MFCC
    signal = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
    # Return the MFCCs as a numpy array
    return np.array(signal)

def extract_prosody_features(path):
    """Extract prosody features from audio file"""
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values > 0]  # Remove unvoiced frames

    # Pitch statistics
    pitch_mean = np.mean(f0_values) if len(f0_values) > 0 else 0
    pitch_min = np.min(f0_values) if len(f0_values) > 0 else 0
    pitch_max = np.max(f0_values) if len(f0_values) > 0 else 0
    pitch_var = np.var(f0_values) if len(f0_values) > 0 else 0

    # Energy (RMS)
    rms = np.sqrt(np.mean(snd.values**2))

    # Duration
    duration = snd.get_total_duration()

    # Formants (F1, F2, F3) at mid-point
    formant = snd.to_formant_burg()
    t = duration / 2
    f1 = formant.get_value_at_time(1, t)
    f2 = formant.get_value_at_time(2, t)
    f3 = formant.get_value_at_time(3, t)

    return {
        "pitch_mean": pitch_mean,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_var": pitch_var,
        "rms": rms,
        "duration": duration,
        "formant1": f1,
        "formant2": f2,
        "formant3": f3
    }