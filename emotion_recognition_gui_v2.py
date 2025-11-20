"""
Emotion Recognition GUI - Version 2
Based on CRNN_RAVDES86 copy 2.ipynb notebook logic
Interface design from emotion_recognition_gui.py
"""

import os
# Fix OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import customtkinter as ctk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
import librosa
import numpy as np
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class EmotionCRNN(nn.Module):
    """
    EmotionCRNN Model - Exact implementation from notebook
    Multi-modal architecture with MFCC (120 features) and Prosody (11 features)
    """
    def __init__(self, num_classes=8):
        super(EmotionCRNN, self).__init__()
        
        # MFCC branch (120 features: 40 MFCC + 40 Delta + 40 Delta-Delta)
        self.mfcc_conv1 = nn.Sequential(
            nn.Conv1d(120, 64, kernel_size=3, padding='same'),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        self.mfcc_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # Prosody branch (11 features)
        self.prosody_conv1 = nn.Sequential(
            nn.Conv1d(11, 32, kernel_size=3, padding='same'),  
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        self.prosody_conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # Shared layers (128 + 64 = 192 channels after concatenation)
        self.shared_conv = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.4)
        )
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm_dropout1 = nn.Dropout(0.4)
        self.lstm2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm_dropout2 = nn.Dropout(0.4)
        
        # Prosody-aware attention components
        self.prosody_projection = nn.Linear(64, 256)  # Project prosody to match LSTM size
        self.query_transform = nn.Linear(256, 256)
        self.key_transform = nn.Linear(256, 256)
        self.value_transform = nn.Linear(256, 256)
        
        # Attention combination
        self.attention_combine = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid()
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc_bn = nn.BatchNorm1d(128)
        self.fc_relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def prosody_aware_attention(self, lstm_output, prosody_features):
        """Prosody-aware attention mechanism from notebook"""
        # Ensure prosody_features have the same sequence length as lstm_output
        if prosody_features.size(1) != lstm_output.size(1):
            prosody_features = torch.nn.functional.interpolate(
                prosody_features.transpose(1, 2),
                size=lstm_output.size(1),
                mode='linear'
            ).transpose(1, 2)
        
        # Project prosody features to match LSTM dimension
        prosody_proj = self.prosody_projection(prosody_features)
        
        # Compute attention scores using scaled dot-product attention
        queries = self.query_transform(lstm_output)
        keys = self.key_transform(prosody_proj)
        values = self.value_transform(lstm_output)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / (256 ** 0.5)  # Scale by sqrt(d_k)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention weights
        context = torch.matmul(attention_weights, values)
        
        # Gating mechanism
        gate = self.attention_combine(context)
        gated_context = gate * context + (1 - gate) * lstm_output
        
        # Global average pooling
        pooled = torch.mean(gated_context, dim=1)
        
        return pooled, attention_weights
        
    def forward(self, inputs):
        """Forward pass - exact implementation from notebook"""
        mfcc_input, prosody_input = inputs
        batch_size = mfcc_input.size(0)
        
        # Process MFCC branch
        mfcc = self.mfcc_conv1(mfcc_input)
        mfcc = self.mfcc_conv2(mfcc)
        
        # Process Prosody branch
        prosody = self.prosody_conv1(prosody_input)
        prosody = self.prosody_conv2(prosody)
        
        # Save prosody features for attention
        prosody_features = prosody.transpose(1, 2)
        
        # Concatenate features along the channel dimension
        x = torch.cat([mfcc, prosody], dim=1)
        
        # Shared processing
        x = self.shared_conv(x)
        
        # Prepare for LSTM (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        
        # Bidirectional LSTM layers
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.lstm_dropout1(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.lstm_dropout2(lstm_out2)
        
        # Apply prosody-aware attention
        attended, _ = self.prosody_aware_attention(lstm_out2, prosody_features)
        
        # Fully connected layers
        x = self.fc1(attended)
        x = self.fc_bn(x)
        x = self.fc_relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        
        return x


def extract_mfcc(file_path, n_mfcc=40):
    """Extract MFCC features - exact implementation from notebook"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=3)
        
        # Ensure consistent length
        if len(y) < sr * 3:
            y = np.pad(y, (0, sr * 3 - len(y)))
        else:
            y = y[:sr * 3]
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Extract delta and delta-delta
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Combine all features (120 = 40 + 40 + 40)
        mfcc_combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # Verify shape
        assert mfcc_combined.shape[0] == 120, f"Expected 120 MFCC features, got {mfcc_combined.shape[0]}"
        
        return mfcc_combined, y, sr
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None, None


def extract_prosodic_features(file_path):
    """Extract prosodic features - exact implementation from notebook"""
    try:
        y, sr = librosa.load(file_path, duration=3)
        
        if len(y) < sr * 3:
            y = np.pad(y, (0, sr * 3 - len(y)))
        else:
            y = y[:sr * 3]
        
        hop_length = 512
        frame_length = 2048
        
        # Extract pitch (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=hop_length
        )
        f0 = np.nan_to_num(f0)
        
        if len(f0) > 1:
            pitch_var = np.abs(np.diff(f0, axis=0))
            pitch_var = np.pad(pitch_var, (0, 1), mode='edge')
        else:
            pitch_var = np.zeros_like(f0)
        
        # Extract intensity (RMS)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        if len(rms) > 1:
            energy_var = np.abs(np.diff(rms, axis=0))
            energy_var = np.pad(energy_var, (0, 1), mode='edge')
        else:
            energy_var = np.zeros_like(rms)
        
        # Extract speech rate
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        
        tempo_frames = np.zeros_like(onset_env)
        if len(beats) > 0:
            tempo_frames[beats] = 1.0
        
        if len(onset_env) > 1:
            rate_var = np.abs(np.diff(onset_env, axis=0))
            rate_var = np.pad(rate_var, (0, 1), mode='edge')
        else:
            rate_var = np.zeros_like(onset_env)
        
        # Extract jitter (pitch perturbation)
        jitter = np.zeros_like(f0)
        voiced_indices = np.where(f0 > 0)[0]
        if len(voiced_indices) > 1:
            for i in range(len(voiced_indices) - 1):
                idx = voiced_indices[i]
                next_idx = voiced_indices[i + 1]
                if f0[idx] > 0 and f0[next_idx] > 0:
                    jitter[idx] = np.abs(f0[next_idx] - f0[idx]) / (f0[idx] + 1e-8)
        
        # Extract shimmer (amplitude perturbation)
        shimmer = np.zeros_like(rms)
        if len(rms) > 1:
            for i in range(len(rms) - 1):
                if rms[i] > 0:
                    shimmer[i] = np.abs(rms[i+1] - rms[i]) / (rms[i] + 1e-8)
        
        min_length = min(len(f0), len(pitch_var), len(jitter), len(rms), 
                        len(energy_var), len(shimmer), len(rate_var), len(tempo_frames))
        
        # Extract formants (first 3 formants)
        formants = np.zeros((3, min_length))
        n_frames = min_length
        frame_samples = len(y) // n_frames if n_frames > 0 else len(y)
        
        for i in range(n_frames):
            start_idx = i * frame_samples
            end_idx = min(start_idx + frame_samples, len(y))
            frame = y[start_idx:end_idx]
            
            if len(frame) > 16:
                try:
                    pre_emphasized = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
                    a = librosa.lpc(pre_emphasized, order=12)
                    roots = np.roots(a)
                    roots = roots[np.imag(roots) >= 0]
                    angles = np.arctan2(np.imag(roots), np.real(roots))
                    freqs = angles * (sr / (2 * np.pi))
                    freqs = sorted(freqs[freqs > 90])
                    
                    for j in range(min(3, len(freqs))):
                        formants[j, i] = freqs[j]
                except:
                    pass
        
        # Truncate features
        f0 = f0[:min_length]
        pitch_var = pitch_var[:min_length]
        jitter = jitter[:min_length]
        rms = rms[:min_length]
        energy_var = energy_var[:min_length]
        shimmer = shimmer[:min_length]
        rate_var = rate_var[:min_length]
        tempo_frames = tempo_frames[:min_length]
        
        # Stack prosodic features (11 features)
        prosodic_features = np.vstack([
            f0, pitch_var, jitter, rms, energy_var, shimmer, 
            rate_var, tempo_frames, formants[0], formants[1], formants[2]
        ])
        
        assert prosodic_features.shape[0] == 11, f"Expected 11 features, got {prosodic_features.shape[0]}"
        
        # Normalize features
        for i in range(prosodic_features.shape[0]):
            if np.any(prosodic_features[i]):
                prosodic_features[i] = (prosodic_features[i] - np.mean(prosodic_features[i])) / (np.std(prosodic_features[i]) + 1e-8)
        
        return prosodic_features
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


class EmotionRecognitionApp(ctk.CTk):
    """GUI Application for Speech Emotion Recognition"""
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Speech Emotion Recognition GUI v2 (Notebook-Based)")
        self.geometry("1200x800")
        
        # Emotion labels and colors (RAVDESS dataset)
        self.emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
        self.emotion_colors = {
            'Neutral': "#eedbb9",
            'Calm': "#fbfbfb",
            'Happy': '#f1c40f',
            'Sad': "#026edb",
            'Angry': '#e74c3c',
            'Fearful': '#9b59b6',
            'Disgust': '#16a085',
            'Surprised': '#e67e22'
        }
        
        self.device = device
        self.model = None
        self.current_audio_path = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Create main container
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header Section
        header_frame = ctk.CTkFrame(main_container, fg_color="#1e1e1e", corner_radius=15)
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Enhancing Speech Emotion Recognition in Mental Health and Wellness Podcasts\nUsing CRNN(BiLSTM) with Prosody-Aware Attention Mechanism",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#ffffff",
            wraplength=1100
        )
        title_label.pack(pady=15)
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Speech Emotion Recognition - RAVDESS Dataset (GUI v2 - Notebook Implementation)",
            font=ctk.CTkFont(size=14),
            text_color="#b0b0b0"
        )
        subtitle_label.pack(pady=(0, 10))
        
        university_label = ctk.CTkLabel(
            header_frame,
            text="To, Mendoza, Cuenca (2025)",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="#808080"
        )
        university_label.pack(pady=(0, 15))
        
        # Content Area (Two columns)
        content_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        content_frame.pack(fill="both", expand=True)
        
        # Left Column - Controls
        left_column = ctk.CTkFrame(content_frame, fg_color="#1e1e1e", corner_radius=15, width=400)
        left_column.pack(side="left", fill="both", padx=(0, 10), expand=False)
        left_column.pack_propagate(False)
        
        # Model Info
        model_frame = ctk.CTkFrame(left_column, fg_color="#2a2a2a", corner_radius=10)
        model_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(
            model_frame,
            text="Model Architecture",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 5))
        
        ctk.CTkLabel(
            model_frame,
            text="CRNN (BiLSTM) w/Prosody-Attention",
            font=ctk.CTkFont(size=12),
            text_color="#3498db"
        ).pack(anchor="w", padx=15, pady=(0, 5))
        
        ctk.CTkLabel(
            model_frame,
            text="📊 120 MFCC + 11 Prosody Features",
            font=ctk.CTkFont(size=10),
            text_color="#95a5a6"
        ).pack(anchor="w", padx=15, pady=(0, 15))
        
        # Upload Section
        upload_frame = ctk.CTkFrame(left_column, fg_color="transparent")
        upload_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            upload_frame,
            text="Upload Audio File",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(0, 10))
        
        self.upload_button = ctk.CTkButton(
            upload_frame,
            text="📁 Upload your Audio File",
            font=ctk.CTkFont(size=14),
            height=45,
            command=self.upload_audio,
            fg_color="#2980b9",
            hover_color="#3498db"
        )
        self.upload_button.pack(fill="x")
        
        self.file_label = ctk.CTkLabel(
            upload_frame,
            text="No file selected",
            font=ctk.CTkFont(size=11),
            text_color="#808080"
        )
        self.file_label.pack(pady=(10, 0))
        
        # Reset button
        self.reset_button = ctk.CTkButton(
            upload_frame,
            text="Reset",
            font=ctk.CTkFont(size=11),
            height=28,
            command=self.reset,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            corner_radius=6
        )
        self.reset_button.pack(pady=(8, 0))
        
        # Waveform Display
        waveform_frame = ctk.CTkFrame(left_column, fg_color="#2a2a2a", corner_radius=10, height=200)
        waveform_frame.pack(fill="x", padx=20, pady=10)
        waveform_frame.pack_propagate(False)
        
        ctk.CTkLabel(
            waveform_frame,
            text="Audio Waveform",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=(10, 5))
        
        self.waveform_canvas_frame = ctk.CTkFrame(waveform_frame, fg_color="#1a1a1a")
        self.waveform_canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Right Column - Results
        right_column = ctk.CTkFrame(content_frame, fg_color="#1e1e1e", corner_radius=15)
        right_column.pack(side="right", fill="both", expand=True)
        
        # Prediction Result
        result_header = ctk.CTkFrame(right_column, fg_color="transparent")
        result_header.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(
            result_header,
            text="Prediction Results",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w")
        
        # Predict Button
        self.predict_button = ctk.CTkButton(
            right_column,
            text="🎯 Predict Emotion",
            font=ctk.CTkFont(size=13),
            height=40,
            width=180,
            command=self.predict_emotion,
            fg_color="#27ae60",
            hover_color="#2ecc71",
            state="normal",
            corner_radius=8
        )
        self.predict_button.pack(anchor="w", padx=20, pady=(0, 15))
        
        # Predicted Emotion Display
        self.emotion_frame = ctk.CTkFrame(right_column, fg_color="#2a2a2a", corner_radius=15, height=150)
        self.emotion_frame.pack(fill="x", padx=20, pady=(0, 20))
        self.emotion_frame.pack_propagate(False)
        
        ctk.CTkLabel(
            self.emotion_frame,
            text="Predicted Emotion",
            font=ctk.CTkFont(size=13),
            text_color="#808080"
        ).pack(pady=(15, 5))
        
        self.emotion_label = ctk.CTkLabel(
            self.emotion_frame,
            text="—",
            font=ctk.CTkFont(size=42, weight="bold"),
            text_color="#ffffff"
        )
        self.emotion_label.pack(pady=10)
        
        self.confidence_label = ctk.CTkLabel(
            self.emotion_frame,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="#b0b0b0"
        )
        self.confidence_label.pack()
        
        # Confidence Scores
        scores_frame = ctk.CTkFrame(right_column, fg_color="#2a2a2a", corner_radius=15)
        scores_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        ctk.CTkLabel(
            scores_frame,
            text="Confidence Distribution",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 10))
        
        self.scores_canvas_frame = ctk.CTkFrame(scores_frame, fg_color="#1a1a1a")
        self.scores_canvas_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Footer
        footer_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        footer_frame.pack(fill="x", pady=(10, 0))
        
        device_label = ctk.CTkLabel(
            footer_frame,
            text=f"🖥️ Running on: {self.device}",
            font=ctk.CTkFont(size=11),
            text_color="#808080"
        )
        device_label.pack(side="left")
        
        version_label = ctk.CTkLabel(
            footer_frame,
            text="v2.0 - Notebook Logic",
            font=ctk.CTkFont(size=11),
            text_color="#808080"
        )
        version_label.pack(side="right")
    
    def upload_audio(self):
        """Handle audio file upload"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.current_audio_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.configure(text=f"File: {filename}", text_color="#2ecc71")
            
            # Display waveform
            self.display_waveform(file_path)
    
    def display_waveform(self, file_path):
        """Display audio waveform"""
        try:
            y, sr = librosa.load(file_path, duration=3)
            
            # Clear previous plot
            for widget in self.waveform_canvas_frame.winfo_children():
                widget.destroy()
            
            fig, ax = plt.subplots(figsize=(4, 2), facecolor='#1a1a1a')
            ax.set_facecolor('#1a1a1a')
            
            time = np.linspace(0, len(y) / sr, num=len(y))
            ax.plot(time, y, color='#3498db', linewidth=0.5)
            ax.set_xlabel('Time (s)', color='white', fontsize=8)
            ax.set_ylabel('Amplitude', color='white', fontsize=8)
            ax.tick_params(colors='white', labelsize=7)
            ax.grid(True, alpha=0.2, color='white')
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=self.waveform_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            plt.close(fig)
        except Exception as e:
            print(f"Error displaying waveform: {str(e)}")
    
    def load_model(self):
        """Load the trained model"""
        if self.model is None:
            try:
                print("Initializing EmotionCRNN model...")
                self.model = EmotionCRNN().to(self.device)
                
                # Get script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Model paths to check
                model_paths = [
                    os.path.join(script_dir, 'new model', 'best_multimodal_model.pth'),
                    r'C:\Users\Ezeniel Cuenca\Documents\GitHub\THESIS-SER\new model\best_multimodal_model.pth',
                    os.path.join(script_dir, 'best_multimodal_model.pth'),
                ]
                
                model_loaded = False
                for model_path in model_paths:
                    if os.path.exists(model_path):
                        try:
                            print(f"Loading model from: {model_path}")
                            checkpoint = torch.load(model_path, map_location=self.device)
                            
                            # Handle different checkpoint formats
                            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                                self.model.load_state_dict(checkpoint['model_state_dict'])
                                print(f"✓ Model loaded from checkpoint (epoch: {checkpoint.get('epoch', 'N/A')})")
                            else:
                                self.model.load_state_dict(checkpoint)
                                print("✓ Model loaded from state dict")
                            
                            model_loaded = True
                            # Don't show messagebox during load, just print to console
                            print(f"✓ Model loaded successfully from: {model_path}")
                            break
                        except Exception as e:
                            print(f"⚠ Error loading {model_path}: {str(e)}")
                            continue
                
                if not model_loaded:
                    messagebox.showerror(
                        "Model Not Found",
                        f"Pre-trained model not found!\n\nSearched locations:\n" +
                        "\n".join(model_paths) +
                        "\n\nPlease train the model first using the notebook."
                    )
                    return False
                
                self.model.eval()
                print("Model set to evaluation mode")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {str(e)}")
                return False
        
        return True
    
    def predict_emotion(self):
        """Predict emotion from uploaded audio file"""
        if not self.current_audio_path:
            messagebox.showwarning("Warning", "Please upload an audio file first!")
            return
        
        if not self.load_model():
            return
        
        try:
            print(f"\nPredicting emotion for: {self.current_audio_path}")
            
            # Extract features using notebook functions
            print("Extracting MFCC features...")
            mfcc, y, sr = extract_mfcc(self.current_audio_path)
            
            print("Extracting prosodic features...")
            prosody = extract_prosodic_features(self.current_audio_path)
            
            if mfcc is None or prosody is None:
                messagebox.showerror("Error", "Failed to extract features from audio file")
                return
            
            print(f"MFCC shape: {mfcc.shape}")
            print(f"Prosody shape: {prosody.shape}")
            
            # Prepare inputs (transpose to get time steps on first dimension)
            mfcc = mfcc.T
            prosody = prosody.T
            
            # Ensure both have the same time dimension
            min_length = min(mfcc.shape[0], prosody.shape[0])
            mfcc = mfcc[:min_length]
            prosody = prosody[:min_length]
            
            print(f"Aligned shapes - MFCC: {mfcc.shape}, Prosody: {prosody.shape}")
            
            # Convert to tensors and add batch dimension
            mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).transpose(1, 2).to(self.device)
            prosody_tensor = torch.FloatTensor(prosody).unsqueeze(0).transpose(1, 2).to(self.device)
            
            print(f"Tensor shapes - MFCC: {mfcc_tensor.shape}, Prosody: {prosody_tensor.shape}")
            
            # Predict
            print("Running model inference...")
            with torch.no_grad():
                outputs = self.model((mfcc_tensor, prosody_tensor))
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item() * 100
            
            print(f"Prediction: {self.emotion_labels[predicted_idx]} ({confidence:.2f}%)")
            
            # Update UI
            predicted_emotion = self.emotion_labels[predicted_idx]
            self.emotion_label.configure(
                text=predicted_emotion,
                text_color=self.emotion_colors[predicted_emotion]
            )
            self.confidence_label.configure(text=f"Confidence: {confidence:.1f}%")
            
            # Display confidence distribution
            self.display_confidence_scores(probabilities[0].cpu().numpy())
            
            print("✓ Prediction completed successfully!")
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error during prediction:\n{str(e)}")
    
    def display_confidence_scores(self, probabilities):
        """Display confidence scores as bar chart"""
        # Clear previous plot
        for widget in self.scores_canvas_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        colors = [self.emotion_colors[label] for label in self.emotion_labels]
        bars = ax.barh(self.emotion_labels, probabilities * 100, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Confidence (%)', color='white', fontsize=10)
        ax.set_xlim(0, 100)
        ax.tick_params(colors='white', labelsize=9)
        ax.grid(axis='x', alpha=0.2, color='white')
        
        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{prob*100:.1f}%', va='center', color='white', fontsize=8)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.scores_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        plt.close(fig)
    
    def reset(self):
        """Reset the application state"""
        self.current_audio_path = None
        self.file_label.configure(text="No file selected", text_color="#808080")
        self.emotion_label.configure(text="—", text_color="#ffffff")
        self.confidence_label.configure(text="")
        
        # Clear plots
        for widget in self.waveform_canvas_frame.winfo_children():
            widget.destroy()
        for widget in self.scores_canvas_frame.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    print("="*80)
    print("EMOTION RECOGNITION GUI v2.0 - Notebook-Based Implementation")
    print("="*80)
    print(f"Device: {device}")
    print("Starting application...")
    
    app = EmotionRecognitionApp()
    app.mainloop()
