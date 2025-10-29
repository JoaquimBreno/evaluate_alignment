from torchfcpe import spawn_bundled_infer_model
import torch
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os

def extract_f0_from_audio(audio_path, output_csv_path=None, plot=True):
    """
    Extract F0 from audio using FCPE and save as CSV
    
    Parameters:
        audio_path (str): Path to the audio file
        output_csv_path (str): Path to save the F0 CSV file
        plot (bool): Whether to plot the F0 contour
    """
    # Configure device and target hop size
    device = 'cpu'  # or 'cuda' if using a GPU
    sr = 16000  # Sample rate
    hop_size = 160  # Hop size for processing
    
    print(f"Processing audio: {audio_path}")
    
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=sr)
    audio = librosa.to_mono(audio)
    audio_length = len(audio)
    f0_target_length = (audio_length // hop_size) + 1
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(device)
    
    # Load the model
    model = spawn_bundled_infer_model(device=device)
    
    # Perform pitch inference
    f0 = model.infer(
        audio_tensor,
        sr=sr,
        decoder_mode='local_argmax',  # Recommended mode
        threshold=0.006,  # Threshold for V/UV decision
        f0_min=80,  # Minimum pitch
        f0_max=880,  # Maximum pitch
        interp_uv=False,  # Interpolate unvoiced frames
        output_interp_target_length=f0_target_length,  # Interpolate to target length
    )
    
    # Convert to numpy and create time array
    f0_numpy = f0.cpu().numpy()
    # Garante que seja 1D
    if f0_numpy.ndim > 1:
        f0_numpy = f0_numpy.squeeze()  # Remove dimensões unitárias
    if f0_numpy.ndim > 1:
        f0_numpy = f0_numpy.flatten()  # Se ainda for multidimensional, achata
    
    times = np.arange(len(f0_numpy)) * hop_size / sr
    
    # Create DataFrame for CSV output
    df = pd.DataFrame({
        'time': times,
        'f0': f0_numpy
    })
    
    # Save to CSV
    if output_csv_path is None:
        output_csv_path = os.path.splitext(audio_path)[0] + "_f0.csv"
    
    df.to_csv(output_csv_path, index=False)
    print(f"F0 data saved to: {output_csv_path}")
    
    # Plot F0 contour if requested
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(times, f0_numpy)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'F0 Contour from Audio - {os.path.basename(audio_path)}')
        plt.grid(True)
        
        # Add MIDI note reference lines
        for midi_note in range(36, 96, 12):  # C2 to C7, octave steps
            freq = 440 * (2 ** ((midi_note - 69) / 12))  # Convert MIDI note to Hz
            plt.axhline(y=freq, color='r', linestyle='--', alpha=0.3)
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_name = note_names[midi_note % 12] + str(midi_note // 12 - 1)
            plt.text(0, freq, f"{note_name}", fontsize=8, ha='left', va='bottom')
        
        plt.show()
    
    print(f"F0 extraction complete. CSV saved to: {output_csv_path}")
    return times, f0_numpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract F0 from audio using FCPE and save as CSV")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--no-plot", action="store_true", help="Disable F0 contour plotting")
    
    args = parser.parse_args()
    
    extract_f0_from_audio(args.audio_file, args.output, not args.no_plot)