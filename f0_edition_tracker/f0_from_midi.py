import pretty_midi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def extract_f0_from_midi(midi_file, output_csv_path=None, method="melody", track_idx=None, 
                         time_resolution=0.01, plot=False):
    """
    Extract F0 (fundamental frequency) from MIDI file using different methods and save as CSV.
    
    Parameters:
        midi_file (str): Path to the MIDI file
        output_csv_path (str): Path to save the F0 CSV file (optional)
        method (str): Method to extract F0:
            - "lowest": Always use the lowest note as F0
            - "melody": Try to extract the melody line (highest non-percussion in melody range)
            - "bass": Extract the bass line (lowest non-percussion)
            - "track": Use a specific track (requires track_idx)
        track_idx (int): Index of the track to use (only for method="track")
        time_resolution (float): Time resolution in seconds
        plot (bool): Whether to plot the extracted F0
        
    Returns:
        tuple: (times, f0) arrays
    """
    # Load MIDI file
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return None, None
    
    # Create time grid
    end_time = midi_data.get_end_time()
    times = np.arange(0, end_time, time_resolution)
    f0 = np.zeros_like(times)
    
    # Filter instruments based on method
    instruments = midi_data.instruments
    
    if method == "track" and track_idx is not None:
        if track_idx < len(instruments):
            instruments = [instruments[track_idx]]
        else:
            print(f"Error: Track index {track_idx} out of range. Using all tracks.")
    
    # Filter out percussion for melody and bass methods
    if method in ["melody", "bass"]:
        instruments = [inst for inst in instruments if not inst.is_drum]
    
    # Process each time point
    for i, time in enumerate(times):
        active_notes = []
        
        # Collect all active notes at this time point
        for instrument in instruments:
            for note in instrument.notes:
                if note.start <= time < note.end:
                    active_notes.append(note)
        
        # Apply the selected method to determine F0
        if active_notes:
            if method == "lowest" or method == "bass":
                # Get the lowest note
                lowest_note = min(active_notes, key=lambda note: note.pitch)
                f0[i] = pretty_midi.note_number_to_hz(lowest_note.pitch)
            
            elif method == "melody":
                # Get the highest note in a reasonable melody range (avoid very high notes)
                melody_notes = [note for note in active_notes if 50 <= note.pitch <= 95]
                if melody_notes:
                    # Prioritize notes with higher velocity in the melody range
                    melody_note = max(melody_notes, key=lambda note: (note.pitch * 0.7 + note.velocity * 0.3))
                    f0[i] = pretty_midi.note_number_to_hz(melody_note.pitch)
                else:
                    # If no notes in melody range, take the highest note
                    highest_note = max(active_notes, key=lambda note: note.pitch)
                    f0[i] = pretty_midi.note_number_to_hz(highest_note.pitch)
            
            elif method == "track":
                # For track method, take the highest note in the selected track
                highest_note = max(active_notes, key=lambda note: note.pitch)
                f0[i] = pretty_midi.note_number_to_hz(highest_note.pitch)
    
    # Create DataFrame for CSV output
    df = pd.DataFrame({
        'time': times,
        'f0': f0
    })
    
    # Save to CSV
    if output_csv_path is None:
        output_csv_path = os.path.splitext(midi_file)[0] + "_f0.csv"
    
    df.to_csv(output_csv_path, index=False)
    print(f"F0 data saved to: {output_csv_path}")
    
    # Plot F0 contour if requested
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(times, f0)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'F0 Contour from MIDI - Method: {method}')
        plt.grid(True)
        
        # Add MIDI note lines for reference
        for midi_note in range(36, 96, 12):  # C2 to C7, octave steps
            freq = pretty_midi.note_number_to_hz(midi_note)
            plt.axhline(y=freq, color='r', linestyle='--', alpha=0.3)
            note_name = pretty_midi.note_number_to_name(midi_note)
            plt.text(0, freq, f"{note_name}", fontsize=8, ha='left', va='bottom')
        
        plt.show()
    
    return times, f0

def analyze_midi_file(midi_file):
    """Print information about the MIDI file to help choose the right track/method."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        print(f"\nMIDI File Analysis: {os.path.basename(midi_file)}")
        print(f"Duration: {midi_data.get_end_time():.2f} seconds")
        print(f"Number of instruments: {len(midi_data.instruments)}")
        
        for i, instrument in enumerate(midi_data.instruments):
            num_notes = len(instrument.notes)
            if num_notes == 0:
                continue
                
            pitch_min = min(note.pitch for note in instrument.notes)
            pitch_max = max(note.pitch for note in instrument.notes)
            pitch_avg = sum(note.pitch for note in instrument.notes) / num_notes
            
            instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            drum_text = " (Drums)" if instrument.is_drum else ""
            
            print(f"\nTrack {i}: {instrument_name}{drum_text}")
            print(f"  Notes: {num_notes}")
            print(f"  Pitch range: {pretty_midi.note_number_to_name(pitch_min)} to "
                  f"{pretty_midi.note_number_to_name(pitch_max)} (MIDI {pitch_min}-{pitch_max})")
            print(f"  Average pitch: {pitch_avg:.1f}")
            
            # Suggest if this track might be melody, bass, etc.
            if instrument.is_drum:
                print("  Role: Percussion")
            elif pitch_avg < 50:
                print("  Likely role: Bass")
            elif 55 <= pitch_avg <= 75:
                print("  Likely role: Melody/Lead")
            elif pitch_avg > 75:
                print("  Likely role: High accompaniment")
            else:
                print("  Likely role: Accompaniment/Chords")
                
    except Exception as e:
        print(f"Error analyzing MIDI file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract F0 from MIDI files")
    parser.add_argument("midi_file", help="Path to the MIDI file")
    parser.add_argument("--output", "-o", help="Output file path (.npy)")
    parser.add_argument("--method", "-m", choices=["lowest", "melody", "bass", "track"], 
                        default="melody", help="Method to extract F0")
    parser.add_argument("--track", "-t", type=int, help="Track index for track method")
    parser.add_argument("--resolution", "-r", type=float, default=0.01, 
                        help="Time resolution in seconds")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot the F0 contour")
    parser.add_argument("--analyze", "-a", action="store_true", 
                        help="Analyze the MIDI file structure")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_midi_file(args.midi_file)
    
    if args.method == "track" and args.track is None:
        print("Error: Track index is required for track method")
        parser.print_help()
    else:
        extract_f0_from_midi(
            args.midi_file, 
            args.output, 
            args.method, 
            args.track, 
            args.resolution, 
            args.plot
        )
