from torchfcpe import spawn_bundled_infer_model
import torch
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import argparse
import os

def extract_and_plot_f0(audio_path, output_midi_path="output.mid", plot=True, method="melody"):
    """
    Extract F0 from audio using FCPE, plot the F0 contour, and save as MIDI
    
    Parameters:
        audio_path (str): Path to the audio file
        output_midi_path (str): Path to save the MIDI file
        plot (bool): Whether to plot the F0 contour
        method (str): Method to extract F0 from MIDI:
            - "lowest": Always use the lowest note as F0
            - "melody": Try to extract the melody line (highest non-percussion in melody range)
            - "bass": Extract the bass line (lowest non-percussion)
    """
    # Configure device and target hop size
    device = 'cpu'  # or 'cuda' if using a GPU
    sr = 16000  # Sample rate
    hop_size = 160  # Hop size for processing
    
    print(f"Processing audio: {audio_path}")
    print(f"Output MIDI will be saved to: {output_midi_path}")
    
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=sr)
    audio = librosa.to_mono(audio)
    audio_length = len(audio)
    f0_target_length = (audio_length // hop_size) + 1
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(device)
    
    # Load the model
    model = spawn_bundled_infer_model(device=device)
    
    # Extract MIDI and F0 from audio
    midi = model.extact_midi(
        audio_tensor,
        sr=sr,
        decoder_mode='local_argmax',  # Recommended mode
        threshold=0.006,  # Threshold for V/UV decision
        f0_min=80,  # Minimum pitch
        f0_max=880,  # Maximum pitch
        output_path=output_midi_path,  # Save MIDI to file
    )
    
    # If plotting is requested
    if plot:
        # Carregar o MIDI gerado para extrair as frequências
        try:
            # Esperar um pouco para garantir que o arquivo MIDI foi salvo
            import time
            time.sleep(0.5)
            
            # Carregar o MIDI gerado
            midi_data = pretty_midi.PrettyMIDI(output_midi_path)
            
            # Criar uma grade de tempo (em segundos)
            end_time = midi_data.get_end_time()
            time_resolution = 0.01  # 10ms
            times = np.arange(0, end_time, time_resolution)
            
            # Extrair F0 (frequência fundamental) para cada ponto de tempo
            f0 = np.zeros_like(times)
            
            # Filtrar instrumentos com base no método
            instruments = midi_data.instruments
            
            # Filtrar percussão para os métodos melody e bass
            if method in ["melody", "bass"]:
                instruments = [inst for inst in instruments if not inst.is_drum]
            
            for i, time in enumerate(times):
                active_notes = []
                
                # Coletar todas as notas ativas neste ponto de tempo
                for instrument in instruments:
                    for note in instrument.notes:
                        if note.start <= time < note.end:
                            active_notes.append(note)
                
                # Aplicar o método selecionado para determinar F0
                if active_notes:
                    if method == "lowest" or method == "bass":
                        # Obter a nota mais baixa
                        lowest_note = min(active_notes, key=lambda note: note.pitch)
                        f0[i] = pretty_midi.note_number_to_hz(lowest_note.pitch)
                    
                    elif method == "melody":
                        # Obter a nota mais alta em um intervalo razoável de melodia (evitar notas muito altas)
                        melody_notes = [note for note in active_notes if 50 <= note.pitch <= 95]
                        if melody_notes:
                            # Priorizar notas com maior velocidade no intervalo de melodia
                            melody_note = max(melody_notes, key=lambda note: (note.pitch * 0.7 + note.velocity * 0.3))
                            f0[i] = pretty_midi.note_number_to_hz(melody_note.pitch)
                        else:
                            # Se não houver notas no intervalo de melodia, pegar a nota mais alta
                            highest_note = max(active_notes, key=lambda note: note.pitch)
                            f0[i] = pretty_midi.note_number_to_hz(highest_note.pitch)
            
            # Plot F0 contour
            plt.figure(figsize=(12, 6))
            plt.plot(times, f0)
            plt.xlabel('Tempo (segundos)')
            plt.ylabel('Frequência (Hz)')
            plt.title(f'Contorno F0 do MIDI Gerado (método: {method}) - {os.path.basename(output_midi_path)}')
            plt.grid(True)
            
            # Adicionar linhas de referência para notas MIDI
            for midi_note in range(36, 96, 12):  # C2 a C7, passos de oitava
                freq = pretty_midi.note_number_to_hz(midi_note)
                plt.axhline(y=freq, color='r', linestyle='--', alpha=0.3)
                note_name = pretty_midi.note_number_to_name(midi_note)
                plt.text(0, freq, f"{note_name}", fontsize=8, ha='left', va='bottom')
            
            plt.show()
            
        except Exception as e:
            print(f"Erro ao plotar o contorno F0 do MIDI: {e}")
            # Tentar usar F0 diretamente do modelo como fallback
            try:
                f0 = model.extact_f0(
                    audio_tensor,
                    sr=sr,
                    decoder_mode='local_argmax',
                    threshold=0.006,
                    f0_min=80,
                    f0_max=880,
                ).cpu().numpy()[0]
                
                # Criar array de tempo
                times = np.arange(len(f0)) * hop_size / sr
                
                plt.figure(figsize=(12, 6))
                plt.plot(times, f0)
                plt.xlabel('Tempo (segundos)')
                plt.ylabel('Frequência (Hz)')
                plt.title(f'Contorno F0 do Áudio (fallback) - {os.path.basename(audio_path)}')
                plt.grid(True)
                
                # Adicionar linhas de referência para notas MIDI
                for midi_note in range(36, 96, 12):
                    freq = pretty_midi.note_number_to_hz(midi_note)
                    plt.axhline(y=freq, color='r', linestyle='--', alpha=0.3)
                    note_name = pretty_midi.note_number_to_name(midi_note)
                    plt.text(0, freq, f"{note_name}", fontsize=8, ha='left', va='bottom')
                
                plt.show()
            except Exception as e2:
                print(f"Também não foi possível usar o fallback: {e2}")
    
    print(f"MIDI extraction complete. File saved to: {output_midi_path}")
    return midi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract F0 from audio and save as MIDI")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--output", "-o", default="output.mid", help="Output MIDI file path")
    parser.add_argument("--no-plot", action="store_true", help="Disable F0 contour plotting")
    parser.add_argument("--method", "-m", choices=["lowest", "melody", "bass"], 
                        default="melody", help="Method to extract F0 from MIDI")
    
    args = parser.parse_args()
    
    extract_and_plot_f0(args.audio_file, args.output, not args.no_plot, args.method)