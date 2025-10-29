import numpy as np
import librosa
import matplotlib.pyplot as plt
import pretty_midi
from scipy.spatial.distance import cdist
from fastdtw import fastdtw
from pathlib import Path

def load_midi_as_chroma(midi_file, sr=22050, hop_length=512):
    """
    Carrega um arquivo MIDI e o converte em uma representação de chroma
    
    Parâmetros:
    - midi_file: caminho para o arquivo MIDI
    - sr: taxa de amostragem (default: 22050 Hz)
    - hop_length: tamanho do hop em samples (default: 512)
    
    Retorna:
    - midi_chroma: representação chroma do MIDI
    - midi_times: array de tempos correspondentes aos frames
    """
    # Carrega o arquivo MIDI
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    # Obtém a duração total em segundos
    total_duration = midi_data.get_end_time()
    
    # Gera o chroma do MIDI
    midi_chroma = midi_data.get_chroma(fs=sr/hop_length)
    
    # Calcula os tempos correspondentes aos frames
    n_frames = midi_chroma.shape[1]
    midi_times = np.linspace(0, total_duration, n_frames)
    
    return midi_chroma, midi_times

def load_audio_as_chroma(audio_file, sr=22050, hop_length=512):
    """
    Carrega um arquivo de áudio e o converte em uma representação de chroma
    
    Parâmetros:
    - audio_file: caminho para o arquivo de áudio
    - sr: taxa de amostragem (default: 22050 Hz)
    - hop_length: tamanho do hop em samples (default: 512)
    
    Retorna:
    - audio_chroma: representação chroma do áudio
    - audio_times: array de tempos correspondentes aos frames
    """
    # Carrega o áudio
    y, sr = librosa.load(audio_file, sr=sr)
    
    # Gera o chroma do áudio
    audio_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    # Calcula os tempos correspondentes aos frames
    audio_times = librosa.times_like(audio_chroma, sr=sr, hop_length=hop_length)
    
    return audio_chroma, audio_times

def align_midi_to_audio(midi_file, audio_file, sr=22050, hop_length=512):
    """
    Alinha um arquivo MIDI a um arquivo de áudio usando DTW
    
    Parâmetros:
    - midi_file: caminho para o arquivo MIDI
    - audio_file: caminho para o arquivo de áudio
    - sr: taxa de amostragem (default: 22050 Hz)
    - hop_length: tamanho do hop em samples (default: 512)
    
    Retorna:
    - midi_times: array de tempos originais do MIDI
    - aligned_times: array de tempos alinhados do MIDI
    - path: caminho do alinhamento DTW
    """
    # Carrega as representações chroma
    midi_chroma, midi_times = load_midi_as_chroma(midi_file, sr, hop_length)
    audio_chroma, audio_times = load_audio_as_chroma(audio_file, sr, hop_length)
    
    # Normaliza as representações chroma
    midi_chroma = librosa.util.normalize(midi_chroma, axis=0)
    audio_chroma = librosa.util.normalize(audio_chroma, axis=0)
    
    # Calcula a matriz de custo entre as duas representações
    cost_matrix = cdist(midi_chroma.T, audio_chroma.T, metric='cosine')
    
    # Aplica o DTW
    distance, path = fastdtw(midi_chroma.T, audio_chroma.T, dist=lambda x, y: np.linalg.norm(x - y))
    
    # Converte o caminho em arrays numpy
    path = np.array(path)
    
    # Extrai os índices do caminho
    midi_idx, audio_idx = path.T
    
    # Mapeia os tempos do MIDI para os tempos do áudio usando o caminho DTW
    aligned_times = np.zeros_like(midi_times)
    for i, j in zip(midi_idx, audio_idx):
        if i < len(midi_times) and j < len(audio_times):
            aligned_times[i] = audio_times[j]
    
    return midi_times, aligned_times, path, midi_chroma, audio_chroma, cost_matrix

def visualize_alignment(midi_file, audio_file, midi_times, aligned_times, path, midi_chroma, audio_chroma, cost_matrix):
    """
    Visualiza o alinhamento entre o MIDI e o áudio
    
    Parâmetros:
    - midi_file: caminho para o arquivo MIDI
    - audio_file: caminho para o arquivo de áudio
    - midi_times: array de tempos originais do MIDI
    - aligned_times: array de tempos alinhados do MIDI
    - path: caminho do alinhamento DTW
    - midi_chroma: representação chroma do MIDI
    - audio_chroma: representação chroma do áudio
    - cost_matrix: matriz de custo entre as duas representações
    """
    # Cria a figura com subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot da matriz de custo e do caminho DTW (otimizado)
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    
    # Reduz a resolução para visualização mais rápida
    downsample_factor = max(1, max(cost_matrix.shape) // 1000)  # Máximo 1000x1000
    cost_matrix_viz = cost_matrix[::downsample_factor, ::downsample_factor]
    path_viz = path[::max(1, len(path) // 2000)]  # Máximo 2000 pontos no caminho
    
    librosa.display.specshow(cost_matrix_viz.T, cmap='viridis')
    plt.colorbar(format='%+2.0f')
    plt.title('Matriz de Custo e Caminho DTW (Downsampled)')
    
    # Plota o caminho DTW reduzido
    plt.plot(path_viz[:, 0] / downsample_factor, path_viz[:, 1] / downsample_factor, 'r-', linewidth=1)
    
    # Plot do chroma do MIDI
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    librosa.display.specshow(midi_chroma, x_axis='time', y_axis='chroma',
                           cmap='viridis')
    plt.colorbar(format='%+2.0f')
    plt.title(f'Chroma do MIDI: {Path(midi_file).name}')
    
    # Plot do chroma do áudio
    ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
    librosa.display.specshow(audio_chroma, x_axis='time', y_axis='chroma',
                           cmap='viridis')
    plt.colorbar(format='%+2.0f')
    plt.title(f'Chroma do Áudio: {Path(audio_file).name}')
    
    # Plot da função de warping
    ax4 = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
    plt.plot(midi_times, aligned_times, 'b-')
    plt.plot([0, max(midi_times)], [0, max(aligned_times)], 'r--')
    plt.xlabel('Tempo MIDI (s)')
    plt.ylabel('Tempo Áudio (s)')
    plt.title('Função de Warping')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Salva a imagem em vez de mostrar
    midi_name = Path(midi_file).stem
    save_path = f"alignment_{midi_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Visualização salva em: {save_path}")
    plt.close()  # Fecha a figura para liberar memória

def align_midi_tokens_to_cqt(midi_file, midi_tokens, cqt_file, audio_file, output_file=None):
    """
    Alinha tokens MIDI a uma representação CQT de áudio
    
    Parâmetros:
    - midi_file: caminho para o arquivo MIDI
    - midi_tokens: objeto de tokens MIDI
    - cqt_file: caminho para o arquivo CQT (.npy)
    - audio_file: caminho para o arquivo de áudio correspondente
    - output_file: caminho para salvar os resultados (opcional)
    
    Retorna:
    - aligned_tokens: tokens MIDI com tempos alinhados
    """
    # Carrega o CQT
    cqt = np.load(cqt_file)
    
    # Realiza o alinhamento entre o MIDI e o áudio
    midi_times, aligned_times, path, midi_chroma, audio_chroma, cost_matrix = align_midi_to_audio(midi_file, audio_file)
    
    # Visualiza o alinhamento
    visualize_alignment(midi_file, audio_file, midi_times, aligned_times, path, midi_chroma, audio_chroma, cost_matrix)
    
    # Cria uma função de mapeamento de tempo MIDI para tempo de áudio
    from scipy.interpolate import interp1d
    time_mapping = interp1d(midi_times, aligned_times, bounds_error=False, fill_value="extrapolate")
    
    # Função para aplicar o mapeamento de tempo a um token
    def map_token_time(token_time, ticks_per_beat, tempo):
        # Converte tempo em ticks para segundos
        seconds = token_time / ticks_per_beat * tempo / 1_000_000
        # Aplica o mapeamento de tempo
        return time_mapping(seconds)
    
    # Aqui você precisaria implementar a lógica para aplicar o mapeamento aos tokens MIDI
    # Isso depende da estrutura específica dos seus tokens MIDI
    
    # Exemplo de como você poderia processar os tokens:
    # for token in midi_tokens:
    #     if hasattr(token, 'time'):
    #         original_time = token.time
    #         token.time = map_token_time(original_time, ticks_per_beat, tempo)
    
    # Se output_file for fornecido, salve os resultados
    if output_file:
        # Implemente a lógica para salvar os tokens alinhados
        pass
    
    return midi_tokens  # Retorna os tokens alinhados

if __name__ == "__main__":
    import sys
    import os
    
    # Verifica argumentos da linha de comando
    if len(sys.argv) != 3:
        print("Uso: python dtw_midi_audio_alignment.py <midi_file> <audio_file>")
        print("Exemplo: python dtw_midi_audio_alignment.py data/song.midi data/song.wav")
        sys.exit(1)
    
    midi_file = sys.argv[1]
    audio_file = sys.argv[2]
    
    # Verifica se os arquivos existem
    if not os.path.exists(midi_file):
        print(f"❌ Arquivo MIDI não encontrado: {midi_file}")
        sys.exit(1)
    
    if not os.path.exists(audio_file):
        print(f"❌ Arquivo de áudio não encontrado: {audio_file}")
        sys.exit(1)
    
    print(f"🎵 MIDI: {os.path.basename(midi_file)}")
    print(f"🔊 Audio: {os.path.basename(audio_file)}")
    print("🔄 Executando alinhamento DTW...")
    
    try:
        # Realiza o alinhamento entre o MIDI e o áudio
        midi_times, aligned_times, path, midi_chroma, audio_chroma, cost_matrix = align_midi_to_audio(midi_file, audio_file)
        
        # Calcula métricas básicas
        import numpy as np
        temporal_error = np.mean(np.abs(midi_times - aligned_times))
        dtw_score = len(path)
        
        print("✅ Alinhamento realizado com sucesso!")
        print(f"📊 DTW Path Length: {dtw_score}")
        print(f"⏱️  Mean Temporal Error: {temporal_error:.4f} seconds")
        print(f"🎵 MIDI frames: {len(midi_times)}")
        print(f"🔊 Audio frames: {audio_chroma.shape[1]}")
        
        # Visualiza o alinhamento
        visualize_alignment(midi_file, audio_file, midi_times, aligned_times, path, midi_chroma, audio_chroma, cost_matrix)
        
    except Exception as e:
        print(f"❌ Erro no alinhamento: {e}")
        sys.exit(1)






