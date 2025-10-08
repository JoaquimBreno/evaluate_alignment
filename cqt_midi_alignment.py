import numpy as np
import librosa
import matplotlib.pyplot as plt
import pretty_midi
import miditok
from pathlib import Path
from scipy.spatial.distance import cdist
from fastdtw import fastdtw
from scipy.interpolate import interp1d
import colorsys
import matplotlib.patches as patches

def load_midi_tokens(midi_file):
    """
    Carrega um arquivo MIDI e o converte em tokens usando miditok
    
    Parâmetros:
    - midi_file: caminho para o arquivo MIDI
    
    Retorna:
    - tokens: objeto TokSequence do miditok
    - tokenizer: tokenizer utilizado
    """
    # Configurar o tokenizer MIDILike
    config = miditok.TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(1, 4): 8},  # 8 ticks por beat
        nb_velocities=32,
        additional_tokens={'Tempo', 'Program'},
        use_programs=True
    )
    tokenizer = miditok.MIDILike(config)
    
    # Tokenizar o MIDI
    tokens = tokenizer(Path(midi_file))
    
    return tokens, tokenizer

def extract_notes_from_tokens(tokens, tokenizer, bpm):
    """
    Extrai informações de notas a partir dos tokens MIDI
    
    Parâmetros:
    - tokens: objeto TokSequence do miditok
    - tokenizer: tokenizer utilizado
    - bpm: tempo em BPM
    
    Retorna:
    - completed_notes: lista de objetos Note com informações das notas
    """
    # Estrutura para armazenar informações das notas
    class Note:
        def __init__(self, pitch, start_time, velocity=None, program=None):
            self.pitch = pitch
            self.start_ticks = start_time
            self.end_ticks = None
            self.velocity = velocity
            self.program = program
            self.aligned_start_time = None
            self.aligned_end_time = None
            
        @property
        def start_time(self):
            return ticks_to_seconds(self.start_ticks, tokenizer.time_division, bpm)
            
        @property
        def end_time(self):
            return ticks_to_seconds(self.end_ticks, tokenizer.time_division, bpm) if self.end_ticks is not None else None
            
        @property
        def duration(self):
            return self.end_time - self.start_time if self.end_time is not None else None
    
    def ticks_to_seconds(ticks, ticks_per_beat, tempo):
        """Converte ticks para segundos usando o tempo"""
        beats = ticks / ticks_per_beat
        seconds = (beats * tempo) / 1_000_000  # Converter microssegundos para segundos
        return seconds
    
    # Processar os eventos para extrair as notas
    active_notes = {}  # pitch -> Note
    completed_notes = []
    
    # Processar cada evento para construir as notas
    for event in tokens.events:
        if event.type_ == 'NoteOn':
            pitch = event.value
            program = event.program
            
            # Procurar a velocidade no próximo evento
            idx = tokens.events.index(event)
            velocity = None
            if idx + 1 < len(tokens.events) and tokens.events[idx + 1].type_ == 'Velocity':
                velocity = tokens.events[idx + 1].value
            
            if pitch not in active_notes:  # Ignorar se a nota já está ativa
                active_notes[pitch] = Note(pitch, event.time, velocity, program)
                
        elif event.type_ == 'NoteOff':
            pitch = event.value
            if pitch in active_notes:  # Só processar se a nota está ativa
                note = active_notes[pitch]
                note.end_ticks = event.time
                if note.end_ticks > note.start_ticks:  # Só adicionar se a nota tem duração positiva
                    completed_notes.append(note)
                del active_notes[pitch]
    
    return completed_notes

def generate_midi_chroma(notes, duration, sr=22050, hop_length=512):
    """
    Gera uma representação chroma a partir das notas MIDI
    
    Parâmetros:
    - notes: lista de objetos Note
    - duration: duração total em segundos
    - sr: taxa de amostragem (default: 22050 Hz)
    - hop_length: tamanho do hop em samples (default: 512)
    
    Retorna:
    - midi_chroma: representação chroma do MIDI
    - midi_times: array de tempos correspondentes aos frames
    """
    # Calcula o número de frames
    n_frames = int(duration * sr / hop_length) + 1
    
    # Cria uma matriz chroma vazia (12 notas x n_frames)
    midi_chroma = np.zeros((12, n_frames))
    
    # Calcula os tempos correspondentes aos frames
    midi_times = np.arange(n_frames) * hop_length / sr
    
    # Preenche a matriz chroma com as notas
    for note in notes:
        if note.end_time is None:
            continue
            
        # Converte o pitch MIDI para índice chroma (0-11)
        chroma_idx = note.pitch % 12
        
        # Converte os tempos de início e fim para índices de frame
        start_frame = int(note.start_time * sr / hop_length)
        end_frame = int(note.end_time * sr / hop_length)
        
        # Limita os índices ao tamanho da matriz
        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = max(0, min(end_frame, n_frames - 1))
        
        # Adiciona a nota à matriz chroma
        if start_frame <= end_frame:
            # Usa a velocidade da nota como intensidade, ou 1.0 se não houver velocidade
            intensity = note.velocity / 127.0 if note.velocity is not None else 1.0
            midi_chroma[chroma_idx, start_frame:end_frame+1] += intensity
    
    # Normaliza a matriz chroma
    midi_chroma = librosa.util.normalize(midi_chroma, axis=0)
    
    return midi_chroma, midi_times

def align_midi_to_audio_with_cqt(midi_file, audio_file, cqt_file, bpm, sr=22050, hop_length=512):
    """
    Alinha um arquivo MIDI a um arquivo de áudio usando DTW e CQT
    
    Parâmetros:
    - midi_file: caminho para o arquivo MIDI
    - audio_file: caminho para o arquivo de áudio
    - cqt_file: caminho para o arquivo CQT (.npy)
    - bpm: tempo em BPM (em microssegundos por beat)
    - sr: taxa de amostragem (default: 22050 Hz)
    - hop_length: tamanho do hop em samples (default: 512)
    
    Retorna:
    - tokens: tokens MIDI
    - notes: notas extraídas dos tokens
    - time_mapping: função de mapeamento de tempo MIDI para tempo de áudio
    - path: caminho do alinhamento DTW
    """
    # Carrega os tokens MIDI
    tokens, tokenizer = load_midi_tokens(midi_file)
    
    # Extrai as notas dos tokens
    notes = extract_notes_from_tokens(tokens, tokenizer, bpm)
    
    # Carrega o CQT
    cqt = np.load(cqt_file)
    
    # Calcula os tempos do CQT
    cqt_times = np.arange(cqt.shape[1]) * hop_length / sr
    
    # Calcula a duração total do áudio
    audio_duration = cqt_times[-1]
    
    # Gera o chroma do MIDI
    midi_chroma, midi_times = generate_midi_chroma(notes, audio_duration, sr, hop_length)
    
    # Converte o CQT para chroma
    # Primeiro, calculamos a magnitude do CQT
    cqt_magnitude = np.abs(cqt)
    
    # Reduz o CQT para 12 bins por oitava (chroma)
    n_octaves = cqt.shape[0] // 12
    audio_chroma = np.zeros((12, cqt.shape[1]))
    
    for i in range(12):
        for octave in range(n_octaves):
            audio_chroma[i] += cqt_magnitude[i + octave * 12]
    
    # Normaliza o chroma do áudio
    audio_chroma = librosa.util.normalize(audio_chroma, axis=0)
    
    # Calcula a matriz de custo entre as duas representações chroma
    cost_matrix = cdist(midi_chroma.T, audio_chroma.T, metric='cosine')
    
    # Aplica o DTW
    distance, path = fastdtw(midi_chroma.T, audio_chroma.T, dist=lambda x, y: np.linalg.norm(x - y))
    
    # Converte o caminho em arrays numpy
    path = np.array(path)
    
    # Extrai os índices do caminho
    midi_idx, audio_idx = path.T
    
    # Cria uma função de mapeamento de tempo MIDI para tempo de áudio
    midi_path_times = midi_times[midi_idx]
    audio_path_times = cqt_times[audio_idx]
    
    # Usa interpolação para criar uma função de mapeamento contínua
    time_mapping = interp1d(midi_path_times, audio_path_times, bounds_error=False, fill_value="extrapolate")
    
    # Aplica o mapeamento de tempo às notas
    for note in notes:
        note.aligned_start_time = time_mapping(note.start_time)
        if note.end_time is not None:
            note.aligned_end_time = time_mapping(note.end_time)
    
    return tokens, notes, time_mapping, path, midi_chroma, audio_chroma, cost_matrix, midi_times, cqt_times

def visualize_alignment(midi_file, audio_file, path, midi_chroma, audio_chroma, cost_matrix, midi_times, audio_times):
    """
    Visualiza o alinhamento entre o MIDI e o áudio
    
    Parâmetros:
    - midi_file: caminho para o arquivo MIDI
    - audio_file: caminho para o arquivo de áudio
    - path: caminho do alinhamento DTW
    - midi_chroma: representação chroma do MIDI
    - audio_chroma: representação chroma do áudio
    - cost_matrix: matriz de custo entre as duas representações
    - midi_times: array de tempos do MIDI
    - audio_times: array de tempos do áudio
    """
    # Cria a figura com subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot da matriz de custo e do caminho DTW
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    librosa.display.specshow(cost_matrix.T, x_axis='time', y_axis='time',
                           cmap='viridis')
    plt.colorbar(format='%+2.0f')
    plt.title('Matriz de Custo e Caminho DTW')
    
    # Plota o caminho DTW
    plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=1)
    
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
    
    # Extrai os tempos do caminho
    midi_path_times = midi_times[path[:, 0]]
    audio_path_times = audio_times[path[:, 1]]
    
    plt.plot(midi_path_times, audio_path_times, 'b-')
    plt.plot([0, max(midi_times)], [0, max(audio_times)], 'r--')
    plt.xlabel('Tempo MIDI (s)')
    plt.ylabel('Tempo Áudio (s)')
    plt.title('Função de Warping')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_aligned_notes(notes, cqt, sr=22050, hop_length=512, start_time=0, end_time=None, figsize=(20, 12)):
    """
    Visualiza as notas alinhadas junto com o CQT
    
    Parâmetros:
    - notes: lista de objetos Note com tempos alinhados
    - cqt: matriz CQT
    - sr: taxa de amostragem (default: 22050 Hz)
    - hop_length: tamanho do hop em samples (default: 512)
    - start_time: tempo inicial em segundos (default: 0)
    - end_time: tempo final em segundos (default: None, mostra até o final)
    - figsize: tupla com tamanho da figura (width, height)
    """
    # Calcula os tempos do CQT
    times = np.arange(cqt.shape[1]) * hop_length / sr
    
    # Se end_time não foi especificado, usar o tempo total do CQT
    if end_time is None:
        end_time = times[-1]
    
    # Encontrar os índices do CQT para o intervalo de tempo desejado
    start_idx = max(0, np.searchsorted(times, start_time))
    end_idx = min(len(times), np.searchsorted(times, end_time))
    
    # Ajustar os tempos de início e fim para corresponder aos frames do CQT
    start_time = times[start_idx]
    end_time = times[end_idx-1]
    
    # Filtrar notas pelo intervalo de tempo desejado
    visible_notes = [
        note for note in notes
        if (note.aligned_end_time >= start_time and note.aligned_start_time <= end_time)
    ]
    
    if not visible_notes:
        print(f"Nenhuma nota encontrada no intervalo de {start_time:.3f}s a {end_time:.3f}s")
        return
    
    # Calcula frequências para o eixo y
    n_bins_per_octave = 12
    n_octaves = cqt.shape[0] // n_bins_per_octave
    fmin = librosa.midi_to_hz(21)  # A0
    freqs = librosa.cqt_frequencies(
        n_bins=n_bins_per_octave * n_octaves,
        fmin=fmin,
        bins_per_octave=n_bins_per_octave
    )
    
    # Criar a figura com GridSpec para melhor controle do layout
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5], width_ratios=[19, 1])
    
    # Criar os subplots
    ax1 = fig.add_subplot(gs[0, 0])  # CQT
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Piano roll
    cax = fig.add_subplot(gs[0, 1])  # Colorbar
    
    # Plot do CQT
    C_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    img = librosa.display.specshow(
        C_db[:, start_idx:end_idx],
        x_coords=times[start_idx:end_idx],
        y_coords=freqs,
        ax=ax1,
        x_axis=None,
        y_axis='cqt_hz',
        cmap='magma'
    )
    
    # Configurar o eixo x do CQT manualmente
    def format_time(x, p):
        return f"{x:.2f}s"
    
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    ax1.set_xlabel('Tempo (segundos)')
    
    # Adicionar colorbar e título
    fig.colorbar(img, cax=cax, format="%+2.f dB")
    ax1.set_title(f'Constant-Q Transform com Notas Alinhadas ({start_time:.3f}s - {end_time:.3f}s)')
    
    # Garantir que os limites do eixo x estejam corretos para o CQT
    ax1.set_xlim(start_time, end_time)
    
    # Plot do piano roll com notas alinhadas
    def get_program_color(program):
        # Criar cores distintas para diferentes programas
        hue = (program * 0.1) % 1.0
        return colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    
    # Criar as barras para cada nota
    for note in visible_notes:
        color = get_program_color(note.program)
        # Ajustar a altura do retângulo para ser menor
        semitone = 1.0594630943592953  # 2^(1/12), razão entre semitons
        height_factor = 0.6  # Fator para reduzir a altura do retângulo
        
        # Calcular a altura em frequência (um semitom acima e abaixo)
        freq = librosa.midi_to_hz(note.pitch)
        height = freq * (semitone - 1/semitone) * height_factor
        
        if note.velocity:
            # Ajustar a altura baseado na velocidade (entre 60% e 100% da altura padrão)
            height = height * (0.6 + 0.4 * (note.velocity / 127))
        
        visible_start = max(note.aligned_start_time, start_time)
        visible_end = min(note.aligned_end_time, end_time)
        
        rect = patches.Rectangle(
            (visible_start, freq - height/2),
            visible_end - visible_start,
            height,
            facecolor=color,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        ax2.add_patch(rect)
    
    # Configurar o eixo y para mostrar frequências e notas
    ax2.set_yscale('log')
    ax2.set_ylim(freqs[0], freqs[-1])
    
    # Configurar os localizadores de ticks para mostrar todas as notas
    ax2.yaxis.set_major_locator(plt.LogLocator(base=2))
    
    def midi_to_note_name(midi_num):
        """Converte número MIDI para nome da nota"""
        NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note = NOTES[midi_num % 12]
        octave = (midi_num // 12) - 1
        return f"{note}{octave}"
    
    def format_note(y, p):
        # Converter frequência para MIDI note number
        midi_num = int(round(librosa.hz_to_midi(y)))
        # Converter MIDI note number para nome da nota
        return midi_to_note_name(midi_num)
    
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_note))
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_title('Piano Roll com Notas Alinhadas')
    
    # Adicionar legenda
    program_patches = []
    unique_programs = sorted(list(set(note.program for note in visible_notes)))
    for program in unique_programs:
        color = get_program_color(program)
        program_patches.append(patches.Patch(color=color, label=f'Instrumento {program}'))
    ax2.legend(handles=program_patches, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Ajustar o espaçamento entre os subplots
    plt.subplots_adjust(hspace=0.3, right=0.85)
    
    # Imprimir algumas estatísticas
    print(f"\n=== Estatísticas do trecho ({start_time:.3f}s - {end_time:.3f}s) ===")
    print(f"Notas visíveis: {len(visible_notes)}")
    print(f"Instrumentos ativos: {len(unique_programs)}")
    print("\nDistribuição de notas por instrumento:")
    for program in unique_programs:
        notes_by_program = [n for n in visible_notes if n.program == program]
        count = len(notes_by_program)
        if count > 0:
            avg_duration = sum(
                min(n.aligned_end_time, end_time) - max(n.aligned_start_time, start_time)
                for n in notes_by_program
            ) / count
            print(f"Instrumento {program}: {count} notas, duração média: {avg_duration:.3f} segundos")
    
    plt.show()

if __name__ == "__main__":
    # Exemplo de uso
    midi_file = "rockylutador.mid"
    audio_file = "unaligned_big_rocky.mp3"
    cqt_file = "unaligned_big_rocky_cqt.npy"
    
    # Carregar o MIDI para obter o BPM
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    tempo_changes = midi_data.get_tempo_changes()
    tempos = tempo_changes[1]  # Array com os tempos em microsegs/beat
    bpm = max(tempos)  # Usar o tempo máximo
    
    # Realizar o alinhamento
    tokens, notes, time_mapping, path, midi_chroma, audio_chroma, cost_matrix, midi_times, audio_times = align_midi_to_audio_with_cqt(
        midi_file, audio_file, cqt_file, bpm
    )
    
    # Visualizar o alinhamento
    visualize_alignment(midi_file, audio_file, path, midi_chroma, audio_chroma, cost_matrix, midi_times, audio_times)
    
    # Visualizar as notas alinhadas com o CQT
    cqt = np.load(cqt_file)
    visualize_aligned_notes(notes, cqt, start_time=0, end_time=10)
    
    # Também visualizar outro trecho
    visualize_aligned_notes(notes, cqt, start_time=30, end_time=40)







