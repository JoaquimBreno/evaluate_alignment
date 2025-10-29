#!/usr/bin/env python3
"""
Script para gerar F0s de todos os arquivos MIDI e áudio do dataset
Organiza os resultados em f0_data/midi e f0_data/audio
"""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Adiciona o diretório pai ao path para importar os módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'f0_edition_tracker'))

from f0_edition_tracker.f0_from_midi import extract_f0_from_midi
from f0_edition_tracker.torch_fcpe import extract_f0_from_audio

def create_directories():
    """Cria as pastas necessárias para organizar os F0s"""
    base_dir = Path("f0_data")
    midi_dir = base_dir / "midi"
    audio_dir = base_dir / "audio"
    
    midi_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    return midi_dir, audio_dir

def process_single_midi(args):
    """
    Processa um único arquivo MIDI (função para multiprocessing)
    
    Args:
        args: tupla (midi_file, output_dir, method)
    
    Returns:
        tuple: (success, error_msg)
    """
    midi_file, output_dir, method = args
    
    try:
        # Gera nome do arquivo CSV baseado no nome do MIDI
        csv_filename = midi_file.stem + "_f0.csv"
        csv_path = output_dir / csv_filename
        
        # Extrai F0 do MIDI (sem plot para ser mais rápido)
        extract_f0_from_midi(
            str(midi_file), 
            str(csv_path), 
            method=method, 
            time_resolution=0.01,  # 10ms de resolução
            plot=False
        )
        return True, None
        
    except Exception as e:
        return False, f"{midi_file.name}: {str(e)}"

def process_midi_files(midi_directories, output_dir, method="melody", n_workers=5):
    """
    Processa todos os arquivos MIDI das pastas especificadas usando multiprocessing
    
    Args:
        midi_directories: Lista de diretórios contendo arquivos MIDI
        output_dir: Diretório de saída para os CSVs
        method: Método de extração F0 (melody, bass, lowest)
        n_workers: Número de workers paralelos
    """
    print(f"🎵 Processando arquivos MIDI com método: {method}")
    print(f"⚡ Usando {n_workers} workers paralelos")
    
    midi_files = []
    for midi_dir in midi_directories:
        if os.path.exists(midi_dir):
            for ext in ['.mid', '.midi']:
                midi_files.extend(Path(midi_dir).glob(f"*{ext}"))
    
    print(f"Encontrados {len(midi_files)} arquivos MIDI")
    
    # Prepara argumentos para multiprocessing
    args_list = [(midi_file, output_dir, method) for midi_file in midi_files]
    
    success_count = 0
    error_count = 0
    errors = []
    
    # Processa em paralelo
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_midi, args_list),
            total=len(args_list),
            desc="Processando MIDIs"
        ))
    
    # Conta sucessos e erros
    for success, error_msg in results:
        if success:
            success_count += 1
        else:
            error_count += 1
            errors.append(error_msg)
    
    # Mostra erros se houver
    if errors:
        print("❌ Erros encontrados:")
        for error in errors[:5]:  # Mostra apenas os primeiros 5
            print(f"   {error}")
        if len(errors) > 5:
            print(f"   ... e mais {len(errors) - 5} erros")
    
    print(f"✅ MIDIs processados: {success_count} sucessos, {error_count} erros")
    return success_count, error_count

def process_single_audio(args):
    """
    Processa um único arquivo de áudio (função para multiprocessing)
    
    Args:
        args: tupla (audio_file, output_dir)
    
    Returns:
        tuple: (success, error_msg)
    """
    audio_file, output_dir = args
    
    try:
        # Gera nome do arquivo CSV baseado no nome do áudio
        csv_filename = audio_file.stem + "_f0.csv"
        csv_path = output_dir / csv_filename
        
        # Extrai F0 do áudio (sem plot para ser mais rápido)
        extract_f0_from_audio(
            str(audio_file), 
            str(csv_path), 
            plot=False
        )
        return True, None
        
    except Exception as e:
        return False, f"{audio_file.name}: {str(e)}"

def process_audio_files(audio_directory, output_dir, n_workers=5):
    """
    Processa todos os arquivos de áudio do diretório especificado usando multiprocessing
    
    Args:
        audio_directory: Diretório contendo arquivos de áudio
        output_dir: Diretório de saída para os CSVs
        n_workers: Número de workers paralelos
    """
    print("🔊 Processando arquivos de áudio")
    print(f"⚡ Usando {n_workers} workers paralelos")
    
    audio_files = []
    if os.path.exists(audio_directory):
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(Path(audio_directory).glob(f"*{ext}"))
    
    print(f"Encontrados {len(audio_files)} arquivos de áudio")
    
    # Prepara argumentos para multiprocessing
    args_list = [(audio_file, output_dir) for audio_file in audio_files]
    
    success_count = 0
    error_count = 0
    errors = []
    
    # Processa em paralelo
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_audio, args_list),
            total=len(args_list),
            desc="Processando Áudios"
        ))
    
    # Conta sucessos e erros
    for success, error_msg in results:
        if success:
            success_count += 1
        else:
            error_count += 1
            errors.append(error_msg)
    
    # Mostra erros se houver
    if errors:
        print("❌ Erros encontrados:")
        for error in errors[:5]:  # Mostra apenas os primeiros 5
            print(f"   {error}")
        if len(errors) > 5:
            print(f"   ... e mais {len(errors) - 5} erros")
    
    print(f"✅ Áudios processados: {success_count} sucessos, {error_count} erros")
    return success_count, error_count

def main():
    parser = argparse.ArgumentParser(description="Gera F0s de todos os arquivos MIDI e áudio")
    parser.add_argument("--midi-method", "-m", choices=["melody", "bass", "lowest"], 
                       default="melody", help="Método de extração F0 para MIDI")
    parser.add_argument("--midi-only", action="store_true", help="Processar apenas MIDIs")
    parser.add_argument("--audio-only", action="store_true", help="Processar apenas áudios")
    parser.add_argument("--data-dir", default="data", help="Diretório dos dados originais")
    parser.add_argument("--corrupted-dir", default="corrupted_data", help="Diretório dos dados corrompidos")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Número de workers paralelos")
    
    args = parser.parse_args()
    
    print("🚀 Iniciando geração de dataset F0")
    print(f"📁 Dados originais: {args.data_dir}")
    print(f"📁 Dados corrompidos: {args.corrupted_dir}")
    print(f"⚡ Workers paralelos: {args.workers}")
    
    # Cria diretórios de saída
    midi_output_dir, audio_output_dir = create_directories()
    
    total_success = 0
    total_errors = 0
    
    # Processa MIDIs se não for apenas áudio
    if not args.audio_only:
        midi_directories = [args.data_dir, args.corrupted_dir]
        success, errors = process_midi_files(midi_directories, midi_output_dir, args.midi_method, args.workers)
        total_success += success
        total_errors += errors
    
    # Processa áudios se não for apenas MIDI
    if not args.midi_only:
        success, errors = process_audio_files(args.data_dir, audio_output_dir, args.workers)
        total_success += success
        total_errors += errors
    
    print(f"\n🎯 Processamento completo!")
    print(f"✅ Total de sucessos: {total_success}")
    print(f"❌ Total de erros: {total_errors}")
    print(f"📊 F0s salvos em:")
    print(f"   🎵 MIDIs: f0_data/midi/")
    print(f"   🔊 Áudios: f0_data/audio/")

if __name__ == "__main__":
    main()
