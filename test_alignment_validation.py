#!/usr/bin/env python3
"""
Teste rápido do sistema de validação de alinhamento
Processa apenas alguns arquivos para validar o funcionamento
"""

import pandas as pd
import numpy as np
from alignment_validation_system import AlignmentValidator, PitchWindowSystem, AlignmentMetrics, CorruptionAnalyzer

def quick_test():
    """Teste rápido com alguns arquivos específicos"""
    
    print("🧪 Teste rápido do sistema de validação")
    
    # Arquivos específicos para teste
    test_files = [
        {
            'midi_normal': 'f0_data/midi/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1_f0.csv',
            'midi_corrupted': 'f0_data/midi/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1_linear_shift_50ticks_f0.csv',
            'audio': 'f0_data/audio/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1_f0.csv'
        }
    ]
    
    # Inicializa componentes
    pitch_system = PitchWindowSystem()
    metrics_calculator = AlignmentMetrics(pitch_system)
    corruption_analyzer = CorruptionAnalyzer()
    
    results = []
    
    for test_file in test_files:
        print(f"\n📊 Processando: {test_file['audio']}")
        
        try:
            # Carrega dados
            audio_df = pd.read_csv(test_file['audio'])
            f0_audio = audio_df['f0'].values
            
            # Processa MIDI normal
            if test_file['midi_normal']:
                print("   🎵 Analisando MIDI normal...")
                midi_normal_df = pd.read_csv(test_file['midi_normal'])
                f0_midi_normal = midi_normal_df['f0'].values
                
                # Alinha tamanhos
                min_len = min(len(f0_midi_normal), len(f0_audio))
                f0_midi_normal_aligned = f0_midi_normal[:min_len]
                f0_audio_aligned = f0_audio[:min_len]
                
                # Calcula métricas
                metrics_normal = metrics_calculator.calculate_all_metrics(f0_midi_normal_aligned, f0_audio_aligned)
                
                result_normal = {
                    'file_type': 'normal',
                    'base_name': 'MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1',
                    'corruption_type': 'normal',
                    'is_corrupted': False,
                    **metrics_normal
                }
                results.append(result_normal)
                
                print(f"      ✅ Pitch accuracy: {metrics_normal['pitch_accuracy']:.4f}")
                print(f"      ✅ Temporal overlap: {metrics_normal['temporal_overlap']:.4f}")
                print(f"      ✅ Edit distance: {metrics_normal['normalized_edit_distance']:.4f}")
            
            # Processa MIDI corrompido
            if test_file['midi_corrupted']:
                print("   🎵 Analisando MIDI corrompido...")
                midi_corrupted_df = pd.read_csv(test_file['midi_corrupted'])
                f0_midi_corrupted = midi_corrupted_df['f0'].values
                
                # Alinha tamanhos
                min_len = min(len(f0_midi_corrupted), len(f0_audio))
                f0_midi_corrupted_aligned = f0_midi_corrupted[:min_len]
                f0_audio_aligned = f0_audio[:min_len]
                
                # Calcula métricas
                metrics_corrupted = metrics_calculator.calculate_all_metrics(f0_midi_corrupted_aligned, f0_audio_aligned)
                
                # Extrai informação de corrupção
                corruption_info = corruption_analyzer.extract_corruption_info(test_file['midi_corrupted'])
                
                result_corrupted = {
                    'file_type': 'corrupted',
                    'base_name': 'MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1',
                    'corruption_type': corruption_info['type'],
                    'corruption_amount': corruption_info['amount'],
                    'is_corrupted': True,
                    **metrics_corrupted
                }
                results.append(result_corrupted)
                
                print(f"      ✅ Pitch accuracy: {metrics_corrupted['pitch_accuracy']:.4f}")
                print(f"      ✅ Temporal overlap: {metrics_corrupted['temporal_overlap']:.4f}")
                print(f"      ✅ Edit distance: {metrics_corrupted['normalized_edit_distance']:.4f}")
                print(f"      ✅ Corruption: {corruption_info['type']} {corruption_info['amount']} {corruption_info['unit']}")
        
        except Exception as e:
            print(f"   ❌ Erro: {e}")
    
    # Salva resultados do teste
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('test_alignment_results.csv', index=False)
        print(f"\n💾 Resultados do teste salvos em: test_alignment_results.csv")
        
        # Mostra comparação
        print("\n📊 COMPARAÇÃO NORMAL vs CORROMPIDO:")
        print("=" * 50)
        
        normal_result = next((r for r in results if not r['is_corrupted']), None)
        corrupted_result = next((r for r in results if r['is_corrupted']), None)
        
        if normal_result and corrupted_result:
            metrics_to_compare = ['pitch_accuracy', 'temporal_overlap', 'normalized_edit_distance', 'pearson_correlation']
            
            for metric in metrics_to_compare:
                normal_val = normal_result.get(metric, 0)
                corrupted_val = corrupted_result.get(metric, 0)
                diff = abs(normal_val - corrupted_val)
                
                print(f"{metric:25} | Normal: {normal_val:8.4f} | Corrompido: {corrupted_val:8.4f} | Diff: {diff:8.4f}")
            
            # Determina se a métrica consegue detectar a diferença
            significant_diff = abs(normal_result['pitch_accuracy'] - corrupted_result['pitch_accuracy']) > 0.1
            print(f"\n🎯 Detecção de desalinhamento: {'✅ SIM' if significant_diff else '❌ NÃO'}")
    
    print("\n🎉 Teste concluído!")

if __name__ == "__main__":
    quick_test()
