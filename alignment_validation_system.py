#!/usr/bin/env python3
"""
Sistema de Validação de Alinhamento MIDI-Audio
Compara F0 de MIDI normal vs áudio e MIDI corrompido vs áudio
Usa janelas de equivalência musical para detectar alinhamento
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import correlate
from scipy.stats import wilcoxon, pearsonr
from sklearn.metrics import roc_auc_score, roc_curve
import argparse
from tqdm import tqdm
import json

class PitchWindowSystem:
    """Sistema de janelas de equivalência musical"""
    
    def __init__(self, tolerance=0.08):
        """
        Inicializa o sistema de janelas
        
        Args:
            tolerance (float): Tolerância em % para cada nota (padrão: 8%)
        """
        self.tolerance = tolerance
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.pitch_windows = self._create_pitch_windows()
    
    def _create_pitch_windows(self):
        """Cria janelas de equivalência para notas musicais com sobreposição"""
        pitch_windows = {}
        
        # Para cada oitava de C1 (32Hz) até C8 (4186Hz)
        for octave in range(1, 9):
            for note_idx, note in enumerate(self.note_names):
                # Frequência central da nota (temperamento igual)
                midi_note = octave * 12 + note_idx
                center_freq = 440 * (2 ** ((midi_note - 69) / 12))
                
                # Janela com tolerância especificada
                freq_min = center_freq * (1 - self.tolerance)
                freq_max = center_freq * (1 + self.tolerance)
                
                note_name = f"{note}{octave}"
                pitch_windows[note_name] = {
                    'center': center_freq,
                    'min': freq_min,
                    'max': freq_max,
                    'midi': midi_note
                }
        
        return pitch_windows
    
    def freq_to_pitch_class(self, frequency):
        """Converte frequência para classe de altura musical"""
        if frequency <= 0:
            return []
        
        matches = []
        for note_name, window in self.pitch_windows.items():
            if window['min'] <= frequency <= window['max']:
                matches.append(note_name)
        
        return matches
    
    def create_pitch_sequence(self, f0_array):
        """Converte sequência F0 para sequência de classes de altura"""
        pitch_sequence = []
        
        for freq in f0_array:
            if freq > 0:
                matches = self.freq_to_pitch_class(freq)
                if matches:
                    # Se múltiplas correspondências, pega a primeira (mais provável)
                    pitch_sequence.append(matches[0])
                else:
                    pitch_sequence.append('UNKNOWN')
            else:
                pitch_sequence.append('SILENCE')
        
        return pitch_sequence

class AlignmentMetrics:
    """Calcula métricas de alinhamento entre sequências F0"""
    
    def __init__(self, pitch_system):
        self.pitch_system = pitch_system
    
    def calculate_pitch_edit_distance(self, seq1, seq2):
        """Distância de edição entre sequências de altura"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Inicialização
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Preenchimento da matriz
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],    # Deleção
                                      dp[i][j-1],     # Inserção
                                      dp[i-1][j-1])   # Substituição
        
        return dp[m][n]
    
    def calculate_interval_sequence(self, pitch_sequence):
        """Calcula sequência de intervalos musicais (em semitons)"""
        intervals = []
        prev_midi = None
        
        for pitch in pitch_sequence:
            if pitch in self.pitch_system.pitch_windows:
                current_midi = self.pitch_system.pitch_windows[pitch]['midi']
                if prev_midi is not None:
                    interval = current_midi - prev_midi
                    intervals.append(interval)
                prev_midi = current_midi
            elif pitch == 'SILENCE':
                prev_midi = None
        
        return intervals
    
    def calculate_interval_similarity(self, intervals1, intervals2):
        """Calcula similaridade entre sequências de intervalos"""
        if not intervals1 or not intervals2:
            return 0.0
        
        # Usa correlação cruzada normalizada
        i1 = np.array(intervals1)
        i2 = np.array(intervals2)
        
        if len(i1) == 0 or len(i2) == 0:
            return 0.0
        
        # Correlação cruzada
        correlation = correlate(i1, i2, mode='full')
        max_corr = np.max(correlation)
        
        # Normaliza pela energia das sequências
        norm_factor = np.sqrt(np.sum(i1**2) * np.sum(i2**2))
        
        return max_corr / norm_factor if norm_factor > 0 else 0.0
    
    def calculate_dtw_distance(self, seq1, seq2, max_length=1000):
        """Calcula Dynamic Time Warping distance (versão otimizada)"""
        # Limita o tamanho das sequências para evitar sobrecarga
        if len(seq1) > max_length:
            seq1 = seq1[:max_length]
        if len(seq2) > max_length:
            seq2 = seq2[:max_length]
            
        try:
            from dtaidistance import dtw
            return dtw.distance(seq1, seq2)
        except ImportError:
            # Fallback para versão simplificada e otimizada
            return self._simple_dtw_optimized(seq1, seq2)
    
    def _simple_dtw_optimized(self, seq1, seq2):
        """Implementação otimizada de DTW com limitação de banda"""
        m, n = len(seq1), len(seq2)
        
        # Se as sequências são muito grandes, usa amostragem
        if m > 500 or n > 500:
            step = max(m // 500, n // 500, 1)
            seq1 = seq1[::step]
            seq2 = seq2[::step]
            m, n = len(seq1), len(seq2)
        
        # DTW com banda limitada (Sakoe-Chiba band)
        band_width = min(50, max(m, n) // 4)
        
        dtw_matrix = np.full((m + 1, n + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, m + 1):
            start_j = max(1, i - band_width)
            end_j = min(n + 1, i + band_width + 1)
            
            for j in range(start_j, end_j):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        return dtw_matrix[m, n]
    
    def calculate_all_metrics(self, f0_midi, f0_audio, max_samples=5000):
        """Calcula métricas essenciais: DTW e distância de edição temporal"""
        # Limita o tamanho dos dados para evitar sobrecarga
        if len(f0_midi) > max_samples:
            step = len(f0_midi) // max_samples
            f0_midi = f0_midi[::step]
            f0_audio = f0_audio[::step]
        
        # Alinha os tamanhos das sequências
        min_len = min(len(f0_midi), len(f0_audio))
        f0_midi = f0_midi[:min_len]
        f0_audio = f0_audio[:min_len]
        
        # Converte para sequências de altura
        midi_pitches = self.pitch_system.create_pitch_sequence(f0_midi)
        audio_pitches = self.pitch_system.create_pitch_sequence(f0_audio)
        
        # Alinha os tamanhos das sequências de altura
        min_len = min(len(midi_pitches), len(audio_pitches))
        midi_pitches = midi_pitches[:min_len]
        audio_pitches = audio_pitches[:min_len]
        
        metrics = {}
        
        # 1. Distância de edição temporal (principal métrica)
        metrics['edit_distance'] = self.calculate_pitch_edit_distance(midi_pitches, audio_pitches)
        metrics['normalized_edit_distance'] = metrics['edit_distance'] / max(len(midi_pitches), len(audio_pitches)) if len(midi_pitches) > 0 else 1.0
        
        # 2. DTW distance (segunda métrica principal)
        midi_nonzero = f0_midi[f0_midi > 0]
        audio_nonzero = f0_audio[f0_audio > 0]
        
        if len(midi_nonzero) > 0 and len(audio_nonzero) > 0:
            try:
                metrics['dtw_distance'] = self.calculate_dtw_distance(midi_nonzero, audio_nonzero)
            except:
                metrics['dtw_distance'] = np.inf
        else:
            metrics['dtw_distance'] = np.inf
        
        # 3. Métricas auxiliares simples
        total_active = sum(1 for m, a in zip(midi_pitches, audio_pitches) 
                          if m != 'SILENCE' or a != 'SILENCE')
        both_active = sum(1 for m, a in zip(midi_pitches, audio_pitches) 
                         if m != 'SILENCE' and a != 'SILENCE')
        
        metrics['temporal_overlap'] = both_active / total_active if total_active > 0 else 0
        metrics['midi_activity_ratio'] = np.sum(f0_midi > 0) / len(f0_midi)
        metrics['audio_activity_ratio'] = np.sum(f0_audio > 0) / len(f0_audio)
        
        return metrics

class CorruptionAnalyzer:
    """Analisa informações de corrupção a partir dos nomes dos arquivos"""
    
    @staticmethod
    def extract_corruption_info(filename):
        """
        Extrai informações de corrupção do nome do arquivo
        
        Exemplos:
        - linear_shift_50ticks -> {'type': 'linear_shift', 'amount': 50, 'unit': 'ticks'}
        - jitter_100ms -> {'type': 'jitter', 'amount': 100, 'unit': 'ms'}
        """
        corruption_info = {
            'type': 'normal',
            'amount': 0,
            'unit': '',
            'is_corrupted': False
        }
        
        # Padrões de corrupção
        patterns = [
            r'linear_shift_(\d+)(ticks|ms)',
            r'jitter_(\d+)(ticks|ms)',
            r'tempo_change_(\d+)percent',
            r'pitch_shift_(\d+)(cents|semitones)',
            r'time_stretch_(\d+)percent'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                corruption_info['is_corrupted'] = True
                corruption_info['type'] = pattern.split('_')[0].replace(r'\(', '').replace(r'\d', '').replace(r'\)', '')
                corruption_info['amount'] = int(match.group(1))
                if len(match.groups()) > 1:
                    corruption_info['unit'] = match.group(2)
                break
        
        return corruption_info

class AlignmentValidator:
    """Sistema principal de validação de alinhamento"""
    
    def __init__(self, f0_midi_dir, f0_audio_dir, output_dir="validation_results"):
        self.f0_midi_dir = Path(f0_midi_dir)
        self.f0_audio_dir = Path(f0_audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.pitch_system = PitchWindowSystem()
        self.metrics_calculator = AlignmentMetrics(self.pitch_system)
        self.corruption_analyzer = CorruptionAnalyzer()
        
        self.results = []
    
    def find_file_pairs(self):
        """Encontra pares de arquivos MIDI-Audio para comparação"""
        pairs = []
        
        # Busca todos os arquivos MIDI F0
        midi_files = list(self.f0_midi_dir.glob("*_f0.csv"))
        
        for midi_file in midi_files:
            # Extrai o nome base do arquivo
            base_name = midi_file.stem.replace("_f0", "")
            
            # Remove sufixos de corrupção para encontrar o áudio correspondente
            audio_base = re.sub(r'_(linear_shift|jitter|tempo_change|pitch_shift|time_stretch)_[^_]+', '', base_name)
            
            # Procura o arquivo de áudio correspondente
            audio_file = self.f0_audio_dir / f"{audio_base}_f0.csv"
            
            if audio_file.exists():
                corruption_info = self.corruption_analyzer.extract_corruption_info(str(midi_file))
                
                pairs.append({
                    'midi_file': midi_file,
                    'audio_file': audio_file,
                    'base_name': audio_base,
                    'corruption_info': corruption_info
                })
        
        return pairs
    
    def process_file_pair(self, pair):
        """Processa um par de arquivos MIDI-Audio"""
        try:
            print(f"📂 Processando: {pair['base_name']}")
            print(f"   MIDI: {pair['midi_file'].name}")
            print(f"   Audio: {pair['audio_file'].name}")
            print(f"   Corrupção: {pair['corruption_info']['type']} ({pair['corruption_info']['amount']} {pair['corruption_info']['unit']})")
            
            # Carrega os dados
            print("   📊 Carregando dados F0...")
            midi_df = pd.read_csv(pair['midi_file'])
            audio_df = pd.read_csv(pair['audio_file'])
            
            # Extrai F0
            f0_midi = midi_df['f0'].values
            f0_audio = audio_df['f0'].values
            print(f"   📏 Tamanhos: MIDI={len(f0_midi)}, Audio={len(f0_audio)}")
            
            # Alinha os tamanhos
            min_len = min(len(f0_midi), len(f0_audio))
            f0_midi = f0_midi[:min_len]
            f0_audio = f0_audio[:min_len]
            print(f"   ✂️ Alinhado para: {min_len} amostras")
            
            # Calcula métricas
            print("   🧮 Calculando métricas...")
            metrics = self.metrics_calculator.calculate_all_metrics(f0_midi, f0_audio)
            
            print(f"   ✅ Edit Distance: {metrics['normalized_edit_distance']:.4f}")
            print(f"   ✅ DTW Distance: {metrics['dtw_distance']:.4f}")
            print(f"   ✅ Temporal Overlap: {metrics['temporal_overlap']:.4f}")
            
            # Adiciona informações do arquivo
            result = {
                'base_name': pair['base_name'],
                'midi_file': str(pair['midi_file'].name),
                'audio_file': str(pair['audio_file'].name),
                'corruption_type': pair['corruption_info']['type'],
                'corruption_amount': pair['corruption_info']['amount'],
                'corruption_unit': pair['corruption_info']['unit'],
                'is_corrupted': pair['corruption_info']['is_corrupted'],
                **metrics
            }
            
            print("   ✅ Processamento concluído!\n")
            return result
            
        except Exception as e:
            print(f"   ❌ Erro processando {pair['midi_file']}: {e}\n")
            return None
    
    def run_validation(self):
        """Executa a validação completa"""
        print("🔍 Encontrando pares de arquivos...")
        pairs = self.find_file_pairs()
        print(f"Encontrados {len(pairs)} pares de arquivos")
        
        # Mostra resumo dos tipos de corrupção
        corruption_types = {}
        for pair in pairs:
            ctype = pair['corruption_info']['type']
            corruption_types[ctype] = corruption_types.get(ctype, 0) + 1
        
        print("\n📋 Tipos de corrupção encontrados:")
        for ctype, count in corruption_types.items():
            print(f"   {ctype}: {count} arquivos")
        
        print(f"\n📊 Iniciando processamento de {len(pairs)} pares...")
        print("=" * 60)
        
        success_count = 0
        for i, pair in enumerate(pairs, 1):
            print(f"[{i}/{len(pairs)}] ", end="")
            result = self.process_file_pair(pair)
            if result:
                self.results.append(result)
                success_count += 1
            
            # Mostra progresso a cada 10 arquivos
            if i % 10 == 0:
                print(f"🎯 Progresso: {i}/{len(pairs)} ({success_count} sucessos)")
                print("-" * 40)
        
        print("=" * 60)
        print(f"✅ Processamento concluído: {success_count}/{len(pairs)} pares processados com sucesso")
        
        if success_count == 0:
            print("❌ Nenhum arquivo foi processado com sucesso. Verifique os dados.")
            return
        
        # Salva resultados
        print("\n💾 Salvando resultados...")
        self.save_results()
        
        # Gera análises estatísticas
        print("📈 Gerando análises estatísticas...")
        self.generate_statistical_analysis()
        
        # Gera visualizações
        print("📊 Gerando visualizações...")
        self.generate_visualizations()
    
    def save_results(self):
        """Salva os resultados em CSV"""
        if not self.results:
            print("❌ Nenhum resultado para salvar")
            return
        
        # Salva resultados completos
        results_df = pd.DataFrame(self.results)
        results_path = self.output_dir / "alignment_metrics_complete.csv"
        results_df.to_csv(results_path, index=False)
        print(f"💾 Resultados salvos em: {results_path}")
        
        # Salva resumo por tipo de corrupção (apenas métricas essenciais)
        summary = results_df.groupby(['corruption_type', 'is_corrupted']).agg({
            'normalized_edit_distance': ['mean', 'std'],
            'dtw_distance': ['mean', 'std'],
            'temporal_overlap': ['mean', 'std']
        }).round(4)
        
        summary_path = self.output_dir / "alignment_summary_by_corruption.csv"
        summary.to_csv(summary_path)
        print(f"📋 Resumo salvo em: {summary_path}")
    
    def generate_statistical_analysis(self):
        """Gera análise estatística comparando normal vs corrompido"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Separa dados normais e corrompidos
        normal_data = df[df['is_corrupted'] == False]
        corrupted_data = df[df['is_corrupted'] == True]
        
        if len(normal_data) == 0 or len(corrupted_data) == 0:
            print("⚠️ Não há dados suficientes para análise estatística")
            return
        
        # Métricas para análise (apenas as essenciais)
        metrics_to_analyze = [
            'normalized_edit_distance', 'dtw_distance', 'temporal_overlap'
        ]
        
        statistical_results = []
        
        for metric in metrics_to_analyze:
            normal_values = normal_data[metric].dropna()
            corrupted_values = corrupted_data[metric].dropna()
            
            if len(normal_values) > 0 and len(corrupted_values) > 0:
                # Teste de Wilcoxon
                try:
                    if metric in ['normalized_edit_distance', 'dtw_distance']:
                        # Para métricas onde menor é melhor
                        stat, p_val = wilcoxon(corrupted_values, normal_values, alternative='greater')
                    else:
                        # Para métricas onde maior é melhor
                        stat, p_val = wilcoxon(normal_values, corrupted_values, alternative='greater')
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(normal_values) - 1) * normal_values.var() + 
                                         (len(corrupted_values) - 1) * corrupted_values.var()) / 
                                        (len(normal_values) + len(corrupted_values) - 2))
                    effect_size = abs(normal_values.mean() - corrupted_values.mean()) / pooled_std
                    
                    statistical_results.append({
                        'metric': metric,
                        'normal_mean': normal_values.mean(),
                        'normal_std': normal_values.std(),
                        'corrupted_mean': corrupted_values.mean(),
                        'corrupted_std': corrupted_values.std(),
                        'wilcoxon_statistic': stat,
                        'p_value': p_val,
                        'effect_size': effect_size,
                        'significant': p_val < 0.05,
                        'n_normal': len(normal_values),
                        'n_corrupted': len(corrupted_values)
                    })
                    
                except Exception as e:
                    print(f"Erro na análise estatística para {metric}: {e}")
        
        # Salva análise estatística
        stats_df = pd.DataFrame(statistical_results)
        stats_path = self.output_dir / "statistical_analysis.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"📈 Análise estatística salva em: {stats_path}")
        
        # Salva resumo da análise
        summary_text = self.generate_analysis_summary(stats_df)
        summary_path = self.output_dir / "analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        print(f"📝 Resumo da análise salvo em: {summary_path}")
    
    def generate_analysis_summary(self, stats_df):
        """Gera resumo textual da análise"""
        summary = "=== RESUMO DA ANÁLISE DE ALINHAMENTO ===\n\n"
        
        significant_metrics = stats_df[stats_df['significant'] == True]
        
        if len(significant_metrics) > 0:
            summary += f"✅ {len(significant_metrics)} métricas mostraram diferença significativa (p < 0.05):\n\n"
            
            for _, row in significant_metrics.iterrows():
                summary += f"📊 {row['metric'].upper()}:\n"
                summary += f"   Normal: {row['normal_mean']:.4f} ± {row['normal_std']:.4f}\n"
                summary += f"   Corrompido: {row['corrupted_mean']:.4f} ± {row['corrupted_std']:.4f}\n"
                summary += f"   P-valor: {row['p_value']:.6f}\n"
                summary += f"   Effect size: {row['effect_size']:.4f}\n\n"
        else:
            summary += "❌ Nenhuma métrica mostrou diferença significativa\n\n"
        
        # Recomendações
        summary += "=== RECOMENDAÇÕES ===\n\n"
        
        best_metrics = stats_df.nlargest(3, 'effect_size')
        summary += "🏆 Melhores métricas para detectar desalinhamento:\n"
        for i, (_, row) in enumerate(best_metrics.iterrows(), 1):
            summary += f"{i}. {row['metric']} (effect size: {row['effect_size']:.4f})\n"
        
        return summary
    
    def generate_visualizations(self):
        """Gera visualizações dos resultados"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Configuração de plots
        plt.style.use('default')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Análise de Métricas de Alinhamento MIDI-Audio', fontsize=16)
        
        metrics_to_plot = [
            'normalized_edit_distance', 'dtw_distance', 'temporal_overlap'
        ]
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Boxplot comparando normal vs corrompido
            normal_data = df[df['is_corrupted'] == False][metric].dropna()
            corrupted_data = df[df['is_corrupted'] == True][metric].dropna()
            
            if len(normal_data) > 0 and len(corrupted_data) > 0:
                ax.boxplot([normal_data, corrupted_data], 
                          labels=['Normal', 'Corrompido'],
                          patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
                
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "alignment_metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Visualizações salvas em: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Sistema de Validação de Alinhamento MIDI-Audio")
    parser.add_argument("--midi-dir", default="f0_data/midi", help="Diretório com F0s de MIDI")
    parser.add_argument("--audio-dir", default="f0_data/audio", help="Diretório com F0s de áudio")
    parser.add_argument("--output-dir", default="validation_results", help="Diretório de saída")
    
    args = parser.parse_args()
    
    print("🚀 Iniciando Sistema de Validação de Alinhamento")
    print(f"📁 MIDI F0s: {args.midi_dir}")
    print(f"📁 Audio F0s: {args.audio_dir}")
    print(f"📁 Saída: {args.output_dir}")
    
    validator = AlignmentValidator(args.midi_dir, args.audio_dir, args.output_dir)
    validator.run_validation()
    
    print("🎯 Validação completa!")

if __name__ == "__main__":
    main()
