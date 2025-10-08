# MIDI Corruption Tool

Este script permite corromper controladamente arquivos MIDI para criar desincronização temporal controlada entre áudio e MIDI, útil para testar algoritmos de alinhamento.

## Funcionalidades

### Tipos de Corrupção

1. **Linear Shift** (`linear_shift`)
   - Aplica um deslocamento constante a todos os eventos
   - Parâmetros: `shift_ticks` (positivo = atraso, negativo = avanço)

2. **Progressive Shift** (`progressive_shift`)
   - Aplica um deslocamento que muda linearmente ao longo do arquivo
   - Parâmetros: `start_shift`, `end_shift`

3. **Section Shifts** (`section_shifts`)
   - Aplica diferentes deslocamentos a diferentes seções do arquivo
   - Parâmetros: `shifts` - lista de `[start_tick, end_tick, shift_amount]`

4. **Random Jitter** (`random_jitter`)
   - Aplica variação aleatória aos tempos dos eventos
   - Parâmetros: `max_jitter`, `seed` (opcional)

### Nomenclatura dos Arquivos

Os arquivos corrompidos são salvos com nomes descritivos que indicam o tipo de corrupção:

- `original_linear_shift_50ticks.midi` - Shift linear de +50 ticks
- `original_progressive_shift_0to100ticks.midi` - Shift progressivo de 0 a 100 ticks
- `original_section_shifts_0-1000_50_1000-3000_-20.midi` - Shifts por seção
- `original_jitter_25ticks_seed42.midi` - Jitter aleatório com seed

## Uso

### Uso Básico
```bash
# Corromper um arquivo MIDI com configurações padrão
python midi_corruption_tool.py input.midi

# Corromper todos os MIDIs de uma pasta
python midi_corruption_tool.py data/ -o corrupted_output/
```

### Uso Avançado
```bash
# Usar configurações customizadas
python midi_corruption_tool.py input.midi -c custom_config.json -o output_dir/

# Especificar arquivo de log
python midi_corruption_tool.py input.midi --log my_corruption_log.json
```

### Configurações Customizadas

Crie um arquivo JSON com suas configurações de corrupção:

```json
{
  "corruptions": [
    {
      "type": "linear_shift",
      "parameters": {"shift_ticks": 75}
    },
    {
      "type": "section_shifts",
      "parameters": {
        "shifts": [[0, 1000, 50], [2000, 4000, -30]]
      }
    }
  ]
}
```

## Exemplos de Uso

### Exemplo 1: Corrupções Básicas
```bash
# Aplicar corrupções padrão a um arquivo
python midi_corruption_tool.py rockylutador.mid -o corrupted/
```

### Exemplo 2: Corrupção de Dataset
```bash
# Corromper todos os MIDIs da pasta data/
python midi_corruption_tool.py data/ -o corrupted_dataset/
```

### Exemplo 3: Configuração Personalizada
```bash
# Usar configurações específicas
python midi_corruption_tool.py data/ -c example_corruption_config.json -o test_corruptions/
```

## Log de Corrupções

O script gera um log JSON com detalhes de todas as corrupções aplicadas:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "input_file": "original.midi",
  "output_file": "corrupted/original_linear_shift_50ticks.midi",
  "corruption_type": "linear_shift",
  "parameters": {"shift_ticks": 50}
}
```

## Dependências

```bash
pip install mido numpy
```

## Estrutura de Saída

```
corrupted_midi/
├── file1_linear_shift_50ticks.midi
├── file1_progressive_shift_0to100ticks.midi
├── file1_section_shifts_0-1000_50.midi
├── file1_jitter_25ticks_seed42.midi
└── ...
```

Cada arquivo corrompido mantém a estrutura MIDI original, apenas com os tempos dos eventos modificados de acordo com a corrupção aplicada.
