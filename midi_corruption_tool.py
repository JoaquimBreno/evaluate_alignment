#!/usr/bin/env python3
"""
MIDI Corruption Tool for Controlled Audio-MIDI Desynchronization

This script applies controlled temporal corruptions to MIDI files to create
desynchronized versions for testing alignment algorithms.

Author: Generated for force_align project
"""

import os
import argparse
import mido
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional
import json
from datetime import datetime


class MIDICorruptor:
    """Class to apply controlled corruptions to MIDI files."""
    
    def __init__(self):
        self.corruption_log = []
    
    def apply_linear_shift(self, midi_file: mido.MidiFile, shift_ticks: int) -> mido.MidiFile:
        """
        Apply a constant shift to all events in the MIDI file.
        
        Args:
            midi_file: Input MIDI file
            shift_ticks: Number of ticks to shift (positive = delay, negative = advance)
        
        Returns:
            Corrupted MIDI file
        """
        new_midi = mido.MidiFile(type=midi_file.type, ticks_per_beat=midi_file.ticks_per_beat)
        
        for track in midi_file.tracks:
            new_track = mido.MidiTrack()
            
            # Add initial delay if positive shift
            if shift_ticks > 0 and len(track) > 0:
                first_msg = track[0].copy()
                first_msg.time += shift_ticks
                new_track.append(first_msg)
                
                # Add remaining messages
                for msg in track[1:]:
                    new_track.append(msg.copy())
            else:
                # For negative shift, we need to be more careful
                accumulated_time = 0
                for i, msg in enumerate(track):
                    new_msg = msg.copy()
                    accumulated_time += msg.time
                    
                    if i == 0 and shift_ticks < 0:
                        # Adjust first message time, but don't go negative
                        new_msg.time = max(0, msg.time + shift_ticks)
                    
                    new_track.append(new_msg)
            
            new_midi.tracks.append(new_track)
        
        return new_midi
    
    def apply_progressive_shift(self, midi_file: mido.MidiFile, 
                              start_shift: int, end_shift: int) -> mido.MidiFile:
        """
        Apply a progressive shift that changes linearly throughout the file.
        
        Args:
            midi_file: Input MIDI file
            start_shift: Initial shift in ticks
            end_shift: Final shift in ticks
        
        Returns:
            Corrupted MIDI file
        """
        new_midi = mido.MidiFile(type=midi_file.type, ticks_per_beat=midi_file.ticks_per_beat)
        
        # Calculate total duration first
        total_ticks = 0
        for track in midi_file.tracks:
            track_ticks = sum(msg.time for msg in track)
            total_ticks = max(total_ticks, track_ticks)
        
        for track in midi_file.tracks:
            new_track = mido.MidiTrack()
            current_time = 0
            
            for msg in track:
                current_time += msg.time
                
                # Calculate progressive shift
                if total_ticks > 0:
                    progress = current_time / total_ticks
                    current_shift = start_shift + (end_shift - start_shift) * progress
                else:
                    current_shift = start_shift
                
                new_msg = msg.copy()
                if msg.time > 0:
                    new_msg.time = max(0, int(msg.time + current_shift * (msg.time / max(1, current_time))))
                
                new_track.append(new_msg)
            
            new_midi.tracks.append(new_track)
        
        return new_midi
    
    def apply_section_shifts(self, midi_file: mido.MidiFile, 
                           shifts: List[Tuple[int, int, int]]) -> mido.MidiFile:
        """
        Apply different shifts to different sections of the MIDI file.
        
        Args:
            midi_file: Input MIDI file
            shifts: List of (start_tick, end_tick, shift_amount) tuples
        
        Returns:
            Corrupted MIDI file
        """
        new_midi = mido.MidiFile(type=midi_file.type, ticks_per_beat=midi_file.ticks_per_beat)
        
        for track in midi_file.tracks:
            new_track = mido.MidiTrack()
            current_time = 0
            
            for msg in track:
                current_time += msg.time
                
                # Find applicable shift for current time
                applicable_shift = 0
                for start_tick, end_tick, shift_amount in shifts:
                    if start_tick <= current_time <= end_tick:
                        applicable_shift = shift_amount
                        break
                
                new_msg = msg.copy()
                if msg.time > 0 and applicable_shift != 0:
                    new_msg.time = max(0, msg.time + applicable_shift)
                
                new_track.append(new_msg)
            
            new_midi.tracks.append(new_track)
        
        return new_midi
    
    def apply_random_jitter(self, midi_file: mido.MidiFile, 
                          max_jitter: int, seed: Optional[int] = None) -> mido.MidiFile:
        """
        Apply random jitter to event timings.
        
        Args:
            midi_file: Input MIDI file
            max_jitter: Maximum jitter amount in ticks (±)
            seed: Random seed for reproducibility
        
        Returns:
            Corrupted MIDI file
        """
        if seed is not None:
            np.random.seed(seed)
        
        new_midi = mido.MidiFile(type=midi_file.type, ticks_per_beat=midi_file.ticks_per_beat)
        
        for track in midi_file.tracks:
            new_track = mido.MidiTrack()
            
            for msg in track:
                new_msg = msg.copy()
                if msg.time > 0:
                    jitter = np.random.randint(-max_jitter, max_jitter + 1)
                    new_msg.time = max(0, msg.time + jitter)
                
                new_track.append(new_msg)
            
            new_midi.tracks.append(new_track)
        
        return new_midi
    
    def generate_filename(self, original_path: str, corruption_type: str, 
                         parameters: dict) -> str:
        """
        Generate descriptive filename for corrupted MIDI file.
        
        Args:
            original_path: Path to original MIDI file
            corruption_type: Type of corruption applied
            parameters: Parameters used for corruption
        
        Returns:
            New filename with corruption description
        """
        original_name = Path(original_path).stem
        extension = Path(original_path).suffix
        
        if corruption_type == "linear_shift":
            shift = parameters["shift_ticks"]
            suffix = f"_linear_shift_{shift}ticks"
        
        elif corruption_type == "progressive_shift":
            start = parameters["start_shift"]
            end = parameters["end_shift"]
            suffix = f"_progressive_shift_{start}to{end}ticks"
        
        elif corruption_type == "section_shifts":
            shifts_desc = []
            for start, end, shift in parameters["shifts"]:
                shifts_desc.append(f"{start}-{end}_{shift}")
            suffix = f"_section_shifts_{'_'.join(shifts_desc)}"
        
        elif corruption_type == "random_jitter":
            max_jitter = parameters["max_jitter"]
            seed = parameters.get("seed", "noseed")
            suffix = f"_jitter_{max_jitter}ticks_seed{seed}"
        
        else:
            suffix = f"_corrupted_{corruption_type}"
        
        return f"{original_name}{suffix}{extension}"
    
    def corrupt_midi_file(self, input_path: str, output_dir: str, 
                         corruption_configs: List[dict]) -> List[str]:
        """
        Apply multiple corruptions to a MIDI file and save results.
        
        Args:
            input_path: Path to input MIDI file
            output_dir: Directory to save corrupted files
            corruption_configs: List of corruption configurations
        
        Returns:
            List of output file paths
        """
        try:
            midi_file = mido.MidiFile(input_path)
        except Exception as e:
            print(f"Error loading MIDI file {input_path}: {e}")
            return []
        
        output_paths = []
        
        for config in corruption_configs:
            corruption_type = config["type"]
            parameters = config["parameters"]
            
            try:
                if corruption_type == "linear_shift":
                    corrupted_midi = self.apply_linear_shift(midi_file, parameters["shift_ticks"])
                
                elif corruption_type == "progressive_shift":
                    corrupted_midi = self.apply_progressive_shift(
                        midi_file, parameters["start_shift"], parameters["end_shift"]
                    )
                
                elif corruption_type == "section_shifts":
                    corrupted_midi = self.apply_section_shifts(midi_file, parameters["shifts"])
                
                elif corruption_type == "random_jitter":
                    corrupted_midi = self.apply_random_jitter(
                        midi_file, parameters["max_jitter"], parameters.get("seed")
                    )
                
                else:
                    print(f"Unknown corruption type: {corruption_type}")
                    continue
                
                # Generate output filename
                output_filename = self.generate_filename(input_path, corruption_type, parameters)
                output_path = os.path.join(output_dir, output_filename)
                
                # Save corrupted MIDI
                corrupted_midi.save(output_path)
                output_paths.append(output_path)
                
                # Log corruption
                self.corruption_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "input_file": input_path,
                    "output_file": output_path,
                    "corruption_type": corruption_type,
                    "parameters": parameters
                })
                
                print(f"Created: {output_filename}")
                
            except Exception as e:
                print(f"Error applying {corruption_type} to {input_path}: {e}")
        
        return output_paths
    
    def save_corruption_log(self, log_path: str):
        """Save corruption log to JSON file."""
        with open(log_path, 'w') as f:
            json.dump(self.corruption_log, f, indent=2)


def create_default_corruptions():
    """Create a set of default corruption configurations."""
    return [
        # Linear shifts
        {
            "type": "linear_shift",
            "parameters": {"shift_ticks": 50}
        },
        {
            "type": "linear_shift",
            "parameters": {"shift_ticks": -30}
        },
        {
            "type": "linear_shift",
            "parameters": {"shift_ticks": 100}
        },
        
        # Progressive shifts
        {
            "type": "progressive_shift",
            "parameters": {"start_shift": 0, "end_shift": 80}
        },
        {
            "type": "progressive_shift",
            "parameters": {"start_shift": 50, "end_shift": -50}
        },
        
        # Section shifts (example: shift beginning and middle differently)
        {
            "type": "section_shifts",
            "parameters": {
                "shifts": [
                    (0, 1000, 50),      # First 1000 ticks: +50 shift
                    (1000, 3000, -20),  # Ticks 1000-3000: -20 shift
                    (3000, 999999, 30)  # Rest: +30 shift
                ]
            }
        },
        
        # Random jitter
        {
            "type": "random_jitter",
            "parameters": {"max_jitter": 25, "seed": 42}
        },
        {
            "type": "random_jitter",
            "parameters": {"max_jitter": 50, "seed": 123}
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="MIDI Corruption Tool for Controlled Desynchronization")
    parser.add_argument("input_path", help="Path to input MIDI file or directory")
    parser.add_argument("-o", "--output", default="corrupted_midi", 
                       help="Output directory for corrupted files")
    parser.add_argument("-c", "--config", help="JSON file with corruption configurations")
    parser.add_argument("--log", default="corruption_log.json", 
                       help="Path to save corruption log")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load corruption configurations
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            corruptions = json.load(f)
    else:
        corruptions = create_default_corruptions()
        print("Using default corruption configurations")
    
    # Initialize corruptor
    corruptor = MIDICorruptor()
    
    # Process input
    if os.path.isfile(args.input_path):
        # Single file
        input_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        # Directory - find all MIDI files
        input_files = []
        for ext in ['.mid', '.midi']:
            input_files.extend(Path(args.input_path).glob(f"*{ext}"))
            input_files.extend(Path(args.input_path).glob(f"**/*{ext}"))
        input_files = [str(f) for f in input_files]
    else:
        print(f"Input path not found: {args.input_path}")
        return
    
    print(f"Found {len(input_files)} MIDI files to process")
    
    # Process each file
    total_created = 0
    for input_file in input_files:
        print(f"\nProcessing: {os.path.basename(input_file)}")
        output_paths = corruptor.corrupt_midi_file(input_file, args.output, corruptions)
        total_created += len(output_paths)
    
    # Save corruption log
    corruptor.save_corruption_log(args.log)
    
    print(f"\n✅ Corruption complete!")
    print(f"📁 Created {total_created} corrupted MIDI files in: {args.output}")
    print(f"📋 Corruption log saved to: {args.log}")


if __name__ == "__main__":
    main()
