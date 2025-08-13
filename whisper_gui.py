# MUST be first - suppress warnings before any imports that trigger them
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*torchaudio.*")
warnings.filterwarnings("ignore", message=".*Triton kernels.*")
warnings.filterwarnings("ignore", message=".*torch.backends.*")
warnings.filterwarnings("ignore", message=".*list_audio_backends.*")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")
warnings.filterwarnings("ignore", module="pyannote.*")
warnings.filterwarnings("ignore", module="torchaudio.*")

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import whisper
import threading
import os
import json
import re
import torch
import tempfile
import subprocess
import sys
import io
import argparse
from datetime import timedelta
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()

class WhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Transcription Tool")
        self.root.geometry("800x700")
        
        self.model = None
        self.current_model_name = None  # Track which model is loaded
        self.transcription_result = None
        self.diarization_pipeline = None
        self.diarization_result = None
        self.temp_files = []  # Track temporary files for cleanup
        
        # Progress tracking
        self.progress_value = 0
        
        self.setup_ui()
        
        # Ensure cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        ttk.Label(main_frame, text="File Selection:", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_var, state="readonly")
        self.file_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        self.browse_btn.grid(row=0, column=1)
        
        ttk.Label(main_frame, text="Model Selection:", font=('Arial', 12, 'bold')).grid(
            row=2, column=0, sticky=tk.W, pady=(10, 5))
        
        self.model_var = tk.StringVar(value="large-v3")
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        for i, model in enumerate(models):
            ttk.Radiobutton(model_frame, text=model, variable=self.model_var, 
                          value=model).grid(row=0, column=i, padx=5)
        
        ttk.Label(main_frame, text="Options:", font=('Arial', 12, 'bold')).grid(
            row=4, column=0, sticky=tk.W, pady=(10, 5))
        
        options_frame = ttk.Frame(main_frame)
        options_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.timestamps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include timestamps", 
                       variable=self.timestamps_var).grid(row=0, column=0, sticky=tk.W)
        
        self.word_timestamps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Word-level timestamps", 
                       variable=self.word_timestamps_var).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        
        self.speaker_diarization_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Speaker diarization", 
                       variable=self.speaker_diarization_var).grid(row=1, column=0, sticky=tk.W)
        
        self.clean_format_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Clean format (segments only)", 
                       variable=self.clean_format_var).grid(row=1, column=1, sticky=tk.W, padx=(20, 0))
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        self.transcribe_btn = ttk.Button(button_frame, text="Transcribe", 
                                       command=self.start_transcription)
        self.transcribe_btn.grid(row=0, column=0, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Save Transcript", 
                                 command=self.save_transcript, state="disabled")
        self.save_btn.grid(row=0, column=1, padx=5)
        
        self.format_btn = ttk.Button(button_frame, text="Format Segments", 
                                   command=self.format_segments, state="disabled")
        self.format_btn.grid(row=0, column=2, padx=5)
        
        # Current task progress bar
        ttk.Label(main_frame, text="Current Task:", font=('Arial', 10)).grid(
            row=7, column=0, sticky=tk.W, pady=(10, 2))
        
        self.current_progress = ttk.Progressbar(main_frame, mode='determinate', maximum=100)
        self.current_progress.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Overall progress bar  
        ttk.Label(main_frame, text="Overall Progress:", font=('Arial', 10)).grid(
            row=9, column=0, sticky=tk.W, pady=(5, 2))
        
        self.progress = ttk.Progressbar(main_frame, mode='determinate', maximum=100)
        self.progress.grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.grid(row=11, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(main_frame, text="Transcript:", font=('Arial', 12, 'bold')).grid(
            row=12, column=0, sticky=tk.W, pady=(10, 5))
        
        self.result_text = scrolledtext.ScrolledText(main_frame, height=20, wrap=tk.WORD)
        self.result_text.grid(row=13, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        main_frame.rowconfigure(13, weight=1)
        
    def browse_file(self):
        file_types = [
            ("All supported", "*.mp4 *.avi *.mov *.mkv *.mp3 *.wav *.m4a *.flac"),
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("Audio files", "*.mp3 *.wav *.m4a *.flac"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select audio/video file",
            filetypes=file_types
        )
        
        if filename:
            self.file_var.set(filename)
    
    def format_timestamp(self, seconds):
        return str(timedelta(seconds=int(seconds)))
    
    def update_progress(self, value):
        """Safely update overall progress bar from worker thread"""
        def _update():
            self.progress['value'] = value
            self.progress_value = value
        self.root.after(0, _update)
    
    def update_current_progress(self, value):
        """Safely update current task progress bar from worker thread"""
        def _update():
            self.current_progress['value'] = value
        self.root.after(0, _update)
    
    def cleanup_temp_files(self):
        """Clean up any temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                # Silently continue if cleanup fails
                pass
        
        # Clear the list after cleanup
        self.temp_files = []
    
    def on_closing(self):
        """Handle application closing and cleanup"""
        self.cleanup_temp_files()
        # Destroy the window
        self.root.destroy()
    
    def convert_to_wav_for_diarization(self, file_path):
        """Convert unsupported formats to WAV for speaker diarization"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # If already supported format, return original
        if file_ext in ['.wav', '.mp3', '.m4a', '.flac']:
            return file_path
            
        # For video files, extract audio to temporary WAV
        try:
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            
            
            # Use ffmpeg to extract audio
            cmd = ['ffmpeg', '-i', file_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', temp_wav.name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Track the temporary file for cleanup
                self.temp_files.append(temp_wav.name)
                return temp_wav.name
            else:
                os.unlink(temp_wav.name)
                return None
                
        except Exception as e:
            return None
    
    def start_transcription(self):
        if not self.file_var.get():
            messagebox.showerror("Error", "Please select a file first.")
            return
        
        if not os.path.exists(self.file_var.get()):
            messagebox.showerror("Error", "Selected file does not exist.")
            return
        
        # Reset previous results and clean up any remaining temp files
        self.transcription_result = None
        self.diarization_result = None
        self.cleanup_temp_files()
        
        self.transcribe_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.format_btn.config(state="disabled")
        self.progress['value'] = 0
        self.current_progress['value'] = 0
        self.status_label.config(text="Transcribing...", foreground="blue")
        self.result_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.transcribe_audio)
        thread.daemon = True
        thread.start()
    
    def transcribe_audio(self):
        try:
            file_path = self.file_var.get()
            
            self.root.after(0, lambda: self.status_label.config(text="Loading models...", foreground="blue"))
            
            if self.model is None or self.current_model_name != self.model_var.get():
                self.model = whisper.load_model(self.model_var.get())
                self.current_model_name = self.model_var.get()
            
            if self.speaker_diarization_var.get():
                if self.diarization_pipeline is None:
                    try:
                        self.root.after(0, lambda: self.status_label.config(text="Loading speaker diarization model...", foreground="blue"))
                        hf_token = os.getenv('TOKEN')
                        
                        # Try with token first, then fallback to True (uses huggingface-cli login)
                        try:
                            self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                                                use_auth_token=hf_token)
                        except Exception as token_error:
                            # Fallback to huggingface-cli login
                            self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                                                use_auth_token=True)
                        
                        # Move to CUDA if available
                        if torch.cuda.is_available():
                            self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                            
                    except Exception as e:
                        self.root.after(0, lambda: self.status_label.config(text="Speaker diarization unavailable, continuing with transcription...", foreground="orange"))
                        self.diarization_pipeline = False
                        self.diarization_result = None
                
                if self.diarization_pipeline and self.diarization_pipeline is not False:
                    try:
                        self.root.after(0, lambda: self.status_label.config(text="Performing speaker diarization...", foreground="blue"))
                        
                        # Convert file if needed for speaker diarization
                        diarization_file = self.convert_to_wav_for_diarization(file_path)
                        if diarization_file:
                            # Use simulated progress for diarization since pyannote doesn't output progress by default
                            import time
                            
                            def simulate_diarization_progress():
                                """Simulate diarization progress over estimated time"""
                                steps = [
                                    ("Speaker diarization: loading models...", 0, 0.5),
                                    ("Speaker diarization: segmentation...", 10, 1.0),
                                    ("Speaker diarization: embeddings...", 25, 2.0),
                                    ("Speaker diarization: clustering...", 40, 1.0),
                                    ("Speaker diarization: finalizing...", 48, 0.5),
                                ]
                                
                                for status, progress, duration in steps:
                                    if hasattr(self, '_diarization_cancelled'):
                                        break
                                    # Update UI safely from background thread
                                    def update_ui(s=status, p=progress):
                                        self.status_label.config(text=s, foreground="blue")
                                        self.update_current_progress(p * 2)  # Scale to 0-100% for current task
                                        self.update_progress(p)  # Scale to 0-50% for overall
                                    
                                    self.root.after(0, update_ui)
                                    time.sleep(duration)
                            
                            # Start simulated progress
                            self._diarization_cancelled = False
                            progress_thread = threading.Thread(target=simulate_diarization_progress)
                            progress_thread.daemon = True
                            progress_thread.start()
                            
                            try:
                                self.diarization_result = self.diarization_pipeline(diarization_file)
                                # Cancel simulation and set to complete
                                self._diarization_cancelled = True
                                self.update_progress(50)
                                self.update_current_progress(100)
                                self.root.after(0, lambda: self.status_label.config(
                                    text="Speaker diarization complete!", foreground="blue"))
                                time.sleep(0.5)  # Brief pause to show completion
                            except Exception as e:
                                self._diarization_cancelled = True
                                raise e
                            
                            # Temporary file will be cleaned up automatically via self.temp_files tracking
                        else:
                            self.diarization_result = None
                            
                    except Exception as e:
                        self.diarization_result = None
            
            self.root.after(0, lambda: self.status_label.config(text="Processing audio...", foreground="blue"))
            # Reset current progress for Whisper task
            self.update_current_progress(0)
            
            # Capture progress by intercepting stderr (where tqdm outputs)
            class ProgressCapture:
                def __init__(self, gui_ref):
                    self.gui_ref = gui_ref
                    self.original_stderr = sys.stderr
                    self.buffer = io.StringIO()
                
                def write(self, text):
                    # Parse tqdm progress from text like "100%|████| 612/612 [00:01<00:00, 327.88frames/s]"
                    if '|' in text and '%' in text:
                        try:
                            # Extract percentage
                            percentage_match = re.search(r'(\d+)%', text)
                            if percentage_match:
                                percentage = int(percentage_match.group(1))
                                # Map Whisper progress based on whether diarization was used
                                if self.gui_ref.speaker_diarization_var.get() and self.gui_ref.diarization_result:
                                    # Map to 50-100% range (after diarization 0-50%)
                                    mapped_percentage = 50 + int(percentage * 0.5)
                                else:
                                    # Use full 0-100% range if no diarization
                                    mapped_percentage = percentage
                                self.gui_ref.update_progress(mapped_percentage)
                                # Update current task progress (always use full percentage for current task)
                                self.gui_ref.update_current_progress(percentage)
                                # Update status with more detail
                                if 'frames/s' in text:
                                    frames_match = re.search(r'(\d+)/(\d+)', text)
                                    if frames_match:
                                        current, total = frames_match.groups()
                                        self.gui_ref.root.after(0, lambda: self.gui_ref.status_label.config(
                                            text=f"Processing audio... ({current}/{total} frames, {percentage}%)", 
                                            foreground="blue"))
                        except:
                            pass
                    
                    # Still write to original stderr for any other output
                    self.original_stderr.write(text)
                
                def flush(self):
                    self.original_stderr.flush()
            
            # Use progress capture during transcription
            progress_capture = ProgressCapture(self)
            original_stderr = sys.stderr
            
            try:
                sys.stderr = progress_capture
                result = self.model.transcribe(
                    file_path,
                    word_timestamps=self.word_timestamps_var.get(),
                    verbose=False
                )
            finally:
                sys.stderr = original_stderr
            
            # Set both progress bars to 100% when done
            self.update_progress(100)
            self.update_current_progress(100)
            
            self.transcription_result = result
            
            self.root.after(0, self.display_results)
            
        except Exception as e:
            # Clean up temporary files on error
            self.cleanup_temp_files()
            error_msg = f"Transcription failed: {str(e)}"
            self.root.after(0, lambda: self.handle_error(error_msg))
    
    def get_speaker_at_time(self, timestamp):
        if not self.diarization_result:
            return None
        
        # First, try exact match
        for turn, _, speaker in self.diarization_result.itertracks(yield_label=True):
            if turn.start <= timestamp <= turn.end:
                return speaker
        
        # If no exact match, find the closest speaker segment with conservative fallback
        closest_speaker = None
        min_distance = float('inf')
        
        for turn, _, speaker in self.diarization_result.itertracks(yield_label=True):
            # Calculate distance to this segment
            if timestamp < turn.start:
                distance = turn.start - timestamp
            elif timestamp > turn.end:
                distance = timestamp - turn.end
            else:
                distance = 0  # Should have been caught above, but just in case
                return speaker
            
            if distance < min_distance:
                min_distance = distance
                closest_speaker = speaker
        
        # Conservative fallback: only assign if very close (≤ 0.8s)
        # This handles small timing misalignments without being too aggressive
        if min_distance <= 0.8:
            return closest_speaker
        
        return None
    
    def display_results(self):
        self.result_text.delete(1.0, tk.END)
        
        if self.clean_format_var.get():
            self.display_clean_format()
        elif self.timestamps_var.get() and 'segments' in self.transcription_result:
            for segment in self.transcription_result['segments']:
                start_time = self.format_timestamp(segment['start'])
                end_time = self.format_timestamp(segment['end'])
                text = segment['text'].strip()
                
                speaker = None
                if self.speaker_diarization_var.get() and self.diarization_result:
                    speaker = self.get_speaker_at_time(segment['start'])
                
                speaker_prefix = f"[{speaker}] " if speaker is not None else ""
                
                if self.word_timestamps_var.get() and 'words' in segment:
                    self.result_text.insert(tk.END, f"[{start_time} - {end_time}] {speaker_prefix}\n")
                    for word in segment['words']:
                        word_start = self.format_timestamp(word['start'])
                        word_end = self.format_timestamp(word['end'])
                        word_text = word['word']
                        word_speaker = None
                        if self.speaker_diarization_var.get() and self.diarization_result:
                            word_speaker = self.get_speaker_at_time(word['start'])
                        word_speaker_prefix = f"[{word_speaker}] " if word_speaker is not None else ""
                        self.result_text.insert(tk.END, f"  {word_start}-{word_end}: {word_speaker_prefix}{word_text}\n")
                    self.result_text.insert(tk.END, f"Full segment: {speaker_prefix}{text}\n\n")
                else:
                    self.result_text.insert(tk.END, f"[{start_time} - {end_time}] {speaker_prefix}{text}\n\n")
        else:
            self.result_text.insert(tk.END, self.transcription_result['text'])
        
        # Keep both progress bars at 100% for completion
        self.progress['value'] = 100
        self.current_progress['value'] = 100
        self.status_label.config(text="Transcription complete!", foreground="green")
        self.transcribe_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.format_btn.config(state="normal")
    
    def handle_error(self, error_message):
        self.progress['value'] = 0
        self.current_progress['value'] = 0
        self.status_label.config(text="Error occurred", foreground="red")
        self.transcribe_btn.config(state="normal")
        messagebox.showerror("Error", error_message)
    
    def save_transcript(self):
        if not self.transcription_result:
            messagebox.showerror("Error", "No transcript to save.")
            return
        
        file_types = [
            ("Text files", "*.txt"),
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save transcript",
            defaultextension=".txt",
            filetypes=file_types
        )
        
        if filename:
            try:
                if filename.lower().endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(self.transcription_result, f, indent=2, ensure_ascii=False)
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(self.result_text.get(1.0, tk.END))
                
                messagebox.showinfo("Success", f"Transcript saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def display_clean_format(self):
        if not self.transcription_result or 'segments' not in self.transcription_result:
            return
        
        for segment in self.transcription_result['segments']:
            start_time = self.format_timestamp(segment['start'])
            end_time = self.format_timestamp(segment['end'])
            text = ' '.join(segment['text'].strip().split())
            
            speaker = None
            if self.speaker_diarization_var.get() and self.diarization_result:
                speaker = self.get_speaker_at_time(segment['start'])
            
            speaker_prefix = f"[{speaker}] " if speaker is not None else ""
            self.result_text.insert(tk.END, f"[{start_time} - {end_time}] {speaker_prefix}{text}\n")
    
    def format_segments(self):
        if not self.transcription_result:
            messagebox.showerror("Error", "No transcript to format.")
            return
        
        content = self.result_text.get(1.0, tk.END)
        
        segments = []
        
        # Check if this is clean format (checkbox was checked) or detailed format
        is_clean_format = self.clean_format_var.get() or 'Full segment:' not in content
        
        if is_clean_format:
            # For clean format, each line is a complete segment with timestamp and speaker
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and re.match(r'\[\d{1,2}:\d{2}:\d{2} - \d{1,2}:\d{2}:\d{2}\]', line):
                    segments.append(line)
        else:
            # For detailed format, extract from "Full segment:" lines
            patterns = [
                # Pattern 1: With speaker in Full segment line
                r'\[(\d{1,2}:\d{2}:\d{2}) - (\d{1,2}:\d{2}:\d{2})\].*?Full segment: (\[SPEAKER_\d+\] )?(.+?)(?=\n\n|\n\[|\Z)',
                # Pattern 2: Speaker info before Full segment
                r'\[(\d{1,2}:\d{2}:\d{2}) - (\d{1,2}:\d{2}:\d{2})\] (\[SPEAKER_\d+\] )?.*?Full segment: (.+?)(?=\n\n|\n\[|\Z)',
                # Pattern 3: Just Full segment without speaker
                r'\[(\d{1,2}:\d{2}:\d{2}) - (\d{1,2}:\d{2}:\d{2})\].*?Full segment: (.+?)(?=\n\n|\n\[|\Z)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                
                if matches:
                    for match in matches:
                        if len(match) == 4:  # Has speaker info
                            start_time, end_time, speaker_part, text = match
                            clean_text = ' '.join(text.strip().split())
                            speaker_prefix = speaker_part if speaker_part and speaker_part.strip() else ""
                            
                            # If no speaker in Full segment line, try to find it from original data
                            if not speaker_prefix and self.speaker_diarization_var.get() and self.diarization_result:
                                # Convert timestamp back to seconds to find speaker
                                time_parts = start_time.split(':')
                                start_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                                speaker = self.get_speaker_at_time(start_seconds)
                                speaker_prefix = f"[{speaker}] " if speaker is not None else ""
                            
                            segments.append(f"[{start_time} - {end_time}] {speaker_prefix}{clean_text}")
                        elif len(match) == 3:  # No speaker info
                            start_time, end_time, text = match
                            clean_text = ' '.join(text.strip().split())
                            
                            # Try to find speaker from original data
                            speaker_prefix = ""
                            if self.speaker_diarization_var.get() and self.diarization_result:
                                time_parts = start_time.split(':')
                                start_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                                speaker = self.get_speaker_at_time(start_seconds)
                                speaker_prefix = f"[{speaker}] " if speaker is not None else ""
                            
                            segments.append(f"[{start_time} - {end_time}] {speaker_prefix}{clean_text}")
                    break  # Use first pattern that works
        
        if not segments:
            messagebox.showwarning("Warning", "No segments found to format.")
            return
        
        file_types = [
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save formatted segments",
            defaultextension=".txt",
            filetypes=file_types
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    for segment in segments:
                        f.write(segment + '\n')
                
                messagebox.showinfo("Success", f"Processed {len(segments)} segments and saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save formatted segments: {str(e)}")

def run_cli(args):
    """Run transcription in CLI mode"""
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return 1
    
    print(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model)
    
    diarization_pipeline = None
    if args.speaker_diarization:
        try:
            print("Loading speaker diarization model...")
            hf_token = os.getenv('TOKEN')
            try:
                diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                                use_auth_token=hf_token)
            except Exception:
                diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                                use_auth_token=True)
            if torch.cuda.is_available():
                diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
        except Exception as e:
            print(f"Warning: Speaker diarization unavailable: {e}")
            diarization_pipeline = None
    
    diarization_result = None
    if diarization_pipeline:
        try:
            print("Performing speaker diarization...")
            # Convert file if needed for diarization
            file_ext = os.path.splitext(args.input)[1].lower()
            if file_ext in ['.wav', '.mp3', '.m4a', '.flac']:
                diarization_file = args.input
            else:
                # Convert video to audio
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_wav.close()
                cmd = ['ffmpeg', '-i', args.input, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', temp_wav.name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    diarization_file = temp_wav.name
                else:
                    print("Warning: Could not convert file for diarization")
                    diarization_file = args.input
            
            diarization_result = diarization_pipeline(diarization_file)
            
            # Cleanup temp file if created
            if diarization_file != args.input:
                try:
                    os.unlink(diarization_file)
                except:
                    pass
        except Exception as e:
            print(f"Warning: Speaker diarization failed: {e}")
            diarization_result = None
    
    print("Processing audio...")
    result = model.transcribe(
        args.input,
        word_timestamps=args.word_timestamps,
        verbose=True  # Show progress in CLI mode
    )
    
    def get_speaker_at_time_cli(timestamp):
        if not diarization_result:
            return None
        
        # First, try exact match
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if turn.start <= timestamp <= turn.end:
                return speaker
        
        # Conservative fallback
        closest_speaker = None
        min_distance = float('inf')
        
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if timestamp < turn.start:
                distance = turn.start - timestamp
            elif timestamp > turn.end:
                distance = timestamp - turn.end
            else:
                distance = 0
                return speaker
            
            if distance < min_distance:
                min_distance = distance
                closest_speaker = speaker
        
        if min_distance <= 0.8:
            return closest_speaker
        
        return None
    
    def format_timestamp_cli(seconds):
        return str(timedelta(seconds=int(seconds)))
    
    # Generate output
    output_lines = []
    
    if args.clean_format:
        for segment in result['segments']:
            start_time = format_timestamp_cli(segment['start'])
            end_time = format_timestamp_cli(segment['end'])
            text = ' '.join(segment['text'].strip().split())
            
            speaker = None
            if args.speaker_diarization and diarization_result:
                speaker = get_speaker_at_time_cli(segment['start'])
            
            speaker_prefix = f"[{speaker}] " if speaker is not None else ""
            output_lines.append(f"[{start_time} - {end_time}] {speaker_prefix}{text}")
    elif args.timestamps and 'segments' in result:
        for segment in result['segments']:
            start_time = format_timestamp_cli(segment['start'])
            end_time = format_timestamp_cli(segment['end'])
            text = segment['text'].strip()
            
            speaker = None
            if args.speaker_diarization and diarization_result:
                speaker = get_speaker_at_time_cli(segment['start'])
            
            speaker_prefix = f"[{speaker}] " if speaker is not None else ""
            output_lines.append(f"[{start_time} - {end_time}] {speaker_prefix}{text}")
    else:
        output_lines.append(result['text'])
    
    # Output results
    output_text = '\n'.join(output_lines)
    
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"\nTranscript saved to: {args.output}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return 1
    else:
        print("\n" + "="*50)
        print("TRANSCRIPT:")
        print("="*50)
        print(output_text)
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Whisper Transcription Tool with Speaker Diarization')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode')
    parser.add_argument('--input', type=str, help='Input audio/video file')
    parser.add_argument('--model', type=str, default='large-v3', 
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper model to use')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--no-timestamps', action='store_true', help='Disable timestamps')
    parser.add_argument('--no-word-timestamps', action='store_true', help='Disable word-level timestamps')
    parser.add_argument('--no-speaker-diarization', action='store_true', help='Disable speaker diarization')
    parser.add_argument('--clean-format', action='store_true', help='Use clean segment format only')
    
    args = parser.parse_args()
    
    if args.cli:
        if not args.input:
            print("Error: --input is required in CLI mode")
            return 1
        
        # Set boolean flags correctly
        args.timestamps = not args.no_timestamps
        args.word_timestamps = not args.no_word_timestamps
        args.speaker_diarization = not args.no_speaker_diarization
        
        return run_cli(args)
    else:
        # GUI mode
        root = tk.Tk()
        app = WhisperGUI(root)
        root.mainloop()
        return 0

if __name__ == "__main__":
    sys.exit(main())