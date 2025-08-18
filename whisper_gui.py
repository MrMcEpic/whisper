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
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
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
from dotenv import load_dotenv
import winreg

# Optional import for speaker diarization
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    Pipeline = None
    PYANNOTE_AVAILABLE = False

load_dotenv()

class WhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Transcription Tool")
        self.root.geometry("800x700")

        # Consistent app font (keeps widget sizes identical across themes)
        self.app_font = tkfont.Font(family='Segoe UI', size=10)
        
        self.model = None
        self.current_model_name = None  # Track which model is loaded
        self.transcription_result = None
        self.diarization_pipeline = None
        self.diarization_result = None
        self.temp_files = []  # Track temporary files for cleanup
        
        # Progress tracking
        self.progress_value = 0
        
        # Dark mode setup
        self.dark_mode = tk.BooleanVar()
        self.setup_dark_mode()
        
        self.setup_ui()
        
        # Apply theme to widgets after UI is created
        self.apply_widget_styles()
        
        # Ensure cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def detect_system_dark_mode(self):
        """Detect if Windows is using dark mode"""
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize')
            value, _ = winreg.QueryValueEx(key, 'AppsUseLightTheme')
            winreg.CloseKey(key)
            return value == 0  # 0 = dark mode, 1 = light mode
        except Exception:
            return False  # Default to light mode if detection fails
    
    def setup_dark_mode(self):
        """Initialize dark mode based on system preference"""
        system_dark = self.detect_system_dark_mode()
        self.dark_mode.set(system_dark)
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the current theme (light or dark)"""
        if self.dark_mode.get():
            # Modern dark mode colors (inspired by VS Code dark theme)
            bg_color = "#1e1e1e"           # Main background
            fg_color = "#cccccc"           # Main text
            entry_bg = "#3c3c3c"           # Input fields
            entry_fg = "#ffffff"           # Input text
            button_bg = "#0e639c"          # Button background
            button_fg = "#ffffff"          # Button text
            button_hover = "#1177bb"       # Button hover
            select_bg = "#264f78"          # Selection background
            border_color = "#464647"       # Borders
            
            # Configure root window
            self.root.configure(bg=bg_color)
            
            # Configure ttk styles for dark mode
            style = ttk.Style()
            style.theme_use('clam')
            
            # Configure various ttk widgets with modern colors
            style.configure('TFrame', background=bg_color, borderwidth=0)
            style.configure('TLabel', background=bg_color, foreground=fg_color)
            
            # Modern button styling
            style.configure('App.TButton', 
                          background=button_bg, 
                          foreground=button_fg,
                          borderwidth=1,
                          focuscolor='none',
                          relief='flat',
                          padding=(10, 6),
                          font=self.app_font)
            style.map('App.TButton', 
                     background=[('active', button_hover), ('pressed', select_bg)],
                     relief=[('pressed', 'flat'), ('!pressed', 'flat')])
            
            # Readonly file entry styling (more subtle)
            style.configure('Dark.Readonly.TEntry',
                          fieldbackground="#2a2a2a",  # Darker, more subtle
                          foreground="#999999",       # Grayed out text
                          insertcolor=entry_fg,
                          bordercolor="#464647",
                          lightcolor="#464647",
                          darkcolor="#464647",
                          relief='flat',
                          focuscolor='none')
            
            # Dark Combobox styling with proper borders
            style.configure('Dark.TCombobox',
                     fieldbackground=entry_bg,
                     background=entry_bg,  # ensure editable field is dark too
                          foreground=entry_fg,
                          arrowcolor=fg_color,
                          bordercolor=border_color,
                          lightcolor=border_color,
                          darkcolor=border_color,
                          relief='flat',
                     focuscolor=button_bg,
                     selectbackground=select_bg,
                     selectforeground=entry_fg)
            style.map('Dark.TCombobox',
                 fieldbackground=[('readonly', entry_bg)],
                 background=[('active', entry_bg), ('!active', entry_bg)],
                 foreground=[('readonly', entry_fg)])
            
            # Checkbox and radio button styling
            style.configure('TCheckbutton', 
                          background=bg_color, 
                          foreground=fg_color,
                          focuscolor='none')
            style.map('TCheckbutton',
                      background=[('active', bg_color), ('selected', bg_color), ('disabled', bg_color)],
                      foreground=[('active', fg_color), ('selected', fg_color), ('disabled', '#777777')])
            style.configure('TRadiobutton', 
                          background=bg_color, 
                          foreground=fg_color,
                          focuscolor='none')
            style.map('TRadiobutton',
                      background=[('active', bg_color), ('selected', bg_color), ('disabled', bg_color)],
                      foreground=[('active', fg_color), ('selected', fg_color), ('disabled', '#777777')])
            
            # Progress bar styling
            style.configure('TProgressbar', 
                          background=button_bg,
                          troughcolor=entry_bg,
                          borderwidth=0,
                          lightcolor=button_bg,
                          darkcolor=button_bg)

            # Indicator-less radiobutton style (uses text bullets for crisp look)
            style.layout('IndicatorLess.TRadiobutton', [
                ('Radiobutton.padding', {
                    'children': [
                        ('Radiobutton.focus', {
                            'children': [ ('Radiobutton.label', {'sticky': 'nswe'}) ],
                            'sticky': 'nswe'
                        })
                    ],
                    'sticky': 'nswe'
                })
            ])
            style.configure('IndicatorLess.TRadiobutton',
                            background=bg_color,
                            foreground=fg_color,
                            focuscolor='none',
                            padding=(2, 0))
            
            # ttk scrollbar styles for dark mode
            style.configure(
                "Dark.Vertical.TScrollbar",
                troughcolor=entry_bg,
                background=button_bg,     # slider color
                bordercolor=border_color,
                lightcolor=button_bg,     # remove 3D bevel
                darkcolor=button_bg,
                arrowcolor=fg_color
            )
            style.map(
                "Dark.Vertical.TScrollbar",
                background=[('active', button_hover), ('pressed', select_bg)]
            )

            # Apply the style to the scrollbar if it exists
            if hasattr(self, 'result_vscroll'):
                self.result_vscroll.configure(style="Dark.Vertical.TScrollbar")
            
        else:
            # Light mode with consistent metrics (use clam + custom colors)
            bg_color = "#f5f5f5"
            fg_color = "#222222"
            entry_bg = "#ffffff"
            entry_fg = "#000000"
            # neutral buttons (no bright blue)
            button_bg = "#e1e1e1"
            button_fg = "#222222"
            button_hover = "#d6d6d6"
            select_bg = "#dcdcdc"
            border_color = "#bdbdbd"

            self.root.configure(bg=bg_color)

            style = ttk.Style()
            style.theme_use('clam')

            style.configure('TFrame', background=bg_color, borderwidth=0)
            style.configure('TLabel', background=bg_color, foreground=fg_color)

            # Buttons: keep same metrics as dark, different colors
            style.configure('App.TButton',
                            background=button_bg,
                            foreground=button_fg,
                            borderwidth=1,
                            focuscolor='none',
                            relief='flat',
                            padding=(10, 6),
                            font=self.app_font)
            style.map('App.TButton',
                      background=[('active', button_hover), ('pressed', '#cfcfcf')],
                      relief=[('pressed', 'flat'), ('!pressed', 'flat')])

            style.configure('Light.Readonly.TEntry',
                            fieldbackground=entry_bg,
                            foreground="#444444",
                            insertcolor=entry_fg,
                            bordercolor=border_color,
                            lightcolor=border_color,
                            darkcolor=border_color,
                            relief='flat',
                            focuscolor='none')

            # Combobox
            style.configure('Light.TCombobox',
                            fieldbackground=entry_bg,
                            background=entry_bg,
                            foreground=entry_fg,
                            arrowcolor=fg_color,
                            bordercolor=border_color,
                            lightcolor=border_color,
                            darkcolor=border_color,
                            relief='flat',
                            focuscolor=button_bg,
                            selectbackground=select_bg,
                            selectforeground=entry_fg)
            style.map('Light.TCombobox',
                      fieldbackground=[('readonly', entry_bg)],
                      background=[('active', entry_bg), ('!active', entry_bg)],
                      foreground=[('readonly', entry_fg)])

            # Checkbuttons/Radio: keep light background, no dark hover/selection blocks
            style.configure('TCheckbutton', background=bg_color, foreground=fg_color, focuscolor='none')
            style.map('TCheckbutton',
                      background=[('active', bg_color), ('selected', bg_color), ('disabled', bg_color)],
                      foreground=[('disabled', '#888888')])
            style.configure('TRadiobutton', background=bg_color, foreground=fg_color, focuscolor='none')
            style.map('TRadiobutton',
                      background=[('active', bg_color), ('selected', bg_color), ('disabled', bg_color)],
                      foreground=[('disabled', '#888888')])

            # Progressbar (green in light mode)
            style.configure('TProgressbar',
                            background='#4caf50',
                            troughcolor='#e6e6e6',
                            borderwidth=0,
                            lightcolor='#4caf50',
                            darkcolor='#4caf50')

            # Scrollbar
            style.configure("Light.Vertical.TScrollbar",
                            troughcolor='#e6e6e6',
                            background='#bdbdbd',
                            bordercolor=border_color,
                            lightcolor='#bdbdbd',
                            darkcolor='#bdbdbd',
                            arrowcolor=fg_color)
            style.map("Light.Vertical.TScrollbar",
                       background=[('active', '#a8a8a8'), ('pressed', '#9e9e9e')])

            if hasattr(self, 'result_vscroll'):
                self.result_vscroll.configure(style="Light.Vertical.TScrollbar")

            # Indicator-less radiobutton style (same layout/colors as dark but light palette)
            style.layout('IndicatorLess.TRadiobutton', [
                ('Radiobutton.padding', {
                    'children': [
                        ('Radiobutton.focus', {
                            'children': [ ('Radiobutton.label', {'sticky': 'nswe'}) ],
                            'sticky': 'nswe'
                        })
                    ],
                    'sticky': 'nswe'
                })
            ])
            style.configure('IndicatorLess.TRadiobutton',
                            background=bg_color,
                            foreground=fg_color,
                            focuscolor='none',
                            padding=(2, 0))
        
        # Update ScrolledText if it exists
        if hasattr(self, 'result_text'):
            self.update_scrolledtext_theme()
        
        # Apply widget styles after theme changes
        self.apply_widget_styles()
        
        # Update status label color to match current theme
        if hasattr(self, 'status_label'):
            current_text = self.status_label.cget('text')
            if 'Ready' in current_text:
                self.set_status(current_text, 'success')
            elif 'complete' in current_text.lower():
                self.set_status(current_text, 'success')
            elif 'error' in current_text.lower():
                self.set_status(current_text, 'error')
            elif 'unavailable' in current_text.lower() or 'warning' in current_text.lower():
                self.set_status(current_text, 'warning')
            else:
                self.set_status(current_text, 'info')
    
    def update_scrolledtext_theme(self):
        """Update Text widget and scrollbar theme"""
        if self.dark_mode.get():
            # Modern dark mode colors for Text widget
            self.result_text.configure(
                bg="#1e1e1e",              # Match main background
                fg="#cccccc",              # Light gray text
                insertbackground="#ffffff", # White cursor
                selectbackground="#264f78", # Blue selection
                selectforeground="#ffffff", # White selected text
                relief='flat'
            )
            # Apply dark scrollbar style
            if hasattr(self, 'result_vscroll'):
                self.result_vscroll.configure(style="Dark.Vertical.TScrollbar")
            
        else:
            # Clean light mode colors for Text widget
            self.result_text.configure(
                bg="white",
                fg="black",
                insertbackground="black",
                selectbackground="#0078d4",
                selectforeground="white",
                relief='flat'
            )
            # Reset to default scrollbar style
            if hasattr(self, 'result_vscroll'):
                self.result_vscroll.configure(style="Light.Vertical.TScrollbar")
    
    def _style_combobox_popup(self, cb):
        """Style the combobox dropdown list for dark mode"""
        # cb is a ttk.Combobox
        try:
            popdown = cb.tk.call("ttk::combobox::PopdownWindow", cb)  # path to the popup
            lb_path = f"{popdown}.f.l"  # the Listbox inside
            entry_bg = "#3c3c3c" if self.dark_mode.get() else "white"
            entry_fg = "#ffffff" if self.dark_mode.get() else "black"
            select_bg = "#264f78" if self.dark_mode.get() else "SystemHighlight"
            select_fg = "#ffffff" if self.dark_mode.get() else "SystemHighlightText"

            cb.tk.call(lb_path, "configure",
                       "-background", entry_bg,
                       "-foreground", entry_fg,
                       "-selectbackground", select_bg,
                       "-selectforeground", select_fg,
                       "-borderwidth", 0,
                       "-highlightthickness", 0)
            # also recolour the popdown frame
            cb.tk.call(f"{popdown}.f", "configure", "-borderwidth", 0, "-background", entry_bg)
        except tk.TclError:
            pass

    def get_theme_color(self, color_type):
        """Get theme-appropriate colors"""
        if self.dark_mode.get():
            # Dark mode colors
            colors = {
                'info': '#4fc1ff',      # Light blue for info messages
                'success': '#73c991',   # Light green for success
                'warning': '#ffcc02',   # Yellow for warnings
                'error': '#f85149'      # Light red for errors
            }
        else:
            # Light mode colors (original)
            colors = {
                'info': 'blue',
                'success': 'green', 
                'warning': 'orange',
                'error': 'red'
            }
        return colors.get(color_type, '#cccccc')
    
    def apply_widget_styles(self):
        """Apply theme-appropriate styles to widgets"""
        if self.dark_mode.get():
            # Apply dark styles to widgets
            if hasattr(self, 'file_entry'):
                self.file_entry.configure(style='Dark.Readonly.TEntry')
            if hasattr(self, 'source_language_combo'):
                self.source_language_combo.configure(style='Dark.TCombobox')
            if hasattr(self, 'target_language_combo'):
                self.target_language_combo.configure(style='Dark.TCombobox')
            
            # Style combobox popups for dark mode
            if hasattr(self, "source_language_combo"):
                self._style_combobox_popup(self.source_language_combo)
            if hasattr(self, "target_language_combo"):
                self._style_combobox_popup(self.target_language_combo)
        else:
            # Reset widgets to default styles
            if hasattr(self, 'file_entry'):
                self.file_entry.configure(style='Light.Readonly.TEntry')
            if hasattr(self, 'source_language_combo'):
                self.source_language_combo.configure(style='Light.TCombobox')
            if hasattr(self, 'target_language_combo'):
                self.target_language_combo.configure(style='Light.TCombobox')
            
            # Style combobox popups for light mode
            if hasattr(self, "source_language_combo"):
                self._style_combobox_popup(self.source_language_combo)
            if hasattr(self, "target_language_combo"):
                self._style_combobox_popup(self.target_language_combo)
    
    def set_status(self, text, status_type='info'):
        """Set status label with theme-appropriate color"""
        color = self.get_theme_color(status_type)
        self.status_label.config(text=text, foreground=color)
    
    def toggle_dark_mode(self):
        """Toggle between light and dark mode"""
        self.dark_mode.set(not self.dark_mode.get())
        self.apply_theme()
        # Update button text
        if hasattr(self, 'dark_mode_btn'):
            if self.dark_mode.get():
                self.dark_mode_btn.config(text="☀️ Light Mode")
            else:
                self.dark_mode_btn.config(text="🌙 Dark Mode")
        
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

        self.browse_btn = ttk.Button(
            file_frame,
            text="Browse",
            command=self.browse_file,
            style='App.TButton'
        )
        self.browse_btn.grid(row=0, column=1)
        
        ttk.Label(main_frame, text="Model Selection:", font=('Arial', 12, 'bold')).grid(
            row=2, column=0, sticky=tk.W, pady=(10, 5))
        
        self.model_var = tk.StringVar(value="large-v3")
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]
        self.model_radio_labels = {}
        for i, model in enumerate(models):
            # indicator-less radiobutton for crisp look; we prepend a bullet in the text
            rb = ttk.Radiobutton(
                model_frame,
                text=f"  {model}",
                variable=self.model_var,
                value=model,
                style='IndicatorLess.TRadiobutton'
            )
            rb.grid(row=0, column=i, padx=5)
            self.model_radio_labels[model] = rb
        
        # Language Settings
        ttk.Label(main_frame, text="Language Settings:", font=('Arial', 12, 'bold')).grid(
            row=4, column=0, sticky=tk.W, pady=(10, 5))
        
        language_frame = ttk.Frame(main_frame)
        language_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Common languages for the dropdown
        common_languages = [
            "auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", 
            "zh", "ar", "hi", "nl", "pl", "tr", "sv", "da", "no", "fi"
        ]
        
        ttk.Label(language_frame, text="Source:").grid(row=0, column=0, sticky=tk.W)
        self.source_language_var = tk.StringVar(value="auto")
        self.source_language_combo = ttk.Combobox(
            language_frame,
            textvariable=self.source_language_var,
            values=common_languages,
            width=15,
            state="readonly"  # avoid white edit field in dark mode
        )
        self.source_language_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 20))
        
        # Translation checkbox
        self.translate_var = tk.BooleanVar(value=False)
        self.translate_check = ttk.Checkbutton(language_frame, text="Translate to:", 
                                              variable=self.translate_var, command=self.toggle_translation)
        self.translate_check.grid(row=0, column=2, sticky=tk.W)
        
        # Target language combobox (initially disabled)
        self.target_language_var = tk.StringVar(value="en")
        self.target_language_combo = ttk.Combobox(
            language_frame,
            textvariable=self.target_language_var,
            values=common_languages[1:],
            width=15,
            state="disabled"  # Exclude "auto" for target
        )
        self.target_language_combo.grid(row=0, column=3, sticky=tk.W, padx=(5, 0))
        
        # Bind combobox popup styling and apply initial styling
        for combo in (self.source_language_combo, self.target_language_combo):
            combo.bind("<Button-1>", lambda e, c=combo: self._style_combobox_popup(c))
            self._style_combobox_popup(combo)  # initial pass

        # Keep radio labels updated with a crisp bullet
        def _update_model_bullets(*_):
            selected = self.model_var.get()
            for name, rb in self.model_radio_labels.items():
                # filled bullet if selected, hollow circle otherwise
                bullet = "●" if name == selected else "○"
                # keep spacing for alignment
                rb.configure(text=f" {bullet} {name}")
        self.model_var.trace_add('write', _update_model_bullets)
        _update_model_bullets()
        
        ttk.Label(main_frame, text="Options:", font=('Arial', 12, 'bold')).grid(
            row=6, column=0, sticky=tk.W, pady=(10, 5))
        
        options_frame = ttk.Frame(main_frame)
        options_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
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
        button_frame.grid(row=8, column=0, columnspan=2, pady=10)

        self.transcribe_btn = ttk.Button(
            button_frame,
            text="Transcribe",
            command=self.start_transcription,
            style='App.TButton'
        )
        self.transcribe_btn.grid(row=0, column=0, padx=5)

        self.save_btn = ttk.Button(
            button_frame,
            text="Save Transcript",
            command=self.save_transcript,
            state="disabled",
            style='App.TButton'
        )
        self.save_btn.grid(row=0, column=1, padx=5)

        self.format_btn = ttk.Button(
            button_frame,
            text="Format Segments",
            command=self.format_segments,
            state="disabled",
            style='App.TButton'
        )
        self.format_btn.grid(row=0, column=2, padx=5)

        self.export_subtitle_btn = ttk.Button(
            button_frame,
            text="Export Subtitles",
            command=self.export_subtitles,
            state="disabled",
            style='App.TButton'
        )
        self.export_subtitle_btn.grid(row=0, column=3, padx=5)
        
        # Set initial dark mode button text based on current state
        initial_text = "☀️ Light Mode" if self.dark_mode.get() else "🌙 Dark Mode"
        self.dark_mode_btn = ttk.Button(
            button_frame,
            text=initial_text,
            command=self.toggle_dark_mode,
            style='App.TButton'
        )
        self.dark_mode_btn.grid(row=0, column=4, padx=5)
        
        # Current task progress bar
        ttk.Label(main_frame, text="Current Task:", font=('Arial', 10)).grid(
            row=9, column=0, sticky=tk.W, pady=(10, 2))
        
        self.current_progress = ttk.Progressbar(main_frame, mode='determinate', maximum=100)
        self.current_progress.grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Overall progress bar  
        ttk.Label(main_frame, text="Overall Progress:", font=('Arial', 10)).grid(
            row=11, column=0, sticky=tk.W, pady=(5, 2))
        
        self.progress = ttk.Progressbar(main_frame, mode='determinate', maximum=100)
        self.progress.grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(main_frame, text="Ready", foreground=self.get_theme_color('success'))
        self.status_label.grid(row=13, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(main_frame, text="Transcript:", font=('Arial', 12, 'bold')).grid(
            row=14, column=0, sticky=tk.W, pady=(10, 5))
        
        # Create custom Text + ttk.Scrollbar for proper theming
        text_holder = ttk.Frame(main_frame)
        text_holder.grid(row=15, column=0, columnspan=2,
                        sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        text_holder.columnconfigure(0, weight=1)
        text_holder.rowconfigure(0, weight=1)

        self.result_text = tk.Text(
            text_holder,
            height=20,
            wrap=tk.WORD,
            borderwidth=0,
            highlightthickness=0
        )
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create a ttk scrollbar we can style
        self.result_vscroll = ttk.Scrollbar(
            text_holder,
            orient='vertical',
            command=self.result_text.yview
        )
        self.result_vscroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.result_text.configure(yscrollcommand=self.result_vscroll.set)
        
        # Apply initial theme to Text widget
        self.update_scrolledtext_theme()
        
        main_frame.rowconfigure(15, weight=1)
    
    def toggle_translation(self):
        """Enable/disable the target language combobox based on translation checkbox"""
        if self.translate_var.get():
            # use readonly to keep dark styling and prevent manual typing
            self.target_language_combo.config(state="readonly")
        else:
            self.target_language_combo.config(state="disabled")
        
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
        self.export_subtitle_btn.config(state="disabled")
        self.progress['value'] = 0
        self.current_progress['value'] = 0
        self.set_status("Transcribing...", 'info')
        self.result_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.transcribe_audio)
        thread.daemon = True
        thread.start()
    
    def transcribe_audio(self):
        try:
            file_path = self.file_var.get()
            
            self.root.after(0, lambda: self.set_status("Loading models...", 'info'))
            
            if self.model is None or self.current_model_name != self.model_var.get():
                self.model = whisper.load_model(self.model_var.get())
                self.current_model_name = self.model_var.get()
            
            if self.speaker_diarization_var.get():
                if not PYANNOTE_AVAILABLE:
                    self.root.after(0, lambda: self.set_status("Speaker diarization unavailable (pyannote.audio not installed)", 'warning'))
                    self.diarization_pipeline = False
                    self.diarization_result = None
                elif self.diarization_pipeline is None:
                    try:
                        self.root.after(0, lambda: self.set_status("Loading speaker diarization model...", 'info'))
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
                        self.root.after(0, lambda: self.set_status("Speaker diarization unavailable, continuing with transcription...", 'warning'))
                        self.diarization_pipeline = False
                        self.diarization_result = None
                
                if self.diarization_pipeline and self.diarization_pipeline is not False:
                    try:
                        self.root.after(0, lambda: self.set_status("Performing speaker diarization...", 'info'))
                        
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
                                        self.set_status(s, 'info')
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
                                self.root.after(0, lambda: self.set_status("Speaker diarization complete!", 'success'))
                                time.sleep(0.5)  # Brief pause to show completion
                            except Exception as e:
                                self._diarization_cancelled = True
                                raise e
                            
                            # Temporary file will be cleaned up automatically via self.temp_files tracking
                        else:
                            self.diarization_result = None
                            
                    except Exception as e:
                        self.diarization_result = None
            
            self.root.after(0, lambda: self.set_status("Processing audio...", 'info'))
            # Reset current progress for Whisper task
            self.update_current_progress(0)
            
            # Capture progress by intercepting stderr (where tqdm outputs)
            class ProgressCapture:
                def __init__(self, gui_ref):
                    self.gui_ref = gui_ref
                    self.original_stderr = sys.stderr
                    # no extra buffer needed; we stream directly to original stderr
                
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
                                        self.gui_ref.root.after(0, lambda: self.gui_ref.set_status(
                                            f"Processing audio... ({current}/{total} frames, {percentage}%)", 'info'))
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
                # Prepare transcription parameters
                transcribe_params = {
                    "word_timestamps": self.word_timestamps_var.get(),
                    "verbose": False
                }
                
                # Add language parameter if not auto-detect
                source_lang = self.source_language_var.get()
                if source_lang and source_lang != "auto":
                    transcribe_params["language"] = source_lang
                
                # Add translation task if enabled
                if self.translate_var.get():
                    transcribe_params["task"] = "translate"
                
                result = self.model.transcribe(file_path, **transcribe_params)
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
        self.set_status("Transcription complete!", 'success')
        self.transcribe_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.format_btn.config(state="normal")
        self.export_subtitle_btn.config(state="normal")
    
    def handle_error(self, error_message):
        self.progress['value'] = 0
        self.current_progress['value'] = 0
        self.set_status("Error occurred", 'error')
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
    
    def export_subtitles(self):
        """Export transcript as subtitle files (SRT/VTT)"""
        if not self.transcription_result or 'segments' not in self.transcription_result:
            messagebox.showerror("Error", "No transcript to export as subtitles.")
            return
        
        file_types = [
            ("SRT files", "*.srt"),
            ("WebVTT files", "*.vtt"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Export subtitles",
            defaultextension=".srt",
            filetypes=file_types
        )
        
        if filename:
            try:
                if filename.lower().endswith('.vtt'):
                    self.export_vtt(filename)
                else:
                    self.export_srt(filename)
                
                messagebox.showinfo("Success", f"Subtitles exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export subtitles: {str(e)}")
    
    def format_subtitle_timestamp_srt(self, seconds):
        """Format timestamp for SRT format: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
        seconds_int = int(seconds_remainder)
        
        return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"
    
    def format_subtitle_timestamp_vtt(self, seconds):
        """Format timestamp for VTT format: HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
        seconds_int = int(seconds_remainder)
        
        return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}.{milliseconds:03d}"
    
    def export_srt(self, filename):
        """Export transcript as SRT subtitle file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(self.transcription_result['segments'], 1):
                start_time = self.format_subtitle_timestamp_srt(segment['start'])
                end_time = self.format_subtitle_timestamp_srt(segment['end'])
                text = segment['text'].strip()
                
                # Add speaker info if available
                if self.speaker_diarization_var.get() and self.diarization_result:
                    speaker = self.get_speaker_at_time(segment['start'])
                    if speaker:
                        text = f"[{speaker}] {text}"
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
    
    def export_vtt(self, filename):
        """Export transcript as WebVTT subtitle file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for segment in self.transcription_result['segments']:
                start_time = self.format_subtitle_timestamp_vtt(segment['start'])
                end_time = self.format_subtitle_timestamp_vtt(segment['end'])
                text = segment['text'].strip()
                
                # Add speaker info if available
                if self.speaker_diarization_var.get() and self.diarization_result:
                    speaker = self.get_speaker_at_time(segment['start'])
                    if speaker:
                        text = f"[{speaker}] {text}"
                
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")

def run_cli(args):
    """Run transcription in CLI mode"""
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return 1
    
    print(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model)
    
    diarization_pipeline = None
    if args.speaker_diarization:
        if not PYANNOTE_AVAILABLE:
            print("Warning: Speaker diarization unavailable (pyannote.audio not installed)")
            diarization_pipeline = None
        else:
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
    
    # Prepare CLI transcription parameters
    transcribe_params = {
        "word_timestamps": args.word_timestamps,
        "verbose": True  # Show progress in CLI mode
    }
    
    # Add language parameter if not auto-detect
    if args.language and args.language != "auto":
        transcribe_params["language"] = args.language
    
    # Add translation task if enabled
    if args.translate:
        transcribe_params["task"] = "translate"
    
    result = model.transcribe(args.input, **transcribe_params)
    
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
    
    def format_subtitle_timestamp_srt_cli(seconds):
        """Format timestamp for SRT format: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
        seconds_int = int(seconds_remainder)
        
        return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"
    
    def format_subtitle_timestamp_vtt_cli(seconds):
        """Format timestamp for VTT format: HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
        seconds_int = int(seconds_remainder)
        
        return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}.{milliseconds:03d}"
    
    def export_srt_cli(filename, result, diarization_result, speaker_diarization):
        """Export transcript as SRT subtitle file in CLI mode"""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                start_time = format_subtitle_timestamp_srt_cli(segment['start'])
                end_time = format_subtitle_timestamp_srt_cli(segment['end'])
                text = segment['text'].strip()
                
                # Add speaker info if available
                if speaker_diarization and diarization_result:
                    speaker = get_speaker_at_time_cli(segment['start'])
                    if speaker:
                        text = f"[{speaker}] {text}"
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
    
    def export_vtt_cli(filename, result, diarization_result, speaker_diarization):
        """Export transcript as WebVTT subtitle file in CLI mode"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for segment in result['segments']:
                start_time = format_subtitle_timestamp_vtt_cli(segment['start'])
                end_time = format_subtitle_timestamp_vtt_cli(segment['end'])
                text = segment['text'].strip()
                
                # Add speaker info if available
                if speaker_diarization and diarization_result:
                    speaker = get_speaker_at_time_cli(segment['start'])
                    if speaker:
                        text = f"[{speaker}] {text}"
                
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
    
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
    
    # Handle subtitle exports
    if args.export_srt:
        try:
            export_srt_cli(args.export_srt, result, diarization_result, args.speaker_diarization)
            print(f"\nSRT subtitles exported to: {args.export_srt}")
        except Exception as e:
            print(f"Error exporting SRT file: {e}")
            return 1
    
    if args.export_vtt:
        try:
            export_vtt_cli(args.export_vtt, result, diarization_result, args.speaker_diarization)
            print(f"\nWebVTT subtitles exported to: {args.export_vtt}")
        except Exception as e:
            print(f"Error exporting VTT file: {e}")
            return 1
    
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"\nTranscript saved to: {args.output}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return 1
    else:
        # Only show transcript output if no subtitle exports were requested
        if not args.export_srt and not args.export_vtt:
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
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'turbo'],
                       help='Whisper model to use')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--no-timestamps', action='store_true', help='Disable timestamps')
    parser.add_argument('--no-word-timestamps', action='store_true', help='Disable word-level timestamps')
    parser.add_argument('--no-speaker-diarization', action='store_true', help='Disable speaker diarization')
    parser.add_argument('--clean-format', action='store_true', help='Use clean segment format only')
    parser.add_argument('--language', type=str, default='auto', help='Source language (auto for auto-detect)')
    parser.add_argument('--translate', action='store_true', help='Translate to English')
    parser.add_argument('--export-srt', type=str, help='Export as SRT subtitle file to specified path')
    parser.add_argument('--export-vtt', type=str, help='Export as WebVTT subtitle file to specified path')
    
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