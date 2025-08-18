# Whisper Transcription Tool

A powerful GUI and CLI tool for audio/video transcription using OpenAI Whisper with advanced speaker diarization capabilities.

## Features

- **OpenAI Whisper Integration**: High-quality transcription using state-of-the-art models
- **Speaker Diarization**: Automatic speaker identification using pyannote.audio
- **Dual Interface**: Both GUI and command-line interfaces
- **Multiple Formats**: Supports MP4, AVI, MOV, MKV, MP3, WAV, M4A, FLAC
- **Real-time Progress**: Live progress tracking for both transcription and diarization
- **Translation Support**: Translate transcripts to 100+ languages with intelligent caching
- **Dark/Light Mode**: Toggle between modern dark and light themes
- **Smart Word Mapping**: Preserves timing accuracy when translating word-level timestamps
- **Flexible Output**: Multiple export formats with clean segment formatting
- **Robust Error Handling**: Comprehensive warning suppression and error recovery

## Installation

1. **Install FFmpeg** (required for video file processing):

   **Windows** (using winget):
   ```bash
   winget install FFmpeg
   ```
   
   **Linux** (using apt):
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```
   
   **macOS** (using Homebrew):
   ```bash
   brew install ffmpeg
   ```

2. Clone this repository:

   ```bash
   git clone <repository-url>
   cd whisper-transcription-tool
   ```

3. Create a virtual environment:

   ```bash
   python -m venv whisper_env
   # Windows
   whisper_env\Scripts\activate
   # Linux/Mac
   source whisper_env/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   **Note for CUDA users**: If you want GPU acceleration, you may need to install PyTorch with CUDA support first. Visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) to get the correct installation command for your CUDA version, then install the requirements.

   **Note for subtitle translation**: The `googletrans` library is included in requirements.txt for subtitle translation to languages other than English. If you don't need this feature, the tool will work without it.

5. Set up speaker diarization (optional):
   
   Follow the [pyannote speaker-diarization-3.1 setup instructions](https://huggingface.co/pyannote/speaker-diarization-3.1):
   - Accept user conditions for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Accept user conditions for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Create an access token at [hf.co/settings/tokens](https://hf.co/settings/tokens)
   - Add your token to a `.env` file: `TOKEN=your_huggingface_token`
   - Or login via: `huggingface-cli login`

## Usage

### GUI Mode (Default)

```bash
python whisper_gui.py
```

### CLI Mode

```bash
python whisper_gui.py --cli --input "audio.mp3" --model "large-v3" --output "transcript.txt"
```

**Export subtitles:**
```bash
# Export as SRT subtitles
python whisper_gui.py --cli --input "video.mp4" --export-srt "subtitles.srt"

# Export as WebVTT subtitles
python whisper_gui.py --cli --input "video.mp4" --export-vtt "subtitles.vtt"

# Export both transcript and subtitles
python whisper_gui.py --cli --input "video.mp4" --output "transcript.txt" --export-srt "subtitles.srt"
```

**Translation:**
```bash
# Translate any language to English (Whisper built-in)
python whisper_gui.py --cli --input "spanish_audio.mp3" --translate --output "english_transcript.txt"

# Export translated subtitles to other languages (requires googletrans)
python whisper_gui.py --cli --input "english_video.mp4" --export-srt-translated "spanish_subs.srt" --subtitle-language "es"
python whisper_gui.py --cli --input "english_video.mp4" --export-vtt-translated "french_subs.vtt" --subtitle-language "fr"

# Combine: Transcribe in original language + export translated subtitles
python whisper_gui.py --cli --input "video.mp4" --output "transcript.txt" --export-srt-translated "spanish_subs.srt" --subtitle-language "es"
```

#### CLI Options

- `--cli`: Enable command-line mode
- `--input`: Input audio/video file (required)
- `--model`: Whisper model (tiny, base, small, medium, large, large-v2, large-v3)
- `--output`: Output file path (optional)
- `--no-timestamps`: Disable timestamps
- `--no-word-timestamps`: Disable word-level timestamps
- `--no-speaker-diarization`: Disable speaker identification
- `--clean-format`: Use clean segment format only
- `--language`: Source language (auto for auto-detect)
- `--translate`: Translate to English (Whisper's built-in translation feature)
- `--target-language`: Target language for translation (currently only "en" supported by Whisper)
- `--export-srt`: Export as SRT subtitle file to specified path
- `--export-vtt`: Export as WebVTT subtitle file to specified path
- `--export-srt-translated`: Export as translated SRT subtitle file to specified path
- `--export-vtt-translated`: Export as translated WebVTT subtitle file to specified path
- `--subtitle-language`: Target language for subtitle translation (default: es for Spanish)

## GUI Features

- **File Browser**: Easy file selection with format filtering
- **Model Selection**: Choose from all available Whisper models
- **Options**:
  - Include timestamps
  - Word-level timestamps
  - Speaker diarization
  - Clean format (segments only)
  - Language selection and translation
- **Theme Toggle**: Switch between dark and light modes
- **Translation Features**:
  - Real-time translation with progress tracking
  - Intelligent caching to prevent double-translation
  - Smart word-level mapping for accurate timestamps
  - Background processing to prevent GUI freezing
- **Progress Tracking**: Dual progress bars showing current task and overall progress
- **Export Options**: Save full transcript, formatted segments, subtitles, and translated subtitles

## Speaker Diarization

The tool uses pyannote.audio for speaker diarization with:

- Conservative speaker assignment (0.8s tolerance)
- Robust fallback algorithms for timing misalignments
- Automatic speaker labeling (SPEAKER_00, SPEAKER_01, etc.)

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- FFmpeg (for video file processing)
- Hugging Face account (for speaker diarization)

## File Format Support

**Audio**: MP3, WAV, M4A, FLAC  
**Video**: MP4, AVI, MOV, MKV (audio extracted automatically)

## Output Formats

1. **Full Transcript**: Complete transcription with timestamps and speakers
2. **Formatted Segments**: Clean segment format with timestamps and speaker labels  
3. **Translated Output**: Transcripts translated to any of 100+ supported languages
4. **JSON Export**: Raw Whisper output with all metadata
5. **Subtitle Export**: SRT and WebVTT subtitle files (original and translated versions)

## Troubleshooting

- **Speaker diarization not working**: Ensure your Hugging Face token is set correctly
- **GPU memory issues**: Try smaller Whisper models (base, small, medium)
- **File format errors**: Ensure FFmpeg is installed for video file support

## License

MIT License
