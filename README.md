# Whisper Transcription Tool

A powerful GUI and CLI tool for audio/video transcription using OpenAI Whisper with advanced speaker diarization capabilities.

## Features

- **OpenAI Whisper Integration**: High-quality transcription using state-of-the-art models
- **Speaker Diarization**: Automatic speaker identification using pyannote.audio
- **Dual Interface**: Both GUI and command-line interfaces
- **Multiple Formats**: Supports MP4, AVI, MOV, MKV, MP3, WAV, M4A, FLAC
- **Real-time Progress**: Live progress tracking for both transcription and diarization
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

#### CLI Options

- `--cli`: Enable command-line mode
- `--input`: Input audio/video file (required)
- `--model`: Whisper model (tiny, base, small, medium, large, large-v2, large-v3)
- `--output`: Output file path (optional)
- `--no-timestamps`: Disable timestamps
- `--no-word-timestamps`: Disable word-level timestamps
- `--no-speaker-diarization`: Disable speaker identification
- `--clean-format`: Use clean segment format only

## GUI Features

- **File Browser**: Easy file selection with format filtering
- **Model Selection**: Choose from all available Whisper models
- **Options**:
  - Include timestamps
  - Word-level timestamps
  - Speaker diarization
  - Clean format (segments only)
- **Progress Tracking**: Dual progress bars showing current task and overall progress
- **Export Options**: Save full transcript or formatted segments

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
3. **JSON Export**: Raw Whisper output with all metadata

## Troubleshooting

- **Speaker diarization not working**: Ensure your Hugging Face token is set correctly
- **GPU memory issues**: Try smaller Whisper models (base, small, medium)
- **File format errors**: Ensure FFmpeg is installed for video file support

## License

MIT License
