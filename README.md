# Speech-to-Text Transcription Tool

A powerful and flexible audio transcription toolkit that processes audio files, transcribes them using OpenAI's Whisper model, and performs intelligent text formatting to produce high-quality transcriptions.

## 📋 Overview

This repository contains tools for automated transcription of audio recordings, with specific optimizations for meeting audio. It solves several common challenges:

- Processing large audio files that exceed API size limits
- Handling long recordings efficiently through parallel processing
- Correcting technical terms and framework names in transcriptions
- Maintaining natural language flow while improving text readability

## 🔑 Key Features

- Audio compression and format conversion
- Automatic bitrate optimization to meet size constraints
- Concurrent audio transcription for faster processing
- Intelligent correction of technical terms and framework names
- Comprehensive error handling and logging

## 🗂️ Repository Structure

- **transcription.py**: Core transcription tool for general audio files
- **meet_transcription.py**: Specialized tool for meeting recordings with advanced features
- **config.ini**: Configuration file for API keys and settings (not tracked in git)

## 🧠 How It Works

### Audio Preprocessing

1. **Compression & Conversion**: Audio files are compressed and converted to compatible formats (MP3/MP4)
2. **Size Optimization**: Files are automatically optimized to meet API size constraints by adjusting:
   - Bitrate
   - Sample rate
   - Audio channels

### Transcription Process

1. **Audio Chunking**: Larger files are split into manageable chunks
2. **Parallel Processing**: Chunks are processed concurrently for improved speed
3. **Text Assembly**: Transcribed chunks are assembled in the correct order
4. **Text Correction**: Framework and library names are automatically corrected
5. **Formatting**: Grammar and coherence are improved for better readability

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- OpenAI API key
- Required Python packages (see Installation)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/speech-to-text.git
   cd speech-to-text
   ```

2. Install dependencies:
   ```bash
   pip install openai pydub
   ```

3. Create a `config.ini` file with your OpenAI API key:
   ```ini
   [OPENAI]
   api_key = your_openai_api_key_here
   ```

### Usage

#### Basic Audio Transcription

For simple audio file transcription:

```bash
python transcription.py
```

This will:
1. Load the audio file specified in the script (default: `audio.m4a`)
2. Compress and convert it if needed
3. Transcribe the audio using OpenAI's Whisper model
4. Format the transcription for readability
5. Save the result to `transcricao_formatada.txt`

#### Meeting Transcription with Technical Term Correction

For meeting recordings with technical discussions:

```bash
python meet_transcription.py
```

This advanced script will:
1. Process the audio file with optimized settings for speech
2. Split the audio into multiple chunks for parallel processing
3. Transcribe all chunks concurrently
4. Correct technical terms, framework names, and libraries
5. Save both the raw and corrected transcriptions

## ⚙️ Customization

You can customize the behavior by modifying:

- Audio file paths in the main functions
- Maximum file size limits (default: 20-25MB)
- Number of processing chunks and workers for parallel processing
- Language settings for transcription (default: Portuguese)
- Formatting prompts for text correction

## 📝 Logging

Both scripts include comprehensive logging that records:
- Process steps and completion status
- File sizes and compression details
- Errors and exceptions
- Processing times and results

Logs are saved to `transcription.log` and also output to the console.

## 🔧 Technical Implementation

### Audio Processing

The project uses `pydub` for audio manipulation, which supports:
- Reading various audio formats
- Bitrate adjustment
- Sample rate conversion
- Audio chunking

### Transcription Engine

Transcription is performed using OpenAI's Whisper model through the OpenAI API, with optimized parameters for accuracy.

### Concurrent Processing

The `concurrent.futures` module enables efficient parallel processing of audio chunks, significantly reducing processing time for large files.

## 📄 License

[Add your license information here]

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 🙏 Acknowledgments

- OpenAI for providing the Whisper transcription model
- Contributors and developers of the pydub library
# speech-to-text