# Vocalize

macOS menu bar app for voice interactions: **Text-to-Speech** (TTS) and **Speech-to-Text** (STT) dictation.

## Features

### Text-to-Speech
- **Global hotkey**: Option+Esc to speak selected text
- Streams audio from `/v1/audio/speech` endpoint
- Play/pause/stop controls
- Adjustable voice and speed

### Speech-to-Text (Dictation)
- **Hold Fn key** to record your voice
- **Release Fn** to transcribe and paste at cursor
- Uses `/v1/audio/transcriptions` endpoint (Whisper)

### General
- Menu bar app with waveform icon
- Logs to `~/Library/Logs/Vocalize.log`

## Requirements
- macOS 13+ (Apple Silicon recommended)
- Xcode
- `xcodegen` (project generator)
- Running backend server with TTS and/or STT enabled

## Setup
```bash
brew install xcodegen
```

Generate the Xcode project:
```bash
cd vocalize
make project
```

## Build and Run
```bash
make run
```

## Install for Stable Permissions
Running from DerivedData causes repeated permission prompts. Install a stable app bundle:
```bash
make install
open /Applications/Vocalize.app
```

## Permissions

The app requires two permissions:

1. **Accessibility** (for simulating Cmd+C and Cmd+V)
   - System Settings → Privacy & Security → Accessibility
   - Enable **Vocalize**

2. **Microphone** (for dictation)
   - System Settings → Privacy & Security → Microphone
   - Enable **Vocalize**

## Backend

Start the server with TTS and STT:
```bash
python serve.py --tts --stt
```

### TTS Endpoint
```
POST http://localhost:8000/v1/audio/speech
```
```json
{
  "input": "Hello world",
  "voice": "af_heart",
  "speed": 1.25,
  "stream": true
}
```

### STT Endpoint
```
POST http://localhost:8000/v1/audio/transcriptions
Content-Type: multipart/form-data
file: <audio.wav>
```

## Hotkeys

| Action | Hotkey |
|--------|--------|
| Speak selected text | Option + Esc |
| Dictate and paste | Hold Fn, release to paste |

## Logging
```bash
tail -f ~/Library/Logs/Vocalize.log
```

## Troubleshooting

- **No text selected**: Enable Accessibility permission
- **Microphone permission denied**: Enable Microphone permission
- **Paste not working**: Enable Accessibility permission
- **No audio**: Verify backend is running at `http://localhost:8000`
- **Repeated permission prompts**: Install to `/Applications`

## Project Files

| File | Purpose |
|------|---------|
| `VocalizeApp.swift` | Menu bar UI |
| `AppState.swift` | App logic and state |
| `StreamingAudioPlayer.swift` | Streaming WAV player |
| `AudioRecorder.swift` | Microphone recording |
| `SpeechToTextClient.swift` | STT API client |
| `TextPaster.swift` | Paste text at cursor |
| `ClipboardReader.swift` | Selection capture |
| `FnKeyMonitor.swift` | Fn key detection |
| `HotkeyManager.swift` | Global hotkey (Option+Esc) |
| `TextSanitizer.swift` | Markdown cleanup |
| `FileLogger.swift` | Log writer |
