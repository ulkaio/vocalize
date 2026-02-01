# MenuTTS (macOS Menu Bar Streaming TTS)

Menu bar app that streams audio from a running Kokoro TTS server. It reads selected text via a global hotkey, calls the backend streaming endpoint, and plays audio in real time.

## Features
- Menu bar app with play/pause/stop
- Global hotkey: Cmd+Shift+T
- Streams from `/v1/audio/speech` and plays immediately
- Voice + speed controls
- Input sanitization for markdown/code text
- Logs to `~/Library/Logs/MenuTTS.log`

## Requirements
- macOS 13+ (Apple Silicon recommended)
- Xcode
- `xcodegen` (project generator)
- Running backend server: `python serve.py --tts`

## Setup
```bash
brew install xcodegen
```

Generate the Xcode project:
```bash
cd macos_menu_tts
make project
```

## Build and Run
```bash
make run
```

## Install for Stable Permissions
Running from Xcode/DerivedData causes repeated Accessibility prompts. Install a stable app bundle:
```bash
make install
open /Applications/MenuTTS.app
```

## Permissions
The app simulates Cmd+C to read the current selection. Grant Accessibility:
- System Settings → Privacy & Security → Accessibility
- Add/enable `/Applications/MenuTTS.app`

If you run from DerivedData, permissions will be lost after rebuilds.

## Backend
Make sure the server is running:
```bash
python serve.py --tts
```

The app calls:
```
POST http://localhost:8000/v1/audio/speech
```
with JSON:
```json
{
  "input": "...",
  "voice": "af_heart",
  "speed": 1.25,
  "stream": true,
  "first_chunk_chars": 80,
  "stream_chunk_chars": 220
}
```

## Logging
Logs are written to:
```
~/Library/Logs/MenuTTS.log
```
To tail:
```bash
tail -f ~/Library/Logs/MenuTTS.log
```

## Troubleshooting
- **No text selected**: ensure Accessibility is enabled and the app is installed to a stable path.
- **No audio**: verify backend is running and reachable at `http://localhost:8000`.
- **App exits**: check `~/Library/Logs/MenuTTS.log` for crash info.
- **Repeated Accessibility prompts**: install to `/Applications` and re‑grant permission.

## Project Files
Key Swift files:
- `MenuTTSApp.swift`: menu bar UI
- `AppState.swift`: app logic
- `StreamingAudioPlayer.swift`: streaming WAV player
- `ClipboardReader.swift`: selection capture
- `TextSanitizer.swift`: markdown cleanup
- `FileLogger.swift`: log writer
- `AppDelegate.swift`: crash logging
