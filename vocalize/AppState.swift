import AppKit
import Foundation

/// Main application state managing TTS playback and STT dictation.
final class AppState: ObservableObject {
    // MARK: - Published Properties

    @Published var isPlaying = false
    @Published var isPaused = false
    @Published var isDictating = false
    @Published var speed: Double = 1.5
    @Published var voice: String = "af_heart"
    @Published var serverURL: String = "http://localhost:8000"
    @Published var statusText: String = "Idle"

    // MARK: - TTS Components

    private let hotkeyManager = HotkeyManager()
    private let clipboardReader = ClipboardReader()
    private let audioPlayer = StreamingAudioPlayer()

    // MARK: - STT/Dictation Components

    private let fnKeyMonitor = FnKeyMonitor()
    private let audioRecorder = AudioRecorder()
    private let sttClient = SpeechToTextClient()
    private let textPaster = TextPaster()
    private var currentRecordingURL: URL?

    // MARK: - Initialization

    init() {
        FileLogger.shared.logSync("AppState initialized")

        // TTS hotkey (Option+Esc)
        hotkeyManager.onHotkey = { [weak self] in
            guard let self else { return }
            if self.isPlaying {
                self.stop()
            } else {
                self.speakSelection()
            }
        }
        hotkeyManager.registerHotkey()

        // STT dictation (Fn key hold)
        fnKeyMonitor.onFnPressed = { [weak self] in
            self?.startDictation()
        }
        fnKeyMonitor.onFnReleased = { [weak self] in
            self?.stopDictationAndPaste()
        }
        fnKeyMonitor.start()
    }

    // MARK: - Dictation Methods

    /// Start recording audio for dictation.
    func startDictation() {
        // Don't start dictation while TTS is playing
        guard !isPlaying else {
            FileLogger.shared.logSync("Dictation blocked: TTS is playing")
            return
        }

        // Check if already dictating
        guard !isDictating else { return }

        // Request microphone permission first
        audioRecorder.requestPermission { [weak self] granted in
            guard let self else { return }

            if !granted {
                self.statusText = "Microphone permission denied"
                FileLogger.shared.logSync("Microphone permission denied")
                return
            }

            do {
                self.currentRecordingURL = try self.audioRecorder.startRecording()
                self.isDictating = true
                self.statusText = "Dictating..."
                FileLogger.shared.logSync("Dictation started")
            } catch {
                self.statusText = "Recording error: \(error.localizedDescription)"
                FileLogger.shared.logSync("Recording error: \(error.localizedDescription)")
            }
        }
    }

    /// Stop recording and transcribe the audio.
    func stopDictationAndPaste() {
        guard isDictating else { return }

        isDictating = false
        statusText = "Transcribing..."

        guard let recordingURL = audioRecorder.stopRecording() else {
            statusText = "No recording"
            FileLogger.shared.logSync("No recording to transcribe")
            return
        }

        currentRecordingURL = recordingURL

        // Send to STT API
        sttClient.transcribe(audioURL: recordingURL, serverURL: serverURL) { [weak self] result in
            guard let self else { return }

            // Clean up recording file
            self.audioRecorder.deleteRecording(at: recordingURL)
            self.currentRecordingURL = nil

            switch result {
            case .success(let text):
                self.statusText = "Pasting..."
                self.textPaster.paste(text)

                // Reset status after paste
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    self.statusText = "Idle"
                }

            case .failure(let error):
                self.statusText = error.localizedDescription
                FileLogger.shared.logSync("STT error: \(error.localizedDescription)")

                // Reset status after showing error
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    if self.statusText == error.localizedDescription {
                        self.statusText = "Idle"
                    }
                }
            }
        }
    }

    /// Cancel ongoing dictation without transcribing.
    func cancelDictation() {
        guard isDictating else { return }

        isDictating = false
        if let url = audioRecorder.stopRecording() {
            audioRecorder.deleteRecording(at: url)
        }
        currentRecordingURL = nil
        statusText = "Idle"
        FileLogger.shared.logSync("Dictation cancelled")
    }

    func speakSelection() {
        stop()

        statusText = "Reading selection..."
        FileLogger.shared.logSync("Hotkey triggered: copying selection")

        clipboardReader.readSelectedText { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let error):
                self.statusText = error.localizedDescription
                FileLogger.shared.logSync("Clipboard error: \(error.localizedDescription)")
                return
            case .success(let text):
                let sanitized = TextSanitizer.sanitize(text)
                if sanitized.isEmpty {
                    self.statusText = "No speakable text"
                    FileLogger.shared.logSync("Sanitizer removed all content")
                    return
                }

                self.statusText = "Streaming"
                self.isPlaying = true
                self.isPaused = false

                FileLogger.shared.logSync("Starting stream: chars=\(sanitized.count)")

                self.audioPlayer.startStream(
                    serverURL: self.serverURL,
                    text: sanitized,
                    voice: self.voice,
                    speed: self.speed,
                    firstChunkChars: 80,
                    streamChunkChars: 220
                ) { [weak self] result in
                    guard let self else { return }
                    switch result {
                    case .success:
                        self.statusText = "Finished"
                        FileLogger.shared.logSync("Stream finished")
                    case .failure(let error):
                        self.statusText = "Error: \(error.localizedDescription)"
                        FileLogger.shared.logSync("Stream error: \(error.localizedDescription)")
                    }
                    self.isPlaying = false
                    self.isPaused = false
                }
            }
        }
    }

    func pause() {
        guard isPlaying else { return }
        audioPlayer.pause()
        isPaused = true
        statusText = "Paused"
        FileLogger.shared.logSync("Playback paused")
    }

    func resume() {
        guard isPlaying else { return }
        audioPlayer.resume()
        isPaused = false
        statusText = "Streaming"
        FileLogger.shared.logSync("Playback resumed")
    }

    func stop() {
        audioPlayer.stop()
        if isPlaying {
            statusText = "Stopped"
            FileLogger.shared.logSync("Playback stopped")
        }
        isPlaying = false
        isPaused = false
    }
}
