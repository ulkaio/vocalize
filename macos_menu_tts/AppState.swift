import AppKit
import Foundation

final class AppState: ObservableObject {
    @Published var isPlaying = false
    @Published var isPaused = false
    @Published var speed: Double = 1.25
    @Published var voice: String = "af_heart"
    @Published var serverURL: String = "http://localhost:8000"
    @Published var statusText: String = "Idle"

    private let hotkeyManager = HotkeyManager()
    private let clipboardReader = ClipboardReader()
    private let audioPlayer = StreamingAudioPlayer()

    init() {
        FileLogger.shared.logSync("AppState initialized")
        hotkeyManager.onHotkey = { [weak self] in
            self?.speakSelection()
        }
        hotkeyManager.registerHotkey()
    }

    func speakSelection() {
        stop()

        statusText = "Copying selection..."
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
