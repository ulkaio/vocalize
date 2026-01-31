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
        hotkeyManager.onHotkey = { [weak self] in
            self?.speakSelection()
        }
        hotkeyManager.registerHotkey()
    }

    func speakSelection() {
        stop()

        statusText = "Copying selection..."
        clipboardReader.readSelectedText { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let error):
                self.statusText = error.localizedDescription
                return
            case .success(let text):
                self.statusText = "Streaming"
                self.isPlaying = true
                self.isPaused = false

                self.audioPlayer.startStream(
                    serverURL: self.serverURL,
                    text: text,
                    voice: self.voice,
                    speed: self.speed,
                    firstChunkChars: 80,
                    streamChunkChars: 220
                ) { [weak self] result in
                    guard let self else { return }
                    switch result {
                    case .success:
                        self.statusText = "Finished"
                    case .failure(let error):
                        self.statusText = "Error: \(error.localizedDescription)"
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
    }

    func resume() {
        guard isPlaying else { return }
        audioPlayer.resume()
        isPaused = false
        statusText = "Streaming"
    }

    func stop() {
        audioPlayer.stop()
        if isPlaying {
            statusText = "Stopped"
        }
        isPlaying = false
        isPaused = false
    }
}
