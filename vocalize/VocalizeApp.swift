import SwiftUI

@main
struct VocalizeApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var appState = AppState()

    var body: some Scene {
        MenuBarExtra("Vocalize", systemImage: menuBarIcon) {
            VStack(alignment: .leading, spacing: 10) {
                // Status text with dictation indicator
                HStack {
                    if appState.isDictating {
                        Image(systemName: "mic.fill")
                            .foregroundColor(.red)
                    }
                    Text(appState.statusText)
                        .font(.caption)
                }

                Divider()

                // TTS Controls
                HStack {
                    Button(appState.isPaused ? "Play" : "Pause") {
                        if appState.isPaused {
                            appState.resume()
                        } else {
                            appState.pause()
                        }
                    }
                    .disabled(!appState.isPlaying)

                    Button("Stop") {
                        appState.stop()
                    }
                    .disabled(!appState.isPlaying)
                }

                // Dictation cancel button (only shown when dictating)
                if appState.isDictating {
                    Button("Cancel Dictation") {
                        appState.cancelDictation()
                    }
                }

                Divider()

                HStack {
                    Text("Speed")
                    Slider(value: $appState.speed, in: 0.5...2.0, step: 0.05)
                    Text(String(format: "%.2f", appState.speed))
                        .frame(width: 44, alignment: .trailing)
                }

                HStack {
                    Text("Voice")
                    TextField("Voice", text: $appState.voice)
                        .frame(width: 120)
                }

                HStack {
                    Text("Server")
                    TextField("URL", text: $appState.serverURL)
                        .frame(width: 220)
                }

                Divider()

                // Hotkey hints
                VStack(alignment: .leading, spacing: 4) {
                    Button("Speak Selection (⌥ Esc)") {
                        appState.speakSelection()
                    }
                    Text("Hold Fn to dictate")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                Button("Quit") {
                    NSApp.terminate(nil)
                }
            }
            .padding(12)
        }
    }

    /// Menu bar icon changes based on state.
    private var menuBarIcon: String {
        if appState.isDictating {
            return "waveform.badge.mic"
        } else if appState.isPlaying {
            return "waveform"
        } else {
            return "waveform"
        }
    }
}
