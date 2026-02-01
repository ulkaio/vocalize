import SwiftUI

@main
struct MenuTTSApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var appState = AppState()

    var body: some Scene {
        MenuBarExtra("TTS", systemImage: "speaker.wave.2") {
            VStack(alignment: .leading, spacing: 10) {
                Text(appState.statusText)
                    .font(.caption)

                Divider()

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

                Button("Speak Selection (⌥ Esc)") {
                    appState.speakSelection()
                }

                Button("Quit") {
                    NSApp.terminate(nil)
                }
            }
            .padding(12)
        }
    }
}
