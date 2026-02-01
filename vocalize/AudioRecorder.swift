import AVFoundation
import Foundation

/// Records audio from the microphone to a WAV file.
///
/// Uses `AVAudioEngine` to capture microphone input and writes it to
/// a file in a format suitable for Whisper STT (16kHz mono preferred).
final class AudioRecorder {
    /// Errors that can occur during recording.
    enum RecorderError: LocalizedError {
        case microphonePermissionDenied
        case engineStartFailed(Error)
        case fileCreationFailed(Error)
        case notRecording

        var errorDescription: String? {
            switch self {
            case .microphonePermissionDenied:
                return "Microphone permission denied"
            case .engineStartFailed(let error):
                return "Failed to start audio engine: \(error.localizedDescription)"
            case .fileCreationFailed(let error):
                return "Failed to create audio file: \(error.localizedDescription)"
            case .notRecording:
                return "Not currently recording"
            }
        }
    }

    private let engine = AVAudioEngine()
    private var audioFile: AVAudioFile?
    private var recordingURL: URL?
    private(set) var isRecording = false

    /// Request microphone permission.
    ///
    /// - Parameter completion: Called with `true` if permission granted.
    func requestPermission(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        case .denied, .restricted:
            completion(false)
        @unknown default:
            completion(false)
        }
    }

    /// Start recording audio to a temporary WAV file.
    ///
    /// - Throws: `RecorderError` if recording cannot start.
    /// - Returns: The URL where the recording will be saved.
    @discardableResult
    func startRecording() throws -> URL {
        if isRecording {
            stopRecording()
        }

        // Create temp file URL
        let tempDir = FileManager.default.temporaryDirectory
        let fileName = "dictation_\(UUID().uuidString).wav"
        let fileURL = tempDir.appendingPathComponent(fileName)
        recordingURL = fileURL

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Create output format: 16kHz mono for Whisper compatibility
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        ) else {
            throw RecorderError.fileCreationFailed(NSError(domain: "AudioRecorder", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create output format"]))
        }

        // Create audio file
        do {
            audioFile = try AVAudioFile(
                forWriting: fileURL,
                settings: outputFormat.settings,
                commonFormat: .pcmFormatFloat32,
                interleaved: false
            )
        } catch {
            throw RecorderError.fileCreationFailed(error)
        }

        // Create converter for resampling if needed
        let converter = AVAudioConverter(from: inputFormat, to: outputFormat)

        // Install tap on input node
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            guard let self, let audioFile = self.audioFile, let converter else { return }

            // Convert to output format
            let frameCount = AVAudioFrameCount(
                Double(buffer.frameLength) * outputFormat.sampleRate / inputFormat.sampleRate
            )
            guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: frameCount) else {
                return
            }

            var error: NSError?
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }

            converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)

            if error == nil && convertedBuffer.frameLength > 0 {
                do {
                    try audioFile.write(from: convertedBuffer)
                } catch {
                    FileLogger.shared.logSync("Error writing audio: \(error.localizedDescription)")
                }
            }
        }

        // Start engine
        do {
            try engine.start()
            isRecording = true
            FileLogger.shared.logSync("Recording started: \(fileURL.path)")
            return fileURL
        } catch {
            audioFile = nil
            recordingURL = nil
            throw RecorderError.engineStartFailed(error)
        }
    }

    /// Stop recording and return the recorded file URL.
    ///
    /// - Returns: The URL of the recorded WAV file, or nil if not recording.
    @discardableResult
    func stopRecording() -> URL? {
        guard isRecording else { return nil }

        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        audioFile = nil
        isRecording = false

        let url = recordingURL
        FileLogger.shared.logSync("Recording stopped: \(url?.path ?? "nil")")
        return url
    }

    /// Delete a recorded file.
    ///
    /// - Parameter url: The URL of the file to delete.
    func deleteRecording(at url: URL) {
        do {
            try FileManager.default.removeItem(at: url)
            FileLogger.shared.logSync("Deleted recording: \(url.path)")
        } catch {
            FileLogger.shared.logSync("Failed to delete recording: \(error.localizedDescription)")
        }
    }

    deinit {
        if isRecording {
            stopRecording()
        }
    }
}
