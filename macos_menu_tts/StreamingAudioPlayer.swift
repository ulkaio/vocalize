import AVFoundation
import Foundation

final class StreamingAudioPlayer: NSObject {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let sessionQueue = DispatchQueue(label: "tts.stream.queue")

    private var urlSession: URLSession?
    private var task: URLSessionDataTask?

    private var pendingData = Data()
    private var audioFormat: AVAudioFormat?
    private var bytesPerFrame: Int = 0
    private var isStopping = false
    private var completion: ((Result<Void, Error>) -> Void)?

    override init() {
        super.init()
        engine.attach(player)
    }

    func startStream(
        serverURL: String,
        text: String,
        voice: String,
        speed: Double,
        firstChunkChars: Int,
        streamChunkChars: Int,
        completion: @escaping (Result<Void, Error>) -> Void
    ) {
        stop()
        self.completion = completion
        isStopping = false
        pendingData.removeAll(keepingCapacity: true)
        audioFormat = nil
        bytesPerFrame = 0

        guard let url = URL(string: "\(serverURL)/v1/audio/speech") else {
            completion(.failure(StreamError.invalidURL))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "input": text,
            "voice": voice,
            "speed": speed,
            "stream": true,
            "first_chunk_chars": firstChunkChars,
            "stream_chunk_chars": streamChunkChars,
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body, options: [])

        let config = URLSessionConfiguration.default
        let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        urlSession = session

        let task = session.dataTask(with: request)
        self.task = task
        task.resume()
    }

    func pause() {
        player.pause()
    }

    func resume() {
        if !player.isPlaying {
            player.play()
        }
    }

    func stop() {
        isStopping = true
        task?.cancel()
        task = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        player.stop()
        engine.stop()
        pendingData.removeAll(keepingCapacity: true)
        audioFormat = nil
        bytesPerFrame = 0
    }
}

extension StreamingAudioPlayer: URLSessionDataDelegate {
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.isStopping { return }

            self.pendingData.append(data)

            if self.audioFormat == nil {
                guard self.pendingData.count >= 44 else { return }
                guard let format = Self.parseWavHeader(self.pendingData) else {
                    self.finish(with: .failure(StreamError.invalidWav))
                    return
                }
                self.audioFormat = format
                self.bytesPerFrame = Int(format.streamDescription.pointee.mBytesPerFrame)
                self.pendingData.removeFirst(44)

                self.engine.connect(self.player, to: self.engine.mainMixerNode, format: format)
                do {
                    try self.engine.start()
                    self.player.play()
                } catch {
                    self.finish(with: .failure(error))
                    return
                }
            }

            self.enqueueBuffersIfPossible()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.isStopping { return }
            if let error {
                self.finish(with: .failure(error))
                return
            }

            self.enqueueBuffersIfPossible(flushRemainder: true)
            self.finish(with: .success(()))
        }
    }

    private func enqueueBuffersIfPossible(flushRemainder: Bool = false) {
        guard let format = audioFormat, bytesPerFrame > 0 else { return }

        let framesPerBuffer: Int = 2048
        let bytesPerBuffer = framesPerBuffer * bytesPerFrame

        while pendingData.count >= bytesPerBuffer {
            let chunk = pendingData.prefix(bytesPerBuffer)
            pendingData.removeFirst(bytesPerBuffer)
            enqueueBuffer(chunk, format: format, frameCount: framesPerBuffer)
        }

        if flushRemainder, !pendingData.isEmpty {
            let remainingFrames = pendingData.count / bytesPerFrame
            if remainingFrames > 0 {
                let chunk = pendingData.prefix(remainingFrames * bytesPerFrame)
                pendingData.removeAll(keepingCapacity: true)
                enqueueBuffer(chunk, format: format, frameCount: remainingFrames)
            }
        }
    }

    private func enqueueBuffer(_ data: Data, format: AVAudioFormat, frameCount: Int) {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount)) else {
            return
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)

        data.withUnsafeBytes { rawBuffer in
            guard let src = rawBuffer.baseAddress else { return }
            if let channelData = buffer.int16ChannelData {
                let dst = channelData[0]
                dst.assign(from: src.assumingMemoryBound(to: Int16.self), count: frameCount)
            }
        }

        player.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
    }

    private func finish(with result: Result<Void, Error>) {
        DispatchQueue.main.async { [weak self] in
            self?.completion?(result)
            self?.completion = nil
        }
    }

    private static func parseWavHeader(_ data: Data) -> AVAudioFormat? {
        if data.count < 44 { return nil }

        let riff = String(data: data[0..<4], encoding: .ascii)
        let wave = String(data: data[8..<12], encoding: .ascii)
        let fmt = String(data: data[12..<16], encoding: .ascii)
        let dataTag = String(data: data[36..<40], encoding: .ascii)

        guard riff == "RIFF", wave == "WAVE", fmt == "fmt ", dataTag == "data" else {
            return nil
        }

        let audioFormat = data.toUInt16LE(at: 20)
        let numChannels = data.toUInt16LE(at: 22)
        let sampleRate = data.toUInt32LE(at: 24)
        let bitsPerSample = data.toUInt16LE(at: 34)

        guard audioFormat == 1, numChannels == 1, bitsPerSample == 16 else {
            return nil
        }

        return AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Double(sampleRate),
            channels: AVAudioChannelCount(numChannels),
            interleaved: false
        )
    }

    enum StreamError: LocalizedError {
        case invalidURL
        case invalidWav

        var errorDescription: String? {
            switch self {
            case .invalidURL:
                return "Invalid server URL"
            case .invalidWav:
                return "Invalid WAV stream"
            }
        }
    }
}

private extension Data {
    func toUInt16LE(at offset: Int) -> UInt16 {
        let value = self[offset..<offset + 2].withUnsafeBytes { ptr -> UInt16 in
            ptr.load(as: UInt16.self)
        }
        return UInt16(littleEndian: value)
    }

    func toUInt32LE(at offset: Int) -> UInt32 {
        let value = self[offset..<offset + 4].withUnsafeBytes { ptr -> UInt32 in
            ptr.load(as: UInt32.self)
        }
        return UInt32(littleEndian: value)
    }
}
