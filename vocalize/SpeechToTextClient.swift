import Foundation

/// Client for sending audio to the Speech-to-Text API.
///
/// Sends audio files to the `/v1/audio/transcriptions` endpoint
/// and returns the transcribed text.
final class SpeechToTextClient {
    /// Errors that can occur during transcription.
    enum STTError: LocalizedError {
        case invalidURL
        case fileReadError(Error)
        case networkError(Error)
        case invalidResponse
        case serverError(Int, String)
        case emptyTranscription

        var errorDescription: String? {
            switch self {
            case .invalidURL:
                return "Invalid server URL"
            case .fileReadError(let error):
                return "Failed to read audio file: \(error.localizedDescription)"
            case .networkError(let error):
                return "Network error: \(error.localizedDescription)"
            case .invalidResponse:
                return "Invalid response from server"
            case .serverError(let code, let message):
                return "Server error (\(code)): \(message)"
            case .emptyTranscription:
                return "No speech detected"
            }
        }
    }

    /// Response from the transcription API.
    private struct TranscriptionResponse: Decodable {
        let text: String
    }

    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    /// Transcribe an audio file using the STT API.
    ///
    /// - Parameters:
    ///   - audioURL: URL of the audio file to transcribe.
    ///   - serverURL: Base URL of the server (e.g., "http://localhost:8000").
    ///   - model: Whisper model to use (default: "base").
    ///   - completion: Called with the transcribed text or an error.
    func transcribe(
        audioURL: URL,
        serverURL: String,
        model: String = "base",
        completion: @escaping (Result<String, STTError>) -> Void
    ) {
        // Build endpoint URL
        guard let url = URL(string: "\(serverURL)/v1/audio/transcriptions") else {
            completion(.failure(.invalidURL))
            return
        }

        // Read audio file data
        let audioData: Data
        do {
            audioData = try Data(contentsOf: audioURL)
        } catch {
            completion(.failure(.fileReadError(error)))
            return
        }

        // Create multipart form data request
        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        // Add model field
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"model\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(model)\r\n".data(using: .utf8)!)

        // Add file field
        let filename = audioURL.lastPathComponent
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(audioData)
        body.append("\r\n".data(using: .utf8)!)

        // Close boundary
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        FileLogger.shared.logSync("Sending audio to STT: \(url.absoluteString), size=\(audioData.count) bytes")

        // Send request
        let task = session.dataTask(with: request) { data, response, error in
            if let error {
                FileLogger.shared.logSync("STT network error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(.networkError(error)))
                }
                return
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                FileLogger.shared.logSync("STT invalid response")
                DispatchQueue.main.async {
                    completion(.failure(.invalidResponse))
                }
                return
            }

            guard let data else {
                FileLogger.shared.logSync("STT no data")
                DispatchQueue.main.async {
                    completion(.failure(.invalidResponse))
                }
                return
            }

            // Check status code
            guard httpResponse.statusCode == 200 else {
                let message = String(data: data, encoding: .utf8) ?? "Unknown error"
                FileLogger.shared.logSync("STT server error \(httpResponse.statusCode): \(message)")
                DispatchQueue.main.async {
                    completion(.failure(.serverError(httpResponse.statusCode, message)))
                }
                return
            }

            // Parse response
            do {
                let response = try JSONDecoder().decode(TranscriptionResponse.self, from: data)
                let text = response.text.trimmingCharacters(in: .whitespacesAndNewlines)

                if text.isEmpty {
                    FileLogger.shared.logSync("STT empty transcription")
                    DispatchQueue.main.async {
                        completion(.failure(.emptyTranscription))
                    }
                    return
                }

                FileLogger.shared.logSync("STT success: \(text.prefix(50))...")
                DispatchQueue.main.async {
                    completion(.success(text))
                }
            } catch {
                FileLogger.shared.logSync("STT parse error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(.invalidResponse))
                }
            }
        }

        task.resume()
    }
}
