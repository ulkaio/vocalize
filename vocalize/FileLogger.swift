import Foundation

final class FileLogger {
    static let shared = FileLogger()

    private let queue = DispatchQueue(label: "tts.file.logger")
    private let logURL: URL

    private init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let logsDir = home.appendingPathComponent("Library/Logs", isDirectory: true)
        logURL = logsDir.appendingPathComponent("Vocalize.log")
    }

    func log(_ message: String) {
        queue.async {
            self.writeLine(message)
        }
    }

    func logSync(_ message: String) {
        writeLine(message)
    }

    private func writeLine(_ message: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let line = "[\(timestamp)] \(message)\n"

        do {
            try FileManager.default.createDirectory(
                at: logURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if !FileManager.default.fileExists(atPath: logURL.path) {
                FileManager.default.createFile(atPath: logURL.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: logURL)
            defer { try? handle.close() }
            try handle.seekToEnd()
            if let data = line.data(using: .utf8) {
                try handle.write(contentsOf: data)
            }
        } catch {
            // Best-effort; avoid crashing on logging failures.
        }
    }
}
