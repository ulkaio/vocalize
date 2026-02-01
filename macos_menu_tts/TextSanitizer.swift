import Foundation

enum TextSanitizer {
    static func sanitize(_ input: String) -> String {
        var text = input

        // Remove code fences
        text = text.replacingOccurrences(of: "```", with: "")

        // Remove inline code backticks
        text = text.replacingOccurrences(of: "`", with: "")

        // Strip common markdown markers
        let markdownPatterns = [
            "^#+\\s+",              // headings
            "^>\\s+",               // blockquote
            "^[-*+]\\s+",           // unordered list
            "^\\d+\\.\\s+",          // ordered list
        ]

        let lines = text
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { line -> String in
                var lineText = String(line)
                for pattern in markdownPatterns {
                    if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
                        let range = NSRange(lineText.startIndex..., in: lineText)
                        lineText = regex.stringByReplacingMatches(in: lineText, options: [], range: range, withTemplate: "")
                    }
                }
                return lineText
            }

        text = lines.joined(separator: "\n")

        // Replace markdown emphasis markers
        text = text.replacingOccurrences(of: "**", with: "")
        text = text.replacingOccurrences(of: "*", with: "")
        text = text.replacingOccurrences(of: "_", with: "")

        // Remove URLs
        if let regex = try? NSRegularExpression(pattern: "https?://\\S+", options: []) {
            let range = NSRange(text.startIndex..., in: text)
            text = regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "")
        }

        // Remove control characters
        text = text.unicodeScalars.filter { $0.value >= 32 || $0 == "\n" || $0 == "\t" }.map(String.init).joined()

        // Collapse whitespace
        if let regex = try? NSRegularExpression(pattern: "[\\t ]+", options: []) {
            let range = NSRange(text.startIndex..., in: text)
            text = regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: " ")
        }
        if let regex = try? NSRegularExpression(pattern: "\\n{3,}", options: []) {
            let range = NSRange(text.startIndex..., in: text)
            text = regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "\n\n")
        }

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
