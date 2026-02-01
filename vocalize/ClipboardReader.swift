import AppKit
import Foundation
import ApplicationServices

final class ClipboardReader {
    private let pasteboard = NSPasteboard.general

    enum ClipboardError: LocalizedError {
        case accessibilityDenied
        case noText

        var errorDescription: String? {
            switch self {
            case .accessibilityDenied:
                return "Enable Accessibility permissions for this app"
            case .noText:
                return "No text selected"
            }
        }
    }

    func readSelectedText(completion: @escaping (Result<String, ClipboardError>) -> Void) {
        let options: NSDictionary = [kAXTrustedCheckOptionPrompt.takeRetainedValue() as String: true]
        if !AXIsProcessTrustedWithOptions(options) {
            completion(.failure(.accessibilityDenied))
            return
        }

        if let text = selectedTextViaAccessibility() {
            completion(.success(text))
            return
        }

        selectedTextViaClipboard(completion: completion)
    }

    private func selectedTextViaAccessibility() -> String? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let pid = app.processIdentifier
        let appElement = AXUIElementCreateApplication(pid)

        var focusedValue: AnyObject?
        let focusedResult = AXUIElementCopyAttributeValue(appElement, kAXFocusedUIElementAttribute as CFString, &focusedValue)
        guard focusedResult == .success else { return nil }
        let focusedElement = focusedValue as! AXUIElement

        var selectedValue: AnyObject?
        let selectedResult = AXUIElementCopyAttributeValue(focusedElement, kAXSelectedTextAttribute as CFString, &selectedValue)
        guard selectedResult == .success, let text = selectedValue as? String else { return nil }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func selectedTextViaClipboard(completion: @escaping (Result<String, ClipboardError>) -> Void) {
        var savedData: [[(NSPasteboard.PasteboardType, Data)]] = []
        if let items = pasteboard.pasteboardItems {
            for item in items {
                var pairs: [(NSPasteboard.PasteboardType, Data)] = []
                for type in item.types {
                    if let data = item.data(forType: type) {
                        pairs.append((type, data))
                    }
                }
                savedData.append(pairs)
            }
        }
        let previousChangeCount = pasteboard.changeCount

        let source = CGEventSource(stateID: .combinedSessionState)
        let keyDown = CGEvent(keyboardEventSource: source, virtualKey: 0x08, keyDown: true)
        keyDown?.flags = .maskCommand
        let keyUp = CGEvent(keyboardEventSource: source, virtualKey: 0x08, keyDown: false)
        keyUp?.flags = .maskCommand

        keyDown?.post(tap: .cghidEventTap)
        keyUp?.post(tap: .cghidEventTap)

        let maxTries = 10
        let delay: TimeInterval = 0.05
        var tries = 0

        func pollPasteboard() {
            tries += 1
            let changeCount = self.pasteboard.changeCount
            if changeCount != previousChangeCount || tries >= maxTries {
                let text = self.pasteboard.string(forType: .string) ?? ""

                self.pasteboard.clearContents()
                if !savedData.isEmpty {
                    let newItems = savedData.map { pairs -> NSPasteboardItem in
                        let item = NSPasteboardItem()
                        for (type, data) in pairs {
                            item.setData(data, forType: type)
                        }
                        return item
                    }
                    self.pasteboard.writeObjects(newItems)
                }

                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty {
                    completion(.failure(.noText))
                } else {
                    completion(.success(trimmed))
                }
                return
            }

            DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: pollPasteboard)
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: pollPasteboard)
    }
}
