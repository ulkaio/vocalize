import AppKit
import ApplicationServices
import Carbon

/// Pastes text at the current cursor position.
///
/// Writes text to the system clipboard and simulates Cmd+V
/// to paste it into the active application.
final class TextPaster {
    private let pasteboard = NSPasteboard.general

    /// Check if accessibility permissions are granted.
    ///
    /// - Parameter prompt: Whether to show the system prompt if not granted.
    /// - Returns: `true` if accessibility is enabled.
    func checkAccessibility(prompt: Bool = true) -> Bool {
        let options: NSDictionary = [kAXTrustedCheckOptionPrompt.takeRetainedValue() as String: prompt]
        return AXIsProcessTrustedWithOptions(options)
    }

    /// Paste text at the current cursor position.
    ///
    /// This saves the current clipboard contents, writes the new text,
    /// simulates Cmd+V, then restores the original clipboard after a delay.
    ///
    /// - Parameter text: The text to paste.
    func paste(_ text: String) {
        // Check accessibility permissions first
        if !checkAccessibility(prompt: true) {
            FileLogger.shared.logSync("Paste failed: Accessibility permission denied")
            return
        }

        // Save current clipboard contents
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

        // Write new text to clipboard
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)

        FileLogger.shared.logSync("Pasting text: \(text.prefix(50))...")

        // Delay to ensure clipboard is updated before simulating paste
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            self?.simulatePaste()

            // Restore original clipboard after paste completes (longer delay)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.restoreClipboard(savedData)
            }
        }
    }

    /// Simulate Cmd+V keypress to paste.
    private func simulatePaste() {
        let source = CGEventSource(stateID: .hidSystemState)

        // V key = 0x09 (kVK_ANSI_V)
        guard let keyDown = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: true),
              let keyUp = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: false) else {
            FileLogger.shared.logSync("Failed to create CGEvent for paste")
            return
        }

        keyDown.flags = .maskCommand
        keyUp.flags = .maskCommand

        keyDown.post(tap: .cgSessionEventTap)
        keyUp.post(tap: .cgSessionEventTap)

        FileLogger.shared.logSync("Simulated Cmd+V")
    }

    /// Restore previously saved clipboard contents.
    ///
    /// - Parameter savedData: The saved clipboard data to restore.
    private func restoreClipboard(_ savedData: [[(NSPasteboard.PasteboardType, Data)]]) {
        guard !savedData.isEmpty else { return }

        pasteboard.clearContents()
        let newItems = savedData.map { pairs -> NSPasteboardItem in
            let item = NSPasteboardItem()
            for (type, data) in pairs {
                item.setData(data, forType: type)
            }
            return item
        }
        pasteboard.writeObjects(newItems)

        FileLogger.shared.logSync("Restored clipboard")
    }
}
