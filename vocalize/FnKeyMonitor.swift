import AppKit

/// Monitor for Fn key press and release events.
///
/// Uses `NSEvent.addGlobalMonitorForEvents` to detect when the Fn key
/// modifier flag changes state. Provides callbacks for press and release.
final class FnKeyMonitor {
    /// Called when Fn key is pressed down.
    var onFnPressed: (() -> Void)?

    /// Called when Fn key is released.
    var onFnReleased: (() -> Void)?

    private var globalMonitor: Any?
    private var localMonitor: Any?
    private var isFnDown = false

    /// Start monitoring for Fn key events.
    ///
    /// Registers both global and local event monitors to catch Fn key
    /// state changes regardless of which app has focus.
    func start() {
        stop()

        // Global monitor for events outside our app
        globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            self?.handleFlagsChanged(event)
        }

        // Local monitor for events when our app has focus
        localMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            self?.handleFlagsChanged(event)
            return event
        }

        FileLogger.shared.logSync("FnKeyMonitor started")
    }

    /// Stop monitoring for Fn key events.
    func stop() {
        if let globalMonitor {
            NSEvent.removeMonitor(globalMonitor)
        }
        if let localMonitor {
            NSEvent.removeMonitor(localMonitor)
        }
        globalMonitor = nil
        localMonitor = nil
        isFnDown = false
    }

    /// Handle modifier flags changed events.
    ///
    /// - Parameter event: The NSEvent containing modifier flag state.
    private func handleFlagsChanged(_ event: NSEvent) {
        let fnPressed = event.modifierFlags.contains(.function)

        if fnPressed && !isFnDown {
            // Fn key just pressed
            isFnDown = true
            FileLogger.shared.logSync("Fn key pressed")
            DispatchQueue.main.async { [weak self] in
                self?.onFnPressed?()
            }
        } else if !fnPressed && isFnDown {
            // Fn key just released
            isFnDown = false
            FileLogger.shared.logSync("Fn key released")
            DispatchQueue.main.async { [weak self] in
                self?.onFnReleased?()
            }
        }
    }

    deinit {
        stop()
    }
}
