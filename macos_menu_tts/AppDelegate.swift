import AppKit
import Foundation

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        FileLogger.shared.logSync("MenuTTS launched")
        installCrashHandlers()
    }

    func applicationWillTerminate(_ notification: Notification) {
        FileLogger.shared.logSync("MenuTTS terminating")
    }

    private func installCrashHandlers() {
        NSSetUncaughtExceptionHandler { exception in
            FileLogger.shared.logSync("Uncaught exception: \(exception.name.rawValue) \(exception.reason ?? "")")
        }

        let signals: [Int32] = [SIGABRT, SIGILL, SIGSEGV, SIGFPE, SIGBUS]
        for sig in signals {
            signal(sig) { signal in
                FileLogger.shared.logSync("Received signal: \(signal)")
                _exit(signal)
            }
        }
    }
}
