import os
import time
import platform
import threading

class ShutdownTimer:
    def __init__(self, countdown_time = 60):
        self.countdown_time = countdown_time
        self.cancel_shutdown_event = threading.Event()

    def _get_shutdown_command(self):
        """Return a gentle shutdown command for the detected OS."""
        system_platform = platform.system().lower()

        if system_platform == 'windows':
            return 'shutdown /s /t 0 /f'
        elif system_platform == 'linux':
            return 'shutdown -h now'
        else:
            raise Exception("Unsupported OS")

    def shutdown(self):
        """Initiates the shutdown process with a countdown and the option to cancel."""
        shutdown_command = self._get_shutdown_command()

        print(f"Shutting down in {self.countdown_time} seconds...")

        while self.countdown_time > 0:
            print(f"{self.countdown_time} seconds remaining...", end="\r")
            time.sleep(1)
            self.countdown_time -= 1

            if self.cancel_shutdown_event.is_set():
                print("\nShutdown canceled!")
                return

        print("\nShutting down safely...")
        os.system(shutdown_command)

    def cancel_shutdown(self):
        """Listen for user pressing ENTER to cancel."""
        input("Press ENTER to cancel the shutdown.\n")
        self.cancel_shutdown_event.set()

    def start(self):
        """Start countdown and cancel listener."""
        cancel_thread = threading.Thread(target=self.cancel_shutdown, daemon=True)
        cancel_thread.start()

        self.shutdown()
