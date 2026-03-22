import os
import signal
import subprocess
import threading

from utils.logger import logger


def terminate_process_tree(pid: int | None = None) -> None:
    """Forcefully terminate a process tree.

    On Windows we use ``taskkill /F /T`` so the current Python process and any
    child worker processes disappear together. On other platforms we fall back
    to sending SIGTERM to the current process.
    """

    target_pid = os.getpid() if pid is None else pid

    if os.name == "nt":
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.Popen(
                ["taskkill", "/F", "/T", "/PID", str(target_pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
        except Exception as exc:
            logger.error(
                "Failed to terminate process tree for pid %s: %s",
                target_pid,
                exc,
            )
        return

    try:
        os.kill(target_pid, signal.SIGTERM)
    except Exception as exc:
        logger.error("Failed to send SIGTERM to pid %s: %s", target_pid, exc)


def install_interrupt_guard(name: str = "process") -> threading.Event:
    """Register Ctrl+C and termination handlers that kill the current process tree."""

    stop_event = threading.Event()

    def _handler(signum, _frame) -> None:
        if stop_event.is_set():
            return

        stop_event.set()
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)

        logger.warning("%s received %s, shutting down now...", name, signame)
        terminate_process_tree()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except Exception:
            logger.debug("Unable to register handler for %s", sig_name)

    return stop_event
