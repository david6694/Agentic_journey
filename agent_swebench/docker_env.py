"""Docker container lifecycle and SIGKILL-proof cleanup for SWE-bench."""
import atexit
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
from types import FrameType
from typing import Any, Callable, Optional


class DockerLifecycleError(RuntimeError):
    """Raised when the SWE-bench Docker container cannot be prepared."""


class SWEBenchContainer:
    """Manage a SWE-bench container with guaranteed cleanup even on SIGKILL.

    Two complementary cleanup layers ensure zero orphan containers:
    1. The container runs `cat >/dev/null` reading from an open stdin pipe
       held by this Python process (`docker run -i --rm`). If this process
       receives SIGKILL (`kill -9`), the OS kernel immediately closes the
       pipe, `cat` exits on EOF, and Docker removes the `--rm` container.
    2. A detached background watchdog process polls `kill -0 <pid>` and runs
       `docker rm -f <container_name>` + removes the temporary host directory
       as soon as this process dies, backed by SIGINT/SIGTERM handlers and
       `atexit` for normal/signal exits.
    """

    def __init__(
        self,
        docker_image: str,
        instance_id: str,
        eval_script: str = "",
        pull_timeout: float = 600.0,
    ) -> None:
        self.docker_image = docker_image
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", instance_id)[:40]
        self.container_name = f"smith-swe-{safe_id}-{os.getpid()}"
        self.eval_script = eval_script
        self.pull_timeout = pull_timeout
        self.host_dir: Optional[Path] = None
        self.testbed_path: Optional[Path] = None
        self._docker_proc: Optional[subprocess.Popen[bytes]] = None
        self._cleaned = False
        self._prev_sigint: Any = None
        self._prev_sigterm: Any = None

    def __enter__(self) -> "SWEBenchContainer":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.cleanup()

    def _ensure_image(self) -> None:
        inspect = subprocess.run(
            ["docker", "image", "inspect", self.docker_image],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if inspect.returncode == 0:
            return
        # SWE-bench images only exist for amd64: arm64 hosts run them emulated.
        pull = subprocess.run(
            ["docker", "pull", "--platform", "linux/amd64", self.docker_image],
            capture_output=True,
            text=True,
            timeout=self.pull_timeout,
        )
        if pull.returncode != 0:
            raise DockerLifecycleError(
                f"Failed to pull image '{self.docker_image}': "
                f"{pull.stderr.strip()}"
            )

    def _spawn_watchdog(self) -> None:
        pid = os.getpid()
        cname = self.container_name
        hdir = str(self.host_dir) if self.host_dir else ""
        script = (
            f"while kill -0 {pid} 2>/dev/null; do sleep 0.2; done; "
            f"docker rm -f {cname} >/dev/null 2>&1 || true; "
            + (f"rm -rf '{hdir}' >/dev/null 2>&1 || true" if hdir else "true")
        )
        subprocess.Popen(
            ["sh", "-c", script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: Optional[FrameType]) -> None:
            self.cleanup()
            raise SystemExit(128 + signum)

        self._prev_sigint = signal.getsignal(signal.SIGINT)
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        atexit.register(self.cleanup)

    def _restore_signal_handlers(self) -> None:
        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)
            self._prev_sigint = None
        if self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)
            self._prev_sigterm = None

    def _wait_container_running(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._docker_proc and self._docker_proc.poll() is not None:
                stderr = (
                    self._docker_proc.stderr.read().decode(
                        "utf-8", errors="replace"
                    )
                    if self._docker_proc.stderr
                    else ""
                )
                raise DockerLifecycleError(
                    f"Container exited prematurely: {stderr.strip()}"
                )
            res = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Running}}",
                    self.container_name,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip() == "true":
                return
            time.sleep(0.1)
        raise DockerLifecycleError(
            f"Container '{self.container_name}' did not start within "
            f"{timeout}s."
        )

    def start(self) -> None:
        """Pull image, start container, and mirror /testbed to host_dir."""
        self.host_dir = Path(tempfile.mkdtemp(prefix="smith_swebench_"))
        self._install_signal_handlers()
        self._spawn_watchdog()
        try:
            self._ensure_image()
            self._docker_proc = subprocess.Popen(
                [
                    "docker",
                    "run",
                    "-i",
                    "--rm",
                    "--name",
                    self.container_name,
                    self.docker_image,
                    "sh",
                    "-c",
                    "cat >/dev/null",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._wait_container_running()

            cp_res = subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{self.container_name}:/testbed",
                    str(self.host_dir / "testbed"),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if cp_res.returncode != 0:
                raise DockerLifecycleError(
                    f"Failed to copy /testbed from container: "
                    f"{cp_res.stderr.strip()}"
                )
            self.testbed_path = self.host_dir / "testbed"
        except Exception:
            self.cleanup()
            raise

    def cleanup(self) -> None:
        """Stop and remove the Docker container and temporary host mirror."""
        if self._cleaned:
            return
        self._cleaned = True
        self._restore_signal_handlers()
        if self._docker_proc is not None:
            try:
                if self._docker_proc.stdin:
                    self._docker_proc.stdin.close()
            except Exception:
                pass
            try:
                self._docker_proc.terminate()
                self._docker_proc.wait(timeout=3)
            except Exception:
                try:
                    self._docker_proc.kill()
                except Exception:
                    pass
            self._docker_proc = None

        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if self.host_dir and self.host_dir.exists():
            shutil.rmtree(self.host_dir, ignore_errors=True)
            self.host_dir = None

    def configure_env(
        self, setter: Callable[[str, str], None] = os.environ.__setitem__
    ) -> None:
        """Set environment variables expected by mcp_tools_swebench.py."""
        if self.testbed_path is not None:
            setter("TESTBED_PATH", str(self.testbed_path))
        setter("SWE_CONTAINER_ID", self.container_name)
        if self.eval_script:
            setter("EVAL_SCRIPT", self.eval_script)
