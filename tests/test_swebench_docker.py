"""SWE-bench container and tools against a real Docker container."""
from collections.abc import Iterator
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from agent_swebench.docker_env import SWEBenchContainer
import mcp_tools_swebench as swe

IMAGE = "smith-test-testbed:local"
DOCKERFILE = """FROM python:3.10-slim
RUN mkdir /testbed \
 && printf 'x = 1\\ny = 1\\ndef f():\\n    return x\\n' > /testbed/app.py
"""


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    info = subprocess.run(
        ["docker", "info"], capture_output=True, timeout=10
    )
    return info.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="needs a running Docker daemon"
)


def _containers(name: str) -> str:
    return subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name={name}"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()


@pytest.fixture(scope="module")
def image() -> Iterator[str]:
    subprocess.run(
        ["docker", "build", "-q", "-t", IMAGE, "-"],
        input=DOCKERFILE, capture_output=True, text=True,
        timeout=300, check=True,
    )
    yield IMAGE
    subprocess.run(["docker", "rmi", "-f", IMAGE], capture_output=True)


@pytest.fixture
def container(
    image: str, monkeypatch: pytest.MonkeyPatch
) -> Iterator[SWEBenchContainer]:
    monkeypatch.delenv("EVAL_SCRIPT", raising=False)
    with SWEBenchContainer(image, "demo__repo-1") as box:
        box.configure_env(monkeypatch.setenv)
        yield box


def test_edit_file_reaches_the_container(
    container: SWEBenchContainer,
) -> None:
    assert "Successfully" in swe.edit_file("app.py", "x = 1", "x = 2")
    assert "x = 2" in swe.run_command("cat app.py")


def test_eval_script_stays_out_of_the_repository(
    image: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SWEBenchContainer(
        image, "demo__repo-4", eval_script="echo EVAL_OK"
    ) as box:
        box.configure_env(monkeypatch.setenv)
        assert swe.list_files() == "/testbed/app.py"
        assert "EVAL_OK" in swe.run_tests()


def test_read_file_sees_run_command_changes(
    container: SWEBenchContainer,
) -> None:
    swe.run_command("sed -i 's/x = 1/x = 2/' app.py")
    assert "1: x = 2" in swe.read_file("app.py")
    swe.run_command("printf 'z = 1\\n' > new.py")
    assert "1: z = 1" in swe.read_file("new.py")


def test_edit_file_keeps_run_command_changes(
    container: SWEBenchContainer,
) -> None:
    swe.run_command("printf 'z = 1\\n' >> app.py")
    assert "Successfully" in swe.edit_file("app.py", "x = 1", "x = 2")
    content = swe.run_command("cat app.py")
    assert "x = 2" in content and "z = 1" in content


def test_read_file_sees_run_tests_changes(
    container: SWEBenchContainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVAL_SCRIPT", "printf 'z = 1\\n' > from_eval.py")
    assert "exit_code: 0" in swe.run_tests()
    assert "1: z = 1" in swe.read_file("from_eval.py")


def test_test_reading_stdin_does_not_eat_the_eval_script(
    container: SWEBenchContainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVAL_SCRIPT", "cat >/dev/null\necho RESTORED\n")
    assert "RESTORED" in swe.run_tests()


def test_timed_out_command_stops_in_the_container(
    container: SWEBenchContainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = "import time; time.sleep(2); open('/tmp/alive', 'w')"
    monkeypatch.setattr(swe, "RUN_COMMAND_TIMEOUT", 1)
    # `; true` keeps bash as the parent: its child must be killed as well.
    result = swe.run_command(f'python -c "{script}"; true')
    assert "timed out after 1 seconds" in result
    time.sleep(2.5)
    monkeypatch.setattr(swe, "RUN_COMMAND_TIMEOUT", 60)
    assert "exit_code: 1" in swe.run_command("test -e /tmp/alive")


def test_timed_out_eval_stops_in_the_container(
    container: SWEBenchContainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVAL_SCRIPT", "sleep 2; touch /tmp/alive")
    monkeypatch.setattr(swe, "RUN_TESTS_TIMEOUT", 1)
    assert "timed out after 1 seconds" in swe.run_tests()
    time.sleep(2.5)
    assert "exit_code: 1" in swe.run_command("test -e /tmp/alive")


def test_symlink_from_container_cannot_read_host_files(
    container: SWEBenchContainer, tmp_path: Path
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP_SECRET\n", encoding="utf-8")
    swe.run_command(f"ln -s {secret} leak.txt")
    assert "TOP_SECRET" not in swe.search_code("TOP_SECRET")
    assert "TOP_SECRET" not in swe.read_file("leak.txt")
    assert container.testbed_path is not None
    assert not (container.testbed_path / "leak.txt").is_symlink()


def test_hostile_tar_cannot_escape_the_local_copy(
    container: SWEBenchContainer,
) -> None:
    """Code in the container can replace tar: the host must not trust it."""
    fake_tar = (
        "#!/usr/local/bin/python\n"
        "import io, sys, tarfile\n"
        "with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as out:\n"
        "    link = tarfile.TarInfo('link')\n"
        "    link.type, link.linkname = tarfile.SYMTYPE, '/etc/hosts'\n"
        "    out.addfile(link)\n"
        "    for name in ('ok.txt', '../outside.txt'):\n"
        "        info = tarfile.TarInfo(name)\n"
        "        info.size = 2\n"
        "        out.addfile(info, io.BytesIO(b'x\\n'))\n"
    )
    assert container.testbed_path is not None
    result = swe.run_command(
        f"cat > /usr/local/bin/tar <<'EOF'\n{fake_tar}EOF\n"
        "chmod +x /usr/local/bin/tar"
    )
    assert "Error" in result
    assert (container.testbed_path / "ok.txt").is_file()
    assert not (container.testbed_path / "link").is_symlink()
    assert not (container.testbed_path.parent / "outside.txt").exists()


def test_sync_size_limit_is_enforced_on_the_host(
    container: SWEBenchContainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(swe, "MAX_SYNC_BYTES", 10)
    assert "too large" in swe.run_command("head -c 100 /dev/zero > big.bin")
    assert "big.bin" not in swe.list_files()


def test_only_reasonable_regular_files_reach_the_host(
    container: SWEBenchContainer,
) -> None:
    swe.run_command("mkfifo pipe; truncate -s 6M big.bin; echo ok > small.txt")
    assert swe.list_files().split() == [
        "/testbed/app.py", "/testbed/small.txt"
    ]


def test_tools_show_paths_usable_in_the_container(
    container: SWEBenchContainer,
) -> None:
    path = swe.list_files()
    assert path == "/testbed/app.py"
    assert "x = 1" in swe.run_command(f"cat {path}")
    assert swe.search_code("y = 1") == "/testbed/app.py:2 y = 1"
    assert swe.search_function_or_class_definition_in_code("f") == (
        "/testbed/app.py:3 def f():"
    )
    assert swe.find_references("y") == "/testbed/app.py:2 y = 1"


def test_cleanup_removes_container_and_local_copy(image: str) -> None:
    box = SWEBenchContainer(image, "demo__repo-2")
    box.start()
    local_copy = box.host_dir
    assert local_copy is not None and local_copy.is_dir()
    assert _containers(box.container_name)
    box.cleanup()
    assert not _containers(box.container_name)
    assert not local_copy.exists()


def test_kill_9_leaves_nothing_behind(image: str) -> None:
    script = (
        "import time\n"
        "from agent_swebench.docker_env import SWEBenchContainer\n"
        f"box = SWEBenchContainer({image!r}, 'demo__repo-3')\n"
        "box.start()\n"
        "print(box.container_name, box.host_dir, flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout is not None
    name, local_copy = proc.stdout.readline().split()
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=5)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and (
        _containers(name) or Path(local_copy).exists()
    ):
        time.sleep(0.2)
    assert not _containers(name)
    assert not Path(local_copy).exists()
