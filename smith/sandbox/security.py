"""Security layers applied inside the sandbox worker process.

Only the standard library is used. The protection is layered:

1. Static AST check: rejects private/dunder attribute access before
   the code runs (blocks `().__class__.__subclasses__()` style escapes).
2. Restricted builtins: dangerous builtins are absent, `__import__`
   only accepts allowlisted modules, getattr-like builtins refuse
   private names.
3. Audit hook (PEP 578): the last line of defence. It sees every
   file, process and socket operation of the interpreter, even when
   reached through an indirect path, and cannot be removed once set.
4. Resource limit: RLIMIT_AS caps the address space (Linux).
"""

import ast
import builtins
import os
import sys
import sysconfig
from collections.abc import Callable, Iterable
from typing import Any


class SecurityError(Exception):
    """Raised when code is rejected before execution."""


# Attributes that give access to interpreter internals without starting
# with an underscore (frames, globals, modules).
FORBIDDEN_ATTRIBUTES = frozenset({
    "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code",
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code",
    "tb_frame", "tb_next", "co_code",
    "sys", "os", "modules", "builtins",
})

SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "classmethod", "complex", "dict", "dir", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "hash", "hex",
    "id", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "object", "oct", "ord", "pow", "print",
    "property", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "staticmethod", "str", "sum", "super", "tuple", "type",
    "zip", "open", "NotImplemented", "Ellipsis", "__build_class__",
)

# Audit events that are always refused once the sandbox is active.
BLOCKED_EVENT_PREFIXES = (
    "socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn",
    "os.spawn", "os.fork", "os.forkpty", "os.kill", "os.killpg",
    "os.putenv", "os.unsetenv", "os.startfile", "ctypes.",
    "resource.setrlimit", "resource.prlimit", "sys.addaudithook",
    "sys.settrace", "sys.setprofile", "urllib.", "http.client.",
    "ftplib.", "smtplib.", "poplib.", "imaplib.", "nntplib.",
    "telnetlib.", "webbrowser.", "mmap.", "pty.", "code.__new__",
)

# Modules that must never be loaded, whatever the import path.
DANGEROUS_MODULES = frozenset({
    "socket", "_socket", "ssl", "_ssl", "subprocess", "_posixsubprocess",
    "ctypes", "_ctypes", "multiprocessing", "pty", "mmap", "resource",
    "urllib.request", "http.client", "asyncio",
})

# Audit events carrying filesystem paths: event -> indexes of path args.
PATH_EVENTS: dict[str, tuple[int, ...]] = {
    "os.listdir": (0,), "os.scandir": (0,), "os.remove": (0,),
    "os.rmdir": (0,), "os.mkdir": (0,), "os.rename": (0, 1),
    "os.chmod": (0,), "os.chown": (0,), "os.truncate": (0,),
    "os.utime": (0,), "os.link": (0, 1), "os.symlink": (0, 1),
    "os.chdir": (0,), "os.chflags": (0,), "os.mkfifo": (0,),
    "os.mknod": (0,), "os.getxattr": (0,), "os.setxattr": (0,),
    "os.listxattr": (0,), "os.removexattr": (0,),
    "shutil.copyfile": (0, 1), "shutil.copymode": (0, 1),
    "shutil.copystat": (0, 1), "shutil.copytree": (0, 1),
    "shutil.rmtree": (0,), "shutil.move": (0, 1),
    "glob.glob": (0,), "glob.glob/2": (0,),
}


def _is_private(name: str) -> bool:
    """Tell whether a name targets interpreter internals."""
    return name.startswith("_") or name in FORBIDDEN_ATTRIBUTES


def validate_code(tree: ast.AST) -> None:
    """Reject code that tries to reach interpreter internals.

    Args:
        tree: Parsed source code.

    Raises:
        SecurityError: The code accesses a forbidden name.
    """
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Attribute) and _is_private(node.attr):
            raise SecurityError(
                f"line {line}: access to attribute '{node.attr}' is "
                "forbidden in the sandbox"
            )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SecurityError(
                f"line {line}: name '{node.id}' is forbidden in the sandbox"
            )
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name.startswith("_"):
                    raise SecurityError(
                        f"line {line}: importing private name "
                        f"'{alias.name}' is forbidden"
                    )
        if isinstance(node, ast.MatchClass):
            for attr in node.kwd_attrs:
                if _is_private(attr):
                    raise SecurityError(
                        f"line {line}: pattern on attribute '{attr}' is "
                        "forbidden in the sandbox"
                    )


def is_import_allowed(name: str, allowlist: Iterable[str]) -> bool:
    """Check a dotted module name against the import allowlist.

    An entry 'pkg' allows 'pkg' only; 'pkg.*' also allows its submodules.

    Args:
        name: Absolute module name, e.g. 'collections.abc'.
        allowlist: Entries of SandboxConfig.authorized_imports.

    Returns:
        True if the import is allowed.
    """
    for entry in allowlist:
        if entry.endswith(".*"):
            base = entry[:-2]
            if name == base or name.startswith(base + "."):
                return True
        elif name == entry:
            return True
    return False


def make_import_guard(allowlist: list[str]) -> Callable[..., Any]:
    """Build a replacement for `__import__` that enforces the allowlist."""
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0:
            raise ImportError("relative imports are not allowed")
        if not is_import_allowed(name, allowlist):
            allowed = ", ".join(sorted(allowlist))
            raise ImportError(
                f"import of '{name}' is not allowed in the sandbox. "
                f"Allowed modules: {allowed}"
            )
        return real_import(name, globals, locals, fromlist, level)

    return guarded_import


def _refuse_private_name(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap getattr-like builtins so a string cannot bypass the AST check."""
    def wrapper(obj: object, name: str, *args: Any) -> Any:
        if isinstance(name, str) and _is_private(name):
            raise AttributeError(
                f"access to attribute '{name}' is forbidden in the sandbox")
        return func(obj, name, *args)

    return wrapper


def build_safe_builtins(allowlist: list[str]) -> dict[str, Any]:
    """Create the restricted builtins dict for the sandbox namespace.

    eval, exec, compile, globals, locals, vars, input, breakpoint, exit
    and quit are deliberately absent.
    """
    safe: dict[str, Any] = {
        name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES
    }
    for name, value in vars(builtins).items():
        if isinstance(value, type) and issubclass(value, BaseException):
            safe[name] = value
    for func in (getattr, setattr, delattr, hasattr):
        safe[func.__name__] = _refuse_private_name(func)
    safe["__import__"] = make_import_guard(allowlist)
    return safe


def _is_under(path: str, roots: list[str]) -> bool:
    """Tell whether path is one of roots or inside one of them."""
    return any(
        path == root or path.startswith(root.rstrip(os.sep) + os.sep)
        for root in roots
    )


def _is_write(mode: object, flags: object) -> bool:
    """Tell whether an 'open' audit event may modify the file."""
    if isinstance(flags, int):
        write_flags = os.O_CREAT | os.O_TRUNC | os.O_APPEND
        return bool(
            (flags & os.O_ACCMODE) != os.O_RDONLY or flags & write_flags
        )
    if isinstance(mode, str):
        return any(char in mode for char in "wax+")
    return True


class AuditPolicy:
    """Audit hook enforcing filesystem, process and network rules."""

    def __init__(self, allowed_directories: list[str]) -> None:
        # realpath resolves symlinks and '..' so prefix checks hold.
        self._allowed = [os.path.realpath(d) for d in allowed_directories]
        paths = sysconfig.get_paths()
        # Interpreter libraries stay readable so allowed modules can
        # finish lazy imports; they are never writable.
        self._read_only = sorted({
            os.path.realpath(paths[key])
            for key in ("stdlib", "platstdlib", "purelib", "platlib")
            if key in paths
        })
        self._active = False

    def activate(self) -> None:
        """Start enforcing. Called once the worker setup is complete."""
        self._active = True

    def __call__(self, event: str, args: tuple[Any, ...]) -> None:
        """Audit hook entry point (see sys.addaudithook)."""
        if not self._active:
            return
        if event.startswith(BLOCKED_EVENT_PREFIXES):
            raise PermissionError(
                f"operation '{event}' is forbidden in the sandbox"
            )
        if event == "import" and args and args[0] in DANGEROUS_MODULES:
            raise ImportError(f"module '{args[0]}' is forbidden")
        if event == "open":
            self._check_open(args)
        elif event in PATH_EVENTS:
            for index in PATH_EVENTS[event]:
                if index < len(args):
                    self._check_path(args[index], event)

    def _check_open(self, args: tuple[Any, ...]) -> None:
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if isinstance(path, int):
            # Raw descriptors include the pipe to the parent process.
            raise PermissionError("opening file descriptors is forbidden")
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        real = os.path.realpath(os.fsdecode(path))
        if _is_under(real, self._allowed):
            return
        if not _is_write(mode, flags) and _is_under(real, self._read_only):
            return
        raise PermissionError(
            f"access to '{real}' is denied: allowed directories are "
            f"{', '.join(self._allowed) or 'none'}"
        )

    def _check_path(self, path: object, event: str) -> None:
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        real = os.path.realpath(os.fsdecode(path))
        if not _is_under(real, self._allowed):
            raise PermissionError(
                f"'{event}' on '{real}' is denied: outside allowed "
                "directories"
            )


def install_audit_policy(allowed_directories: list[str]) -> AuditPolicy:
    """Register the audit hook (irreversible for this process)."""
    policy = AuditPolicy(allowed_directories)
    sys.addaudithook(policy)
    return policy


def apply_memory_limit(max_memory_mb: int) -> None:
    """Cap the worker's address space with RLIMIT_AS (Linux only).

    The budget is added on top of the worker's current footprint so
    the interpreter itself does not eat the user code's allowance.
    macOS does not enforce RLIMIT_AS: the parent's RSS watchdog covers
    that case.
    """
    try:
        import resource
        with open("/proc/self/statm", encoding="ascii") as statm:
            pages = int(statm.read().split()[0])
        baseline = pages * os.sysconf("SC_PAGE_SIZE")
        limit = baseline + max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, OSError, ValueError, IndexError):
        pass
