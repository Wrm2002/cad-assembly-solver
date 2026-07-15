"""Minimal SSH/SFTP client for the temporary AutoDL instance.

The password is read only from AUTODL_PASSWORD.  It is never accepted as a
command-line argument and is never written to disk.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
import time
from pathlib import Path

import paramiko


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWN_HOSTS = (
    PROJECT_ROOT / "joinable_gpu_reproduction" / "autodl_known_hosts"
)


def fingerprint(key: paramiko.PKey) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get("AUTODL_PASSWORD")
    if not password:
        raise RuntimeError("AUTODL_PASSWORD is not set")

    known_hosts = args.known_hosts.resolve()
    last_error: Exception | None = None
    for attempt in range(1, args.connect_attempts + 1):
        client = paramiko.SSHClient()
        if known_hosts.is_file():
            client.load_host_keys(str(known_hosts))
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
        else:
            known_hosts.parent.mkdir(parents=True, exist_ok=True)
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=args.host,
                port=args.port,
                username=args.user,
                password=password,
                timeout=args.timeout,
                banner_timeout=args.timeout,
                auth_timeout=args.timeout,
                look_for_keys=False,
                allow_agent=False,
            )
            break
        except (
            EOFError,
            OSError,
            paramiko.SSHException,
        ) as exc:
            client.close()
            last_error = exc
            if attempt >= args.connect_attempts:
                raise
            delay = min(15, attempt * 3)
            print(
                f"connect attempt {attempt} failed: "
                f"{type(exc).__name__}:{exc}; retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    else:
        raise RuntimeError("SSH connection attempts exhausted") from last_error
    if not known_hosts.is_file():
        client.save_host_keys(str(known_hosts))
    key = client.get_transport().get_remote_server_key()
    print(
        f"connected host={args.host} port={args.port} "
        f"key_type={key.get_name()} fingerprint={fingerprint(key)}",
        flush=True,
    )
    return client


def execute(client: paramiko.SSHClient, command: str) -> int:
    _, stdout, stderr = client.exec_command(command, get_pty=False)
    while not stdout.channel.exit_status_ready():
        if stdout.channel.recv_ready():
            sys.stdout.buffer.write(stdout.channel.recv(65536))
            sys.stdout.buffer.flush()
        if stdout.channel.recv_stderr_ready():
            sys.stderr.buffer.write(
                stdout.channel.recv_stderr(65536)
            )
            sys.stderr.buffer.flush()
    while stdout.channel.recv_ready():
        sys.stdout.buffer.write(stdout.channel.recv(65536))
    while stdout.channel.recv_stderr_ready():
        sys.stderr.buffer.write(stdout.channel.recv_stderr(65536))
    sys.stdout.buffer.flush()
    sys.stderr.buffer.flush()
    return stdout.channel.recv_exit_status()


def upload(
    client: paramiko.SSHClient, local_path: Path, remote_path: str
) -> None:
    size = local_path.stat().st_size
    last_percent = -1

    def progress(transferred: int, total: int) -> None:
        nonlocal last_percent
        percent = int(100 * transferred / max(1, total))
        if percent >= last_percent + 5 or transferred == total:
            print(
                f"upload {percent:3d}% "
                f"({transferred / 2**20:.1f}/{size / 2**20:.1f} MiB)",
                flush=True,
            )
            last_percent = percent

    with client.open_sftp() as sftp:
        sftp.put(str(local_path), remote_path, callback=progress)


def download(
    client: paramiko.SSHClient, remote_path: str, local_path: Path
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = local_path.with_name(local_path.name + ".part")
    last_percent = -1

    def progress(transferred: int, total: int) -> None:
        nonlocal last_percent
        percent = int(100 * transferred / max(1, total))
        if percent >= last_percent + 5 or transferred == total:
            print(
                f"download {percent:3d}% "
                f"({transferred / 2**20:.1f}/"
                f"{total / 2**20:.1f} MiB)",
                flush=True,
            )
            last_percent = percent

    try:
        with client.open_sftp() as sftp:
            total = int(sftp.stat(remote_path).st_size)
            sftp.get(
                remote_path,
                str(temporary_path),
                callback=progress,
                prefetch=True,
                max_concurrent_prefetch_requests=4,
            )
        transferred = temporary_path.stat().st_size
        if transferred != total:
            raise OSError(
                f"incomplete download: {transferred}/{total} bytes"
            )
        temporary_path.replace(local_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--connect-attempts", type=int, default=4)
    parser.add_argument(
        "--known-hosts", type=Path, default=DEFAULT_KNOWN_HOSTS
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    exec_parser = subparsers.add_parser("exec")
    exec_parser.add_argument("--command", required=True)

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("--local", type=Path, required=True)
    upload_parser.add_argument("--remote", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--remote", required=True)
    download_parser.add_argument("--local", type=Path, required=True)

    args = parser.parse_args()
    client = connect(args)
    try:
        if args.action == "exec":
            return execute(client, args.command)
        if args.action == "upload":
            upload(client, args.local.resolve(), args.remote)
            return 0
        if args.action == "download":
            download(client, args.remote, args.local.resolve())
            return 0
        raise AssertionError(args.action)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
