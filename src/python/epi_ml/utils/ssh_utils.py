"""Utility functions for SSH and SCP."""
from __future__ import annotations

from pathlib import Path
from typing import List

import paramiko
from scp import SCPClient


def createSSHClient(hostname: str, port: int, username: str):
    """Create SSH client from paramiko. Needs to be closed later."""
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh_key_filepath = f"{str(Path.home())}/.ssh/id_ed25519"
    client.connect(
        hostname=hostname, port=port, username=username, key_filename=ssh_key_filepath
    )
    return client


def createSCPClient(ssh_client: paramiko.SSHClient):
    """Create SCP client from paramiko. Needs to be closed later.

    Use this with scp.get and scp.put.
    """
    return SCPClient(ssh_client.get_transport())  # type: ignore


def run_commands_via_ssh(
    cmds: List[str], hostname: str, port: int, username: str, verbose: bool = True
) -> List:
    """
    Run a command on a remote server via SSH using SSH key authentication and return the decoded result.

    Args:
        hostname (str): The hostname of the remote server.
        port (int): The port number for the SSH service.
        username (str): The username for SSH login.
        verbose (bool): To print executed commands.
    Returns:
        list: The decoded output of the commands.
    """
    client = createSSHClient(hostname, port, username)
    results = []
    for cmd in cmds:
        if verbose:
            print(f"Running command: {cmd}")
        _, stdout, _ = client.exec_command(cmd)
        result = stdout.read().decode("utf-8")
        results.append(result)

    client.close()

    return results


# unknown_files = (
#     subprocess.check_output(
#         (
#             "find",
#             f"{gen_base_dir}",
#             "-mindepth",
#             "5",
#             "-maxdepth",
#             "6",
#             "-type",
#             "d",
#             "-name",
#             "predictions",
#         )
#     )
#     .decode("utf-8")
#     .splitlines()
# )
