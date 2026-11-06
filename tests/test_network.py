"""Ensure the default pytest configuration prevents network access."""

import socket

import pytest
from pytest_socket import SocketBlockedError


@pytest.mark.parametrize("family", [socket.AF_INET, socket.AF_INET6])
def test_network_is_blocked(family: socket.AddressFamily) -> None:
    with (
        pytest.warns(UserWarning, match="A test tried to use socket.socket"),
        pytest.raises(SocketBlockedError),
    ):
        socket.socket(family, socket.SOCK_STREAM)
