#!/usr/bin/env python3
"""A minimal MCP server over stdio, for tests that need a real peer.

Speaks just enough of the protocol for meka to complete `initialize` and `tools/list`: one
newline-delimited JSON-RPC message per line, which is what the stdio transport uses.

The tool set and the instructions string are read from a *state file* named by argv[1], re-read on
every request, so a test can change what the server advertises between two connections to it. The
file is JSON:

    {"tools": ["search", "create_page"], "instructions": "call search first", "exit_after": 1}

`exit_after` is how many `tools/list` calls to serve before exiting, which is how a test closes the
transport under meka without needing the child's pid. Omit it to stay up.
"""

import json
import sys


def state(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def reply(message_id, result):
    sys.stdout.write(
        json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result}) + "\n"
    )
    sys.stdout.flush()


def main():
    path = sys.argv[1]
    served_lists = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = request.get("method")
        message_id = request.get("id")
        # Notifications carry no id and take no response.
        if message_id is None:
            continue
        current = state(path)
        if method == "initialize":
            result = {
                "protocolVersion": request.get("params", {}).get(
                    "protocolVersion", "2025-06-18"
                ),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "stub", "version": "0"},
            }
            if current.get("instructions") is not None:
                result["instructions"] = current["instructions"]
            reply(message_id, result)
        elif method == "tools/list":
            reply(
                message_id,
                {
                    "tools": [
                        {
                            "name": name,
                            "description": f"stub {name}",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                        for name in current.get("tools", [])
                    ]
                },
            )
            served_lists += 1
            limit = current.get("exit_after")
            if limit is not None and served_lists >= limit:
                return
        elif method == "ping":
            reply(message_id, {})
        else:
            sys.stdout.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "error": {"code": -32601, "message": f"no such method: {method}"},
                    }
                )
                + "\n"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
