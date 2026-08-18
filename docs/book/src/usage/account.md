# Account Info

`meka account` exposes read-only account information obtained through a provider's OAuth API, so you
can script things that aren't otherwise reachable (a status bar, a cron alert). Every subcommand
takes an optional profile (defaults to the active provider, same as `--provider`) and a
`--format plain|json`. The requested data goes to **stdout**; notes and errors go to **stderr**, so
`meka account … 2>/dev/null | jq` stays clean.

Availability is per backend: `claude-subscription` and `chatgpt-subscription` (subscription OAuth) support these;
API-key backends, OpenAI-compatible endpoints, and Ollama print a short "not available" note and
exit non-zero.

## `meka account usage`

Current rate-limit windows (percentage used + reset time):

```console
$ meka account usage
Account usage
  5-hour (session)   [##--------]  23% used  (resets in 1h 58m, 2026-07-02 02:10)
  Weekly             [----------]   4% used  (resets in 12h 48m, 2026-07-02 13:00)

$ meka account usage --format json
{
  "provider": "claude-max",
  "windows": [
    { "label": "5-hour (session)", "used_percent": 23.0, "resets_at": 1782958200 },
    { "label": "Weekly", "used_percent": 4.0, "resets_at": 1782997200 }
  ],
  "extra_usage": { "enabled": false, "utilization": null, "used": 0.0,
                   "balance": null, "currency": "USD" },
  "note": null
}
```

`resets_at` is a Unix timestamp in seconds (`date -d @1782958200`). The `extra_usage` block reports
pay-as-you-go / overage state (whether it's enabled, percent of the extra-usage limit consumed,
amount spent, and remaining credit balance); the plain view shows a line only when it's enabled or
has a balance.

## `meka account whoami`

Account identity, plan, and **local** auth status. The auth block is computed from the stored
credential (no network), so even when the identity call fails because the token needs a re-login,
`whoami` still reports it and exits non-zero:

```console
$ meka account whoami
Account: claude-max (claude-subscription)
  Auth:          valid (5h 45m)
  Plan:          claude_max
  Tier:          default_claude_max_20x
  Subscription:  active
  Role:          admin

$ meka account whoami --format json
{
  "provider": "claude-max",
  "backend": "claude-subscription",
  "auth": { "valid": true, "expires_at": 1782971829, "expires_in_seconds": 20709 },
  "identity": { "plan": "claude_max", "tier": "default_claude_max_20x",
                "subscription_status": "active", "role": "admin", ... }
}
```

`identity` is `null` when the backend has no identity endpoint. `expires_at` / `expires_in_seconds`
are in seconds; a negative `expires_in_seconds` (or `valid: false`) means "run `meka provider login`".

## `meka account stats`

Historical usage. `chatgpt-subscription` is rich (lifetime tokens, peak day, streaks, and per-day token
counts); `claude-subscription` reports only a first-used date:

```console
$ meka account stats
Account history: claude-max
  First used:        2026-04-01

$ meka account stats --format json
{ "provider": "claude-max", "lifetime_tokens": null, "peak_daily_tokens": null,
  "current_streak_days": null, "longest_streak_days": null,
  "first_used": "2026-04-01T17:36:16.996974Z", "daily": [] }
```

For Codex, `daily` is a list of `{ "date": "YYYY-MM-DD", "tokens": N }` you can feed into a graph.

## Example: i3blocks

A block that shows the Claude 5-hour and weekly usage, refreshed every 5 minutes:

```sh
#!/bin/sh
# ~/.config/i3blocks/meka-usage   (set interval=300)
u=$(meka account usage claude-max --format json 2>/dev/null) || { echo "claude ?"; exit 0; }
echo "$u" | jq -r '
  (.windows[] | select(.label|startswith("5-hour")).used_percent) as $s |
  (.windows[] | select(.label=="Weekly").used_percent) as $w |
  "claude 5h:\($s|floor)% wk:\($w|floor)%"'
```

Each invocation makes one API call, so keep the poll interval sane (minutes, not seconds). The token
is refreshed automatically when near expiry and written back to the database, exactly as during a
normal session.
