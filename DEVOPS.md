# DevOps Notes

## Local Service

- Foreground run: `jclaw run`
- Install launch agent: `jclaw install-launchd`
- Remove launch agent: `jclaw uninstall-launchd`
- Tail logs: `tail -f ~/Library/Logs/JClaw/stdout.log`

## Safe Upgrade Flow

1. Pull changes.
2. Reinstall editable package if dependencies changed.
3. Run `jclaw doctor`.
4. Restart the launch agent with `jclaw install-launchd`.

## Packaging

RPM-related files are placeholders for parity with the original OpenClaw-inspired structure. The primary supported target in this bootstrap is macOS.
