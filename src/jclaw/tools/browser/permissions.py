from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PermissionStatus:
    accessibility: bool
    screen_recording: bool
    automation: bool


def check_permissions() -> PermissionStatus:
    return PermissionStatus(
        accessibility=False,
        screen_recording=False,
        automation=False,
    )


def human_setup_instructions() -> list[str]:
    return [
        "Enable Accessibility for JClaw if desktop UI control is needed.",
        "Enable Screen Recording if JClaw needs to observe live app windows.",
        "Allow Automation if JClaw needs to control browser apps via Apple Events.",
    ]

