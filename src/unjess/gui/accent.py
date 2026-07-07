"""Accent color utilities for dynamic theme customization.

Provides helper functions to convert a hex accent color into
CSS custom properties and rgba variants used throughout the GUI.
"""

from __future__ import annotations


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple.

    Args:
        hex_color: Hex color string like ``#7c3aed``.

    Returns:
        Tuple of (r, g, b) integers.
    """
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def accent_css(hex_color: str) -> str:
    """Generate CSS overrides for the accent color.

    Args:
        hex_color: Hex color string like ``#7c3aed``.

    Returns:
        CSS string to inject via ``ui.add_css()``.
    """
    r, g, b = hex_to_rgb(hex_color)

    # Derive dim variant (darken by ~30%)
    dim_r = max(0, int(r * 0.7))
    dim_g = max(0, int(g * 0.7))
    dim_b = max(0, int(b * 0.7))
    dim_hex = f"#{dim_r:02x}{dim_g:02x}{dim_b:02x}"

    # Derive light variant (lighten for text on dark)
    light_r = min(255, int(r + (255 - r) * 0.4))
    light_g = min(255, int(g + (255 - g) * 0.4))
    light_b = min(255, int(b + (255 - b) * 0.4))
    light_hex = f"#{light_r:02x}{light_g:02x}{light_b:02x}"

    return f"""
    :root {{
        --accent:       {hex_color} !important;
        --accent-dim:   {dim_hex} !important;
        --accent-glow:  rgba({r}, {g}, {b}, 0.1) !important;
        --accent-rgb:   {r}, {g}, {b};
        --accent-light: {light_hex};
    }}
    /* Override Quasar primary */
    .bg-primary {{ background-color: {hex_color} !important; }}
    .text-primary {{ color: {hex_color} !important; }}
    /* Override purple/violet Tailwind classes used in the app */
    .bg-purple-600, .bg-violet-600 {{ background-color: {hex_color} !important; }}
    .text-purple-400, .text-violet-400, .text-violet-500 {{ color: {light_hex} !important; }}
    .text-purple-600, .text-violet-600 {{ color: {hex_color} !important; }}
    """
