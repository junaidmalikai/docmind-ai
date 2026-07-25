"""Minimal Streamlit-compatible CSS + footer.

Colors and widgets come from `.streamlit/config.toml` (official theming).
Only layout helpers that config.toml cannot express live here.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

_FOOTER_PX = 40
_GAP_PX = 10

_THEME_CSS = f"""
<style>
/* Fixed header — does not move when chat scrolls */
.dm-header {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: #000000;
    color: #ffffff;
    padding: 0.75rem 1.15rem;
    border-radius: 0.5rem;
    margin: 0 0 0.75rem 0;
}}
.dm-header h1 {{
    color: #ffffff !important;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0 !important;
    padding: 0 !important;
    line-height: 1.25;
}}
.dm-header p {{
    color: #ffffff !important;
    margin: 0 !important;
    font-size: 0.95rem;
    opacity: 0.9;
}}

/* Scrollable chat only — header + input stay put */
.dm-chat {{
    display: flex;
    flex-direction: column-reverse;
    gap: 0.85rem;
    padding: 0.9rem 1rem;
    height: calc(100vh - 330px);
    min-height: 200px;
    overflow-y: auto;
    overflow-x: hidden;
    scroll-behavior: smooth;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    background: #ffffff;
    scrollbar-gutter: stable;
    scrollbar-width: thin;
    scrollbar-color: #9ca3af #e5e7eb;
}}
.dm-chat::-webkit-scrollbar {{ width: 10px; }}
.dm-chat::-webkit-scrollbar-track {{ background: #e5e7eb; border-radius: 999px; }}
.dm-chat::-webkit-scrollbar-thumb {{ background: #9ca3af; border-radius: 999px; border: 2px solid #e5e7eb; }}
.dm-chat::-webkit-scrollbar-thumb:hover {{ background: #6b7280; }}
.dm-row {{ display: flex; align-items: flex-end; gap: 0.65rem; max-width: 100%; }}
.dm-row-ai {{ justify-content: flex-start; }}
.dm-row-user {{ justify-content: flex-end; }}
.dm-avatar {{
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    flex-shrink: 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}}
.dm-avatar-ai {{ background: #111827; color: #fff; }}
.dm-avatar-user {{ background: #2563eb; color: #fff; }}
.dm-bubble {{
    max-width: min(72%, 720px);
    padding: 0.75rem 1rem;
    border-radius: 14px;
    line-height: 1.5;
    font-size: 0.98rem;
    word-wrap: break-word;
    white-space: pre-wrap;
}}
.dm-bubble-ai {{
    background: #f3f4f6 !important;
    color: #111827 !important;
    border: 1px solid #e5e7eb !important;
    border-bottom-left-radius: 4px;
}}
.dm-bubble-user {{
    background: #2563eb;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}}
.dm-sources {{ margin-top: 0.45rem; font-size: 0.8rem; color: #6b7280; }}
.dm-empty {{ color: #9ca3af; font-size: 0.95rem; text-align: center; margin: auto 0; }}

[data-testid="stMain"] {{
    overflow: hidden !important;
}}
[data-testid="stMain"] .block-container {{
    padding-top: 1rem !important;
    padding-bottom: 7.5rem !important;
    max-width: 100% !important;
}}

/* Footer at absolute bottom */
.dm-footer {{
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 999;
    margin: 0;
    padding: 0 1rem;
    border-top: 1px solid rgba(49, 51, 63, 0.2);
    background: #f0f2f6;
    text-align: center;
    color: var(--text-color, #31333F);
    font-size: 0.85rem;
    line-height: 1.35;
    height: {_FOOTER_PX}px;
    box-sizing: border-box;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.dm-footer .dm-footer-name {{ font-weight: 700; }}
.dm-footer a {{ color: var(--primary-color, #FF4B4B); text-decoration: none; }}
.dm-footer a:hover {{ text-decoration: underline; }}
.dm-footer .dm-footer-sep {{ margin: 0 0.5rem; opacity: 0.55; }}

/* Compact delete / × buttons in sidebar rows */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button) > div:last-child {{
    flex: 0 0 2.5rem !important;
    width: 2.5rem !important;
    min-width: 2.5rem !important;
    max-width: 2.5rem !important;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button) > div:last-child button {{
    width: 2.35rem !important;
    min-height: 2.35rem !important;
    padding: 0 !important;
}}
</style>
"""

# Streamlit locks stBottom at bottom:0 via Emotion CSS; pure CSS often loses.
# This script writes inline styles on the parent document so the input bar's
# opaque background runs all the way down to the footer (no chat bleeding through).
_LIFT_JS = f"""
<script>
(function () {{
  const FOOTER = {_FOOTER_PX};
  const GAP = {_GAP_PX};

  function liftChatInput() {{
    const doc = window.parent.document;
    const bottom = doc.querySelector('[data-testid="stBottom"]');
    if (!bottom) return;

    // Bottom edge sits exactly on the footer, so the background touches it.
    bottom.style.setProperty("bottom", FOOTER + "px", "important");
    bottom.style.setProperty("left", "0", "important");
    bottom.style.setProperty("right", "0", "important");
    bottom.style.setProperty("z-index", "1000", "important");
    bottom.style.setProperty("background", "#ffffff", "important");
    bottom.style.setProperty("background-color", "#ffffff", "important");
    bottom.style.setProperty("border-top", "1px solid #e5e7eb", "important");
    bottom.style.setProperty("padding-top", "0.4rem", "important");
    bottom.style.setProperty("padding-bottom", GAP + "px", "important");
    bottom.style.setProperty("box-shadow", "none", "important");

    // Inner wrappers must be transparent-free so no gap shows the chat behind.
    bottom.querySelectorAll("div").forEach(function (el) {{
      if (el.getAttribute("data-testid") === "stChatInput") return;
      const testId = el.getAttribute("data-testid") || "";
      if (testId.includes("stBottomBlockContainer") || el === bottom.firstElementChild) {{
        el.style.setProperty("background", "#ffffff", "important");
        el.style.setProperty("padding-top", "0", "important");
        el.style.setProperty("padding-bottom", "0", "important");
      }}
    }});
  }}

  liftChatInput();
  const obs = new MutationObserver(liftChatInput);
  obs.observe(window.parent.document.body, {{ childList: true, subtree: true }});
  window.parent.addEventListener("resize", liftChatInput);
}})();
</script>
"""


def inject_theme() -> None:
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        """
        <div class="dm-footer">
          <span class="dm-footer-name">Muhammad Junaid</span>
          <span class="dm-footer-sep">|</span>
          <a href="tel:+923041659294">03041659294</a>
          <span class="dm-footer-sep">|</span>
          <a href="mailto:junaidfazal08@gmail.com">junaidfazal08@gmail.com</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Run AFTER chat_input exists so we can lift it above the footer
    components.html(_LIFT_JS, height=0, width=0)
