"""Text preprocessing for speech: markdown stripping + TTS-summary extraction.

Claude (and most LLM/CLI output) is heavy with markdown — code fences, URLs,
heading hashes, bold/italic stars, table pipes. Read aloud verbatim it sounds
like line noise. Two transforms make raw LLM output safe to pipe straight into
``speak``:

``strip_markdown(text)``
    Best-effort flatten of markdown to readable plaintext.

``extract_tts_summary(text)``
    If the text embeds a concise spoken version inside a
    ``<!-- TTS_SUMMARY ... TTS_SUMMARY -->`` HTML comment, pull it out. This
    lets a generator write a long visible answer and a short *spoken* one in
    the same blob.

``preprocess_for_speech(text, strip=True)``
    The one entry point the daemon calls: prefer an embedded TTS summary;
    otherwise strip markdown (unless ``strip`` is False).
"""
from __future__ import annotations

import re

# <!-- TTS_SUMMARY ...inner... TTS_SUMMARY -->  (DOTALL so it can span lines)
_TTS_SUMMARY_RE = re.compile(r"<!-- TTS_SUMMARY(.*?)TTS_SUMMARY -->", re.DOTALL)

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)            # fenced code blocks
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")                  # inline `code`
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")             # ![alt](src)
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")             # [text](url) -> text
_BARE_URL_RE = re.compile(r"https?://\S+")                  # naked URLs
_HTML_TAG_RE = re.compile(r"<[^>]+>")                       # any <tag>
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"(\*\*|\*|__|_)")            # emphasis markers
_BLANK_LINES_RE = re.compile(r"\n\s*\n\s*\n+")             # 3+ newlines -> 2


def strip_markdown(text: str) -> str:
    """Flatten markdown to readable plaintext.

    Removes code fences, inline code backticks, image syntax, links (keeping
    their text), bare URLs, HTML tags, heading/blockquote/list markers, and
    bold/italic markers; collapses runs of blank lines. Order matters: code
    fences go before inline code, images before links.
    """
    text = _FENCE_RE.sub(" ", text)
    text = _IMAGE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _BARE_URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HEADING_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _LIST_MARKER_RE.sub("", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = text.replace("|", " ")  # table cell separators
    text = _BLANK_LINES_RE.sub("\n\n", text)
    # Tidy intra-line whitespace runs left by the substitutions.
    text = "\n".join(re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.splitlines())
    return text.strip()


def extract_tts_summary(text: str) -> str | None:
    """Return the inner text of a ``<!-- TTS_SUMMARY ... TTS_SUMMARY -->``
    comment if present, else None."""
    m = _TTS_SUMMARY_RE.search(text)
    if m is None:
        return None
    return m.group(1).strip()


def preprocess_for_speech(text: str, strip: bool = True) -> str:
    """Prepare ``text`` for synthesis.

    1. If an embedded TTS summary is present, speak only that.
    2. Else if ``strip`` is True, flatten markdown.
    3. Else return the text unchanged.
    """
    summary = extract_tts_summary(text)
    if summary is not None:
        return summary
    if strip:
        return strip_markdown(text)
    return text
