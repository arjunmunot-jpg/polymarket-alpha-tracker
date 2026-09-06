"""CLI tool: sign/verify the triggers TOML.

Usage:
  python -m brain.sign_triggers --approve [--input PATH] [--output PATH]
  python -m brain.sign_triggers --verify [--input PATH]

--approve:
  1. Reads triggers.proposed.v5.toml (or --input).
  2. Resolves the committer identity (from --identity flag OR git config).
  3. Rejects if file already signed by a DIFFERENT identity (prevents silent
     overwrite); accepts if unsigned or signed by the same identity.
  4. Computes SHA-256 of file content with signature field VALUES blanked.
  5. Writes intent_hash + last_signed_at (UTC now) + last_signed_by into
     output (triggers.signed.toml).

--verify:
  Recomputes hash; prints OK or MISMATCH. Exits 0 on match, 1 otherwise.

Per brain_correction_layer_v2.md section 3 (F5 mitigation: trigger list
signed by USER, not author).

Per whitelisted_signing_identities (TOML): identity MUST be in emails list
if present. Whitelisting is advisory at the CLI layer — final enforcement
is in the brain loader (intent_hash match is the real gate).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from brain.toml_loader import (
    DEFAULT_TRIGGERS_PATH,
    SIGNATURE_FIELDS,
    TamperedTriggersError,
    compute_intent_hash,
)


# R3 (2026-08-04): anchored to the repo root — see brain/halt_state.py.
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "data/brain/triggers.proposed.v5.toml"
DEFAULT_OUTPUT = DEFAULT_TRIGGERS_PATH  # data/brain/triggers.signed.toml


# ---- Identity resolution ----


def _git_committer_email() -> str | None:
    """Return current git user.email via `git config` (committer proxy)."""
    try:
        out = subprocess.check_output(
            ["git", "config", "--get", "user.email"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def resolve_identity(cli_flag: str | None) -> str:
    """Pick signing identity. --identity wins; else git config; else error."""
    if cli_flag:
        return cli_flag.strip()
    git_id = _git_committer_email()
    if git_id:
        return git_id
    raise SystemExit(
        "ERROR: couldn't resolve signing identity. "
        "Pass --identity EMAIL or configure `git config --global user.email`."
    )


# ---- File-level ops ----


def read_file(p: Path) -> str:
    if not p.exists():
        raise SystemExit(f"ERROR: input file not found: {p}")
    return p.read_text(encoding="utf-8")


def parse_current_signature(content: str) -> tuple[str, str, str]:
    """Extract current values of last_signed_at, last_signed_by, intent_hash.

    Returns empty strings if a field is absent or value is empty.
    Only matches at top-level (before first [section] header).
    """
    head = content.partition("\n[")[0]
    values = {f: "" for f in SIGNATURE_FIELDS}
    for field in SIGNATURE_FIELDS:
        m = re.search(
            rf'^{re.escape(field)}\s*=\s*"([^"]*)"',
            head, re.MULTILINE,
        )
        if m:
            values[field] = m.group(1)
    return values["last_signed_at"], values["last_signed_by"], values["intent_hash"]


def set_signature_fields(content: str, at: str, by: str, intent: str) -> str:
    """Replace (or insert) the three signature fields in the top-level head.

    If a field is already present, rewrite its value.
    If missing, append to the top-level head before the first [section].
    """
    head, sep, tail = content.partition("\n[")
    updates = {
        "last_signed_at": at,
        "last_signed_by": by,
        "intent_hash": intent,
    }
    for field, val in updates.items():
        pattern = re.compile(
            rf'^{re.escape(field)}\s*=\s*"[^"]*"', re.MULTILINE
        )
        replacement = f'{field} = "{val}"'
        if pattern.search(head):
            head = pattern.sub(replacement, head, count=1)
        else:
            # Append at end of head section
            if not head.endswith("\n"):
                head += "\n"
            head += replacement + "\n"
    return head + sep + tail


# ---- Main operations ----


def approve(input_path: Path, output_path: Path,
            identity: str, whitelist: list[str] | None = None,
            force: bool = False) -> int:
    """Sign the file. Returns exit code."""
    content = read_file(input_path)
    _, current_by, current_hash = parse_current_signature(content)

    if current_hash and current_by and current_by != identity and not force:
        print(
            f"ERROR: file already signed by {current_by!r}. "
            f"Refuse to overwrite with {identity!r}. Pass --force to override.",
            file=sys.stderr,
        )
        return 2

    if whitelist and identity not in whitelist:
        print(
            f"WARNING: identity {identity!r} not in whitelist {whitelist}. "
            f"Proceeding (brain loader performs the real enforcement).",
            file=sys.stderr,
        )

    # Blank signature fields BEFORE hashing (canonical pre-image).
    # compute_intent_hash does this automatically, but we also want the
    # OUTPUT file to have the real values. Process:
    #   1. Normalize signature fields to empty in a STAGING copy.
    #   2. Compute hash of staging.
    #   3. Write output with real at/by/hash filled in.
    at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Compute hash using the loader's canonicalization (treats fields as empty)
    digest = compute_intent_hash(content)

    # Now write file with real values
    signed_content = set_signature_fields(content, at, identity, digest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(signed_content, encoding="utf-8")

    print(f"Signed: {output_path}")
    print(f"  last_signed_at = {at}")
    print(f"  last_signed_by = {identity}")
    print(f"  intent_hash    = {digest}")
    return 0


def verify(input_path: Path) -> int:
    """Check current signature. Returns 0 on match, 1 on mismatch, 3 unsigned."""
    content = read_file(input_path)
    _, by, stored = parse_current_signature(content)
    if not stored:
        print(f"UNSIGNED: {input_path} has empty intent_hash.", file=sys.stderr)
        return 3
    computed = compute_intent_hash(content)
    if computed == stored:
        print(f"OK: {input_path}")
        print(f"  signed_by   = {by}")
        print(f"  intent_hash = {stored}")
        return 0
    print(
        f"MISMATCH: {input_path}\n"
        f"  stored   = {stored}\n"
        f"  computed = {computed}",
        file=sys.stderr,
    )
    return 1


# ---- CLI ----


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m brain.sign_triggers",
        description="Sign or verify the Polybot brain triggers TOML.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--approve", action="store_true",
                      help="Compute hash and write signed output.")
    mode.add_argument("--verify", action="store_true",
                      help="Verify existing signature (no writes).")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help=f"Input path (default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output path for --approve (default: {DEFAULT_OUTPUT})")
    p.add_argument("--identity", type=str, default=None,
                   help="Signer email (default: git config user.email)")
    p.add_argument("--force", action="store_true",
                   help="Override refusal when file signed by different identity")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify:
        return verify(args.input)
    # approve
    identity = resolve_identity(args.identity)
    # Load whitelist from input file if it has one (best-effort)
    whitelist = _extract_whitelist(read_file(args.input))
    return approve(args.input, args.output, identity, whitelist, args.force)


def _extract_whitelist(content: str) -> list[str]:
    """Best-effort parse of [whitelisted_signing_identities].emails.

    We don't use the full TOML loader here to avoid circular validation at
    sign time (we're about to produce the valid-signed file).
    """
    try:
        try:
            import tomllib  # type: ignore
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore
        data = tomllib.loads(content)
        w = data.get("whitelisted_signing_identities", {})
        return list(w.get("emails", []))
    except Exception:
        return []


if __name__ == "__main__":
    sys.exit(main())
