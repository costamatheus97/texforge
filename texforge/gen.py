"""Image generation behind one small interface, with a pluggable backend.

Every generation request in texforge (text-to-image and image-to-image edits) goes through `_one()` below, which
dispatches on the backend chosen by `TEXFORGE_GENERATOR` (or the global `--generator` option):

  codex    (default) the `codex` CLI, exactly as before (see below)
  none     never generate; raises a clear error that asks for an existing image (`--image`, `make <image>`, ...)
  command  run a user-supplied command template (`TEXFORGE_GENERATOR_COMMAND`, see `_command_one`)

Backend `codex`: image generation through the `codex` CLI (non-interactive, built-in image generation tool).

How it works (verified against codex-cli 0.166):
  codex exec --skip-git-repo-check --ephemeral -s workspace-write --json -C <dir> "<instruction>"
The model calls its built-in image-generation tool, which writes a PNG to
~/.codex/generated_images/<thread-id>/*.png (1254x1254 for square prompts; 1536x1024 for wide ones).
pick the PNG from the thread's subfolder, and copy it into our output folder.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GEN_DIR = Path.home() / ".codex" / "generated_images"

INSTRUCTION = (
    "Use your built-in image generation tool to generate exactly ONE image from the prompt between the "
    "markers. Pass the prompt verbatim and do not rewrite it. Do not edit, crop or post-process the result "
    "and do not run any other command. Reply with the single word done.\n"
    "<<<PROMPT\n{prompt}\nPROMPT>>>"
)


def codex_available() -> bool:
    return shutil.which("codex") is not None


EDIT_INSTRUCTION = (
    "The attached image is the reference. Use your built-in image generation tool to generate exactly ONE new "
    "image that is an EDIT of the attached image, following the prompt between the markers. Keep exactly the "
    "same layout, framing, scale and alignment as the attached image. Pass the prompt verbatim. Do not post-process "
    "the result and do not run any other command. Reply with the single word done.\n"
    "<<<PROMPT\n{prompt}\nPROMPT>>>"
)


BACKENDS = ("codex", "none", "command")
_override: str | None = None   # set by the --generator option; wins over the environment


def set_backend(name: str | None) -> None:
    """Select the backend for this process (None = fall back to TEXFORGE_GENERATOR, then `codex`)."""
    global _override
    if name is not None and name not in BACKENDS:
        raise ValueError(f"unknown generator {name!r}; choose one of {', '.join(BACKENDS)}")
    _override = name


def backend() -> str:
    name = (_override or os.environ.get("TEXFORGE_GENERATOR") or "codex").strip().lower()
    if name not in BACKENDS:
        raise RuntimeError(f"unknown generator {name!r} (TEXFORGE_GENERATOR); choose one of {', '.join(BACKENDS)}")
    return name


def _none_one(prompt: str, dest: Path, workdir: Path, timeout: int, image: Path | None = None) -> Path:
    raise RuntimeError("image generation is disabled (generator 'none'): pass an existing image instead "
                       "(`texforge make <image>`, `--image`, `--height-image`, `--normal-image`), "
                       "or choose another generator with --generator / TEXFORGE_GENERATOR")


def _command_one(prompt: str, dest: Path, workdir: Path, timeout: int, image: Path | None = None) -> Path:
    """Run the user's command template (TEXFORGE_GENERATOR_COMMAND). It is split like a shell line (shlex) and run
    WITHOUT a shell; these placeholders are replaced inside every argument:
      {prompt}       the prompt text            {prompt_file}  a UTF-8 text file holding the prompt
      {input}        the input image to edit (image-to-image only; an argument that is exactly {input} is dropped
                     when there is no input image)
      {output}       the PNG path the command must write
    The same values are also in the environment as TEXFORGE_PROMPT, TEXFORGE_PROMPT_FILE, TEXFORGE_INPUT,
    TEXFORGE_OUTPUT. The command must exit 0 and leave a file at {output}."""
    tmpl = os.environ.get("TEXFORGE_GENERATOR_COMMAND", "").strip()
    if not tmpl:
        raise RuntimeError("generator 'command' needs TEXFORGE_GENERATOR_COMMAND, e.g. "
                           "'my-image-tool --prompt-file {prompt_file} --out {output}'")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pfile = dest.with_name(dest.stem + "_prompt.txt")
    pfile.write_text(prompt, encoding="utf-8")
    vals = {"prompt": prompt, "prompt_file": str(pfile), "input": str(image) if image is not None else "", "output": str(dest)}
    args = []
    for tok in shlex.split(tmpl, posix=(os.name != "nt")):
        if tok == "{input}" and image is None:
            continue
        for k, v in vals.items():
            tok = tok.replace("{" + k + "}", v)
        args.append(tok)
    if not args:
        raise RuntimeError("TEXFORGE_GENERATOR_COMMAND is empty")
    env = dict(os.environ, TEXFORGE_PROMPT=prompt, TEXFORGE_PROMPT_FILE=vals["prompt_file"],
               TEXFORGE_INPUT=vals["input"], TEXFORGE_OUTPUT=vals["output"])
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(workdir))
    except FileNotFoundError as e:
        raise RuntimeError(f"generator command not found: {e}") from e
    if proc.returncode != 0 or not dest.exists():
        raise RuntimeError(f"generator command produced no image at {dest} (exit {proc.returncode}).\n"
                           f"{proc.stdout[-800:]}\n{proc.stderr[-400:]}")
    return dest


def _one(prompt: str, dest: Path, workdir: Path, timeout: int, image: Path | None = None) -> Path:
    """The single generation entry point: write one image for `prompt` (optionally an edit of `image`) to `dest`."""
    b = backend()
    if b == "none":
        return _none_one(prompt, dest, workdir, timeout, image)
    if b == "command":
        return _command_one(prompt, dest, workdir, timeout, image)
    return _codex_one(prompt, dest, workdir, timeout, image)


def _codex_one(prompt: str, dest: Path, workdir: Path, timeout: int, image: Path | None = None) -> Path:
    cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "workspace-write", "--json", "-C", str(workdir)]
    if image is not None:
        cmd += ["-i", str(image)]
        # `-i` is variadic: end the option list with `--` so the instruction is not swallowed as a file name
        cmd += ["--", EDIT_INSTRUCTION.format(prompt=prompt)]
    else:
        cmd += [INSTRUCTION.format(prompt=prompt)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    # The thread id from the JSON event stream names the generated_images/<thread-id>/ subfolder, so
    # concurrent runs cannot pick up each other's images.
    tid = None
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "thread.started":
            tid = ev.get("thread_id")
            break
    new = sorted((GEN_DIR / tid).glob("*.png")) if tid and (GEN_DIR / tid).exists() else []
    if not new:
        raise RuntimeError(f"codex produced no image (exit {proc.returncode}).\n{proc.stdout[-800:]}\n{proc.stderr[-400:]}")
    src = max(new, key=lambda p: p.stat().st_mtime)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest


def generate(prompt: str, out_dir: Path, name: str, n: int = 1, timeout: int = 420, parallel: int = 3) -> list[Path]:
    """Generate n images -> out_dir/<name>_src_<i>.png. Raises RuntimeError with a printable fallback."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend() == "none":
        _none_one(prompt, out_dir, out_dir, timeout)
    if backend() == "codex" and not codex_available():
        raise RuntimeError("`codex` CLI not found on PATH")
    jobs = [(prompt, out_dir / f"{name}_src_{i + 1}.png") for i in range(n)]
    results: list[Path] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, min(parallel, n))) as ex:
        futs = [ex.submit(_one, p, d, out_dir, timeout) for p, d in jobs]
        for f in futs:
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001 - report every failure
                errors.append(str(e))
    for e in errors:
        print(f"[gen] failed: {e}", file=sys.stderr)
    if not results:
        raise RuntimeError("image generation failed")
    return results


def print_manual(prompt: str) -> None:
    via = "via codex" if backend() == "codex" else f"(generator '{backend()}')"
    print(f"\nImage generation {via} is unavailable. Paste this prompt into any image generator "
          "(ChatGPT, etc.), save the PNG, then run `texforge make <image> ...`:\n")
    print(prompt)


def generate_edit(prompt: str, image: Path, out_dir: Path, name: str, timeout: int = 420) -> Path:
    """Image-to-image: ask codex to edit `image` (attached with -i). One run, one result -> out_dir/<name>.png"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend() == "codex" and not codex_available():
        raise RuntimeError("`codex` CLI not found on PATH")
    return _one(prompt, out_dir / f"{name}.png", out_dir, timeout, image=Path(image))
