"""
Rendering layer. The controller (app.py) talks only to this interface, so the same
control flow drives both the zero-dependency plain tier and the optional Rich tier.

PlainUI uses stdlib print/input with ANSI colour gated on a real TTY (the same rule
run.sh uses). RichUI overrides the heavy rendering with `rich`; input still goes
through input() so it works identically over flaky SSH.
"""

import os
import sys


def make_ui(prefer_rich=True, force_plain=False):
    if force_plain:
        return PlainUI()
    if prefer_rich:
        try:
            import rich  # noqa: F401
            return RichUI()
        except Exception:
            pass
    return PlainUI()


class PlainUI:
    rich = False

    def __init__(self):
        self.color = (sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"
                      and "NO_COLOR" not in os.environ)

    # ---- styling helpers ----
    def _c(self, code, s):
        return "\033[%sm%s\033[0m" % (code, s) if self.color else s

    def bold(self, s):
        return self._c("1", s)

    def dim(self, s):
        return self._c("2", s)

    # ---- primitives ----
    def clear(self):
        if self.color:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
        else:
            print("\n" * 2)

    def print(self, s=""):
        print(s)

    def header(self, title, lines=None):
        bar = "=" * 64
        print(self.bold(bar))
        print(self.bold(" " + title))
        print(self.bold(bar))
        for ln in (lines or []):
            print(" " + ln)

    def rule(self, text=""):
        if text:
            print(self.dim("-- %s %s" % (text, "-" * max(0, 59 - len(text)))))
        else:
            print(self.dim("-" * 64))

    _TAGS = {"ok": ("32", "[ ok ]"), "info": ("36", "[info]"),
             "warn": ("33", "[warn]"), "err": ("31", "[ !! ]"), "fail": ("31", "[FAIL]")}

    def note(self, level, msg):
        code, label = self._TAGS.get(level, ("0", "[--]"))
        print(" %s %s" % (self._c(code, label), msg))

    def table(self, headers, rows):
        n = len(headers)
        w = [len(str(h)) for h in headers]
        for r in rows:
            for i in range(n):
                w[i] = max(w[i], len(str(r[i])))

        def line(cells):
            return "  ".join(str(cells[i]).ljust(w[i]) for i in range(n))
        print(" " + self.bold(line(headers)))
        for r in rows:
            print(" " + line(r))

    # ---- input ----
    def ask(self, prompt, default=None):
        sfx = " [%s]" % default if default not in (None, "") else ""
        try:
            r = input(self._c("36", "%s%s > " % (prompt, sfx)))
        except EOFError:
            # A leaf prompt with a default returns it; a control prompt (no default)
            # propagates EOF so the run loop exits instead of spinning on empty input.
            if default is None:
                raise
            return str(default)
        r = r.strip()
        if r:
            return r
        return "" if default is None else str(default)

    def confirm(self, prompt, default=False):
        r = self.ask("%s (%s)" % (prompt, "Y/n" if default else "y/N")).strip().lower()
        if not r:
            return default
        return r in ("y", "yes")

    def pause(self, msg="press Enter"):
        try:
            input(self.dim(" -- %s --" % msg))
        except EOFError:
            pass


class RichUI(PlainUI):
    rich = True

    def __init__(self):
        super().__init__()
        from rich.console import Console
        self._console = Console()

    def clear(self):
        self._console.clear()

    def header(self, title, lines=None):
        from rich.panel import Panel
        body = "\n".join(lines or [])
        self._console.print(Panel.fit(("[bold]%s[/bold]\n" % title) + body,
                                      border_style="cyan"))

    def rule(self, text=""):
        self._console.rule("[dim]%s[/dim]" % text if text else "", style="grey42")

    _RICH_TAGS = {"ok": ("green", "ok"), "info": ("cyan", "info"),
                  "warn": ("yellow", "warn"), "err": ("bold red", "!!"),
                  "fail": ("bold red", "FAIL")}

    def note(self, level, msg):
        style, label = self._RICH_TAGS.get(level, ("white", "--"))
        self._console.print(" [%s]\\[%s][/%s] %s" % (style, label, style, msg))

    def table(self, headers, rows):
        from rich.table import Table
        t = Table(show_edge=False, header_style="bold", pad_edge=False)
        for h in headers:
            t.add_column(str(h))
        for r in rows:
            t.add_row(*[str(c) for c in r])
        self._console.print(t)
