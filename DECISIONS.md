# Decisions

## Use of AI tools

AI assistants (Claude via Claude Code) were used throughout this build, as the
README anticipates. What they were used for:

- **Scaffolding and iteration**: stories were planned and implemented through
  BMad-style ticketing; code was written with AI assistance and every line was
  reviewed before commit.
- **Test generation**: the pytest suites under `tests/` (tracer, all-pages,
  fuzzy, upload-validation) were drafted with AI and checked against the
  documented contract in `README.md`.
- **A manual bug-hunt that found a real defect**: driving the live server with
  curl (rather than only running the suite) exposed a crash in
  `POST /api/extract` when `names_json` contained plain strings instead of
  `{first_name, last_name}` pairs — the suite was green because no test posted
  non-empty `names_json`. Fixed at the router boundary with two regression
  tests (`bb52e65`).
- **Documentation**: this file and `DESIGN.md` were drafted with AI assistance
  and edited by hand.

The engineering decisions recorded in `DESIGN.md` were made by reviewing and
owning the resulting code, per the README's "own every line" condition.
