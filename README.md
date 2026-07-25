<div align="center">

<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://media.x.ai/v1/website/spacexai-symbol-white-transparent-0c31957f.png">
    <source media="(prefers-color-scheme: light)" srcset="https://media.x.ai/v1/website/spacexai-symbol-black-transparent-6435cf42.png">
    <img alt="SpaceXAI logo" src="https://media.x.ai/v1/website/spacexai-symbol-black-transparent-6435cf42.png" width="96">
  </picture>
  <br>
  Grok Build (<code>grok</code>)
</h1>

**Grok Build** is SpaceXAI's terminal-based AI coding agent. It runs as a
full-screen TUI that understands your codebase, edits files, executes shell
commands, searches the web, and manages long-running tasks — interactively,
headlessly for scripting/CI, or embedded in editors via the Agent Client
Protocol (ACP).

[Installing the released binary](#installing-the-released-binary) ·
[Building from source](#building-from-source) ·
[Fork customizations](#fork-customizations) ·
[Documentation](#documentation) ·
[Repository layout](#repository-layout) ·
[Development](#development) ·
[Contributing](#contributing) ·
[License](#license)

![Grok Build TUI](https://media.x.ai/v1/website/universe-tui-screenshot-6f7a0837.png)

**Learn more about Grok Build at [x.ai/cli](https://x.ai/cli)**

This repository contains the Rust source for the `grok` CLI/TUI and its agent
runtime. It is synced periodically from the SpaceXAI monorepo.

A small `SOURCE_REV` file at the root records the full monorepo commit SHA
for the version of the code present in this tree.

</div>

---

## Installing the released binary

Prebuilt binaries are published for macOS, Linux, and Windows:

```sh
curl -fsSL https://x.ai/cli/install.sh | bash   # macOS / Linux / Git Bash
irm https://x.ai/cli/install.ps1 | iex          # Windows PowerShell
grok --version
```

See the [changelog](https://x.ai/build/changelog) for the latest fixes,
features, and improvements in each release.

## Building from source

Requirements:

- **Rust** — the toolchain is pinned by [`rust-toolchain.toml`](rust-toolchain.toml);
  `rustup` installs it automatically on first build.
- **[DotSlash](https://dotslash-cli.com)** — required so hermetic tools under
  [`bin/`](bin/) (notably [`bin/protoc`](bin/protoc)) can download and run.
  Install it and ensure `dotslash` is on your `PATH` **before** building:

  ```sh
  cargo install dotslash
  # or: prebuilt packages — https://dotslash-cli.com/docs/installation/
  /usr/bin/env dotslash --help   # sanity check
  ```

- **protoc** — proto codegen resolves [`bin/protoc`](bin/protoc) via DotSlash,
  or falls back to a `protoc` on `PATH` / `$PROTOC`.
- macOS and Linux are supported build hosts; Windows builds are best-effort
  and not currently tested from this tree.

```sh
cargo run -p xai-grok-pager-bin              # build + launch the TUI
cargo build -p xai-grok-pager-bin --release  # release binary: target/release/xai-grok-pager
cargo check -p xai-grok-pager-bin            # fast validation
```

The binary artifact is named `xai-grok-pager`; official installs ship it as
`grok`. On first launch it opens your browser to authenticate — see the
[authentication guide](crates/codegen/xai-grok-pager/docs/user-guide/02-authentication.md).

## Fork customizations

> [!IMPORTANT]
> This fork intentionally diverges from `xai-org/grok-build`. Preserve the
> behavior and configuration contracts in this section when rebasing onto a
> newer upstream version. The initial implementation was based on upstream
> commit `6e38642`.

### Configuration additions

The following fields can be set globally under `[models]` and overridden for
one model under `[model."<id>"]`:

```toml
[models]
stream = true
user_agent = "my-client/1.0"
responses_system_prompt_as_instructions = false

[model."third-party-responses"]
# Existing provider/model fields go here.
stream = true
user_agent = "my-client/1.0"
responses_system_prompt_as_instructions = true
```

- `stream` defaults to `true`. Set it to `false` to use a non-streaming HTTP
  request. Chat Completions, Responses, and Anthropic Messages all use the
  same completion, empty-response, truncation, and retry pipeline in either
  mode.
- `user_agent` sets the HTTP `User-Agent`. A per-model value overrides the
  global value and takes precedence over a `User-Agent` entry in
  `extra_headers`.
- `responses_system_prompt_as_instructions` defaults to `false`. When enabled,
  system messages are removed from Responses `input` and joined, in order,
  into the top-level `instructions` field. This supports OpenAI-compatible
  providers that reject system-role input items.

### Retry and authentication policy

- Every HTTP status code, including 4xx, 429, and 5xx, is retryable with
  exponential backoff. `Retry-After` is still honored when present, and
  `x-should-retry: false` does not bypass this fork's retry policy.
- The retry loop has a 600-second wall-clock budget and retains `max_retries`
  as a count-based safety net; whichever limit is reached first stops the
  request. Non-HTTP errors that cannot be repaired by resampling, such as
  invalid configuration, serialization failures, and max-token truncation,
  retain their dedicated terminal behavior.
- On HTTP 401, refresh-capable sessions (OAuth/session bearer resolver,
  configured auth provider, or devbox) hand the first failure to the session
  for credential refresh and resubmission. Static API-key/BYOK endpoints have
  no refresh mechanism, so their 401 responses use the normal exponential
  backoff path instead.

### Responses streaming compatibility

- A non-standard `response.metadata` SSE event is ignored only when the JSON
  payload's top-level `type` exactly matches that name. Other unknown events
  remain serialization errors so future content events are not hidden.
- If `response.completed.response.output` contains no assistant text but the
  stream already delivered non-empty `response.output_text.delta` events, the
  final assistant text is reconstructed from those deltas. A non-empty
  terminal response remains authoritative, and `response.incomplete` is not
  converted into a successful completion.

### Upstream rebase checklist

Add the official repository as `upstream` once, then rebase the fork branch:

```sh
git remote add upstream git@github.com:xai-org/grok-build.git # once
git fetch upstream
git switch main
git rebase upstream/main
```

The most likely conflict areas are:

- model/config propagation in `xai-grok-shell`, `xai-chat-state`, and
  `xai-grok-sampler::SamplerConfig`;
- request dispatch and terminal response construction in
  `xai-grok-sampler/src/actor/request_task.rs`;
- HTTP headers and Responses SSE decoding in
  `xai-grok-sampler/src/client.rs`;
- retry classification/time budgeting in `xai-grok-sampler/src/retry.rs`,
  `xai-grok-sampler/src/actor/request_task.rs`, and
  `xai-grok-sampling-types/src/error.rs`;
- 401 refresh eligibility in
  `xai-grok-shell/src/session/acp_session_impl/sampler_turn.rs`.

After resolving conflicts, verify all fork contracts before updating the
published branch:

```sh
cargo fmt --all -- --check
cargo test -p xai-grok-sampler
cargo clippy -p xai-grok-sampler --all-targets -- -D warnings
cargo clippy -p xai-grok-pager-bin --all-targets -- -D warnings
cargo build -p xai-grok-pager-bin --release
```

Because a rebase rewrites published history, update the fork only with lease
protection after reviewing the result:

```sh
git push --force-with-lease origin main
```

## Documentation

Full online documentation is available at
[docs.x.ai/build/overview](https://docs.x.ai/build/overview).

The user guide ships with the pager crate:
[`crates/codegen/xai-grok-pager/docs/user-guide/`](crates/codegen/xai-grok-pager/docs/user-guide/)
— getting started, keyboard shortcuts, slash commands, configuration, theming,
MCP servers, skills, plugins, hooks, headless mode, sandboxing, and more.

## Repository layout

| Path | Contents |
|------|----------|
| `crates/codegen/xai-grok-pager-bin` | Composition-root package; builds the `xai-grok-pager` binary |
| `crates/codegen/xai-grok-pager` | The TUI: scrollback, prompt, modals, rendering |
| `crates/codegen/xai-grok-shell` | Agent runtime + leader/stdio/headless entry points |
| `crates/codegen/xai-grok-tools` | Tool implementations (terminal, file edit, search, ...) |
| `crates/codegen/xai-grok-workspace` | Host filesystem, VCS, execution, checkpoints |
| `crates/codegen/...` | The rest of the CLI crate closure (config, MCP, markdown, sandbox, ...) |
| `crates/common/`, `crates/build/`, `prod/mc/` | Small shared leaf crates pulled in by the closure |
| `third_party/` | Vendored upstream source (Mermaid diagram stack) — see below |

> [!IMPORTANT]
> The root `Cargo.toml` (workspace members, dependency versions, lints,
> profiles) is **generated** — treat it as read-only. Prefer editing per-crate
> `Cargo.toml` files.

## Development

```sh
cargo check -p <crate>        # always target specific crates; full-workspace builds are slow
cargo test -p xai-grok-config # per-crate tests
cargo clippy -p <crate>       # lint config: clippy.toml at the repo root
cargo fmt --all               # rustfmt.toml at the repo root
```

## Contributing

> [!NOTE]
> External contributions are not accepted. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

First-party code in this repository is licensed under the **Apache License,
Version 2.0** — see [`LICENSE`](LICENSE).

Third-party and vendored code remains under its original licenses. See:

- [`THIRD-PARTY-NOTICES`](THIRD-PARTY-NOTICES) — crates.io / git dependencies,
  bundled UI themes, and **in-tree source ports** (including openai/codex and
  sst/opencode tool implementations)
- [`crates/codegen/xai-grok-tools/THIRD_PARTY_NOTICES.md`](crates/codegen/xai-grok-tools/THIRD_PARTY_NOTICES.md)
  — crate-local notice for the codex and opencode ports (license texts +
  Apache §4(b) change notice)
- [`third_party/NOTICE`](third_party/NOTICE) — vendored Mermaid-stack index
