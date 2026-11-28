# Tasks and the check-in protocol

> **The runtime's base changed on 2026-11-28** (`THIRD-PARTY.md`). The tasks under `issues/` were written against
> the earlier base and are being re-checked against the new one, one by one. Before you claim one, confirm that the
> files it names still exist under `victory/` and that its fault still shows; if not, say so on the issue instead
> of working it. `KNOWN-FAULTS.md` is current.

A task is one GitHub issue, labelled `task`, its area and its cards. The files under `issues/` carry the same text as the GitHub
issues, one per number. Anyone with a GitHub account can take one, work it on their own fork, and send it back as a
pull request.

**Where tasks live.** Every task is an issue on this repository, the same one that holds the runtime in `victory/`
and the tools in `harness/`. Each issue's title starts with the task's number in this kit. GitHub gives the issue
its own number as well, and the two can differ: wherever this page writes `#<n>`, use GitHub's number, so the link
works, and keep the task's own number for your branch name.

Runtime paths in a task are relative to `victory/`; harness paths are relative to the repository root.

This page is the protocol. Follow it literally: the comment lines are read by people and by the claims bot, and a
claim is only a claim when it has the exact form below.

## Why issues, and not one shared document

A shared document would be edited by every agent at once: two agents change the same lines, one overwrites the
other, and nobody can tell who started first. An issue comment is atomic and time-stamped by GitHub. The first
valid `CHECK-IN` on a free task is the claim, and there is no race to lose: either your comment is there or it is
not.

## The three areas

The project is split into three areas, each led by one maintainer. Every task names one area in its `Area:` line,
and its issue carries that area's label. A task lists the files it may change; the table says which area a file
belongs to, so a question about a file goes to the right task and maintainer.

| Area | Label | Holds (runtime paths under `victory/`) |
|---|---|---|
| Tools and measurement | `area: tools` | `harness/`, `scripts/claims.mjs`, this kit; the runtime's diagnostics and counters: `crates/victory-gpu/src/budget.rs`, `host/stats.rs`, `copystat.rs`, `stamps.rs`, `stalls.rs`, `kernelstat.rs`, `live.rs`, `events.rs`, `capture.rs`, `passes.rs`, `host/recheck.rs` |
| CPU side: speed and memory | `area: cpu` | the submitting thread's work and its waits: `crates/victory-gpu/src/host.rs` and `host/` (`refresh.rs`, `relay.rs`, `waits.rs`, `fetch.rs`, `land.rs`), `shadow.rs`, `watch.rs`, `queue.rs`, `memory.rs`, `memclass.rs`, `resource.rs`; the `victory-hle` and `victory-loader` crates |
| GPU side: picture, shaders, the card | `area: gpu` | the `victory-shader` crate; in `crates/victory-gpu/src`: `texture.rs`, `receipts.rs`, `host/hash.rs`, `descriptor/`, `draw.rs`, `pipeline/`, `list.rs`, `indirect.rs` |

Every task carries the same model requirement ([AGENTS.md](AGENTS.md), "Which model"); no task lowers it, so an
issue has no model field.

## Cards

Every task gives a `Cards:` line, and its issue the matching label: `cards: all`, `cards: nvidia`, `cards: amd` or
`cards: intel`, for the graphics cards whose behaviour the task concerns. A task marked for one vendor still needs a
no-regression statement for the others in its pull request: what ran on which cards, with the software adapter's
tests counting where the behaviour is not vendor-specific. How the runtime handles the vendors is in
[CONTRIBUTING.md](CONTRIBUTING.md), "Two vendors, one codebase".

## Held by a maintainer

This work is held by the maintainers: do not claim or start it, and do not open a task for it. Ask on the nearest
public task if your work touches it. A held item that becomes public gets an issue of its own.

- **Tools:** the live switches and an experiment notebook built on them; one end-of-run report holding every
  tool's result, with web addresses masked in stored log lines; a timeline of one texture from its read request on
  disk to its first draw; the fixes an independent check asked of the kernel-call counters, the stale-upload trap
  and the texture classifier; a count of draws whose target format differs from their pipeline's; first real
  captures with the harness's RenderDoc and Nsight recipes; a dry-run measurement for the card-steps design; a
  landing test that failed once.
- **CPU side:** spreading the submitting thread's work across cores, which rebuilds item by item the speed work of
  another public fork: a change journal many threads can ask at once, refreshes split across threads, guards kept
  as bits with a rest after a write, read-write lookup locks, reused snapshot memory, a memory copy that survives
  faults, copy-back landing on several threads, and the runtime's own copy steps done on the card.
- **CPU side:** the fixes for title buffer copies that run out of order, for a title memory copy written while
  earlier readers still run, and for page mapping that does not wait for submissions in the air; two small
  unmerged fixes to copy-back landing; title buffers written straight into card memory (GPU upload heaps,
  where the PC has Resizable BAR); the native memory model (mapping lifetimes, missed dependencies on the card's writes,
  texture refresh, the memory census); copy-back measurement (the fetch queue, whether the title reads what
  copy-back lands, copies without a system call, the narrowed wait, the upload pool); title memory in 2 MiB
  sections; a title thread's spin wait; a faster guard scan; ports from the runtime's upstream; page-number
  recycling, a DirectStorage file reference's lifetime and cancel-by-tag.
- **GPU side:** the fix for 16-bit integer values read through integer views; step two of texture receipts (the
  black seam dashes); falling back to the unlinked pair when the card refuses a linked mesh pipeline; card-side
  ports from another public fork (a texture digest, a cheaper texture change check, staging that skips the write
  watch, texture write-back); the follow-ups of the cascade-shadow port; pipelines carried across builds and made
  ahead; the default form of the card's hash; building ray-tracing structures and putting DLSS in place of the
  title's upscaler, which wait for the maintainers' decision (their starter tasks are public).
- **Parked for now, by the maintainers' reading of what moves the frame:** a disk cache of decoded textures,
  texture decode on the card, skipping the warm-up when nothing changed, async compute on a second queue, the one
  untranslated instruction no program of the title needs, the fog program's loop cap, the card-side work for when
  the card is the limit, and the picks of a third fork's review. Ask on a related task before starting one.

## Before you claim

1. Read [AGENTS.md](AGENTS.md), [KNOWN-FAULTS.md](KNOWN-FAULTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
2. List the open tasks and read the ones you could take, newest comments first. Skip what "Held by a maintainer"
   lists.
3. Choose a task labelled `open`. The bot keeps the `open` and `claimed` labels; when the newest comments disagree
   with a label, the comments decide.

## Take: post one CHECK-IN comment

Post exactly this line as one comment on the task:

```text
CHECK-IN: <handle> | model: <model name> | effort: <level> | plan: <one line> | branch: <branch>
```

- `<handle>` is your GitHub login: the one that posts the comment and opens the pull request.
- `<model name>` is the model you are actually running, as your tool reports it, or `human` when you work by hand.
  If you do not know it, write `unknown`; a maintainer decides.
- `<effort>` is the effort level you set for this task: `low`, `medium`, `high`, `xhigh` or `max`, or the name your
  tool uses for it.
- `<plan>` is one line: what you will change first. It is not a promise of the finished design.
- `<branch>` is the branch on your fork, `task/<number>-<short-name>`, for example `task/40-sampler-branch`.
- Keep it to one line, with no `|` inside a field.

The first valid `CHECK-IN` on a free task holds it; GitHub's timestamp decides when two arrive close together. The
bot labels the task `claimed`, and answers a later `CHECK-IN` by someone else with one comment naming the holder.
You need no assignment and no permission: the comment is the claim. If your model, effort or branch changes, post a
new `CHECK-IN` with the real values.

## Claims: one per issue, alive by the hour

- **One open claim per issue, and one claimed issue per agent session**, whatever number of models or tools you drive.
- **Check in at least every 60 minutes**, with one of:
  - a new comment, one line: `PROGRESS: <what changed> | next: <what is next>` (a new `CHECK-IN` counts too);
  - a push of a new commit to the claimed branch of your fork, whose pull request starts `Task: #<n>`.

  Reason: an abandoned agent session otherwise blocks the task silently, and an hour is long for an agent.
- **A person working by hand** writes `model: human` and checks in at least every 48 hours, because a person does
  not check in hourly.
- **After the window passes without a check-in, the claim has lapsed.** The bot posts
  `CLAIM LAPSED: <handle>, no check-in since <time>; open to claim`, turns the label to `open`, and anyone may claim
  the task with a new `CHECK-IN`, the old holder included. A `PROGRESS` after the lapse does not revive the claim.
- **A lapsed holder's branch stays theirs.** The new holder may build on it and credits them in the pull request.
- **While you hold a task you own the files it lists**, until you release it or it lapses. A change you need in
  another claimed task's files goes to its holder as a comment on that task, not as an edit.
- **Edits are not read.** To correct a comment, post a new one; an edited `CHECK-IN` or `PROGRESS` counts for
  nothing.
- Only your own comments and pushes keep your claim alive; a question, a design note or someone else's comment on
  the task does not.
- Ask on the task when you need an answer, then keep working or `RELEASE`.

The bot (`.github/workflows/claims.yml`, rules in `scripts/claims.mjs`) reads every open task every 15 minutes and
each issue again when it gets a comment. GitHub runs a schedule on a best-effort basis, so a lapse can be noticed some
minutes late; the times in the comments decide.

## Work: your fork, your branch, one pull request

1. Fork this repository and push your work to the branch your claim names.
2. Open a draft pull request against `main` before you change code.
3. Keep the pull request to one task.
4. The description starts with these four lines (the pull request template has them):

   ```text
   Task: #<n>
   Model: <model name>
   Effort: <level>
   Tested on: <vendor>, <card>, <driver>
   ```

   Then it lists every command you ran with its result, and says what you could not test and on which vendors:

   ```markdown
   ## Commands run

   - `VICTORY_WARP=1 cargo test -p victory-shader --locked sampler_choice` — 12 passed, 0 failed
   - `cargo clippy --workspace --all-targets --locked -- -D warnings` — clean

   ## What could not be tested, and on which vendors

   The fixture has no title data; the programs still need a warm-up on a real copy. Not run on an AMD card.
   ```

5. Meet the quality gates ([CONTRIBUTING.md](CONTRIBUTING.md), "Quality gates") and the task's own "done when"
   commands. A bug fix starts with a test that fails before the change; where the task says no test can exist, say
   why in the description.
6. Commit only what may be published. No game content, no captures, no dumps, no bulk extracted table, no personal
   details and no machine paths.

## Done: post CHECK-OUT

When the pull request is ready for review, post this line on the task:

```text
CHECK-OUT: <handle> | PR: #<n> | result: <one line>
```

`<n>` is the pull request's number in this repository. The result is one line: what changed and what the commands
showed. From then on the claim holds without check-ins while the pull request is in review.

The maintainers' review pipeline reads the pull request. It is screened and reviewed before anything is built; then
it is built and tested. An accepted change is taken into the main line as the maintainers' commit with you credited,
and the task is closed then. The merge is the maintainers' call; `CHECK-OUT` marks the task ready.

If review finds a problem, fix it on the same branch and comment on the pull request. If you cannot finish, use
`RELEASE` so someone else may take the task.

## Stuck or wrong task: post RELEASE

Post this line on the task:

```text
RELEASE: <handle> | reason: <one line>
```

The task is free again at once, and anyone may post `CHECK-IN`. Say what you learned: the reason line, or a
comment after it, should name the command, the error or the part of the design that stopped you. Leave your branch
where it is; it may save the next person a day. A maintainer's `RELEASE` frees a task too, for example when its
pull request closes without a merge.

## New tasks

Anyone may propose a task with the task template (`.github/ISSUE_TEMPLATE/task.md`). A proposal says the area, the
cards, the goal in two sentences, what is known and how, where the code is, what "done" means with exact commands,
the test and whether it runs without the title, what is out of scope, the size, and what hardware or files it
needs. The maintainers may reword it for the public rules, then add it as an issue.

The maintainers add tasks when fewer than three are open in an area. An area with three open tasks is not short of
work, even when the tasks are hard.

## The comment forms at a glance

| Moment | Comment |
|---|---|
| Take | `CHECK-IN: <handle> \| model: <model name> \| effort: <level> \| plan: <one line> \| branch: <branch>` |
| Check in | `PROGRESS: <what changed> \| next: <what is next>` |
| Done | `CHECK-OUT: <handle> \| PR: #<n> \| result: <one line>` |
| Stuck or wrong | `RELEASE: <handle> \| reason: <one line>` |

Each starts the comment. Nothing else is a claim, a check-in, a release or a completion. The bot's own comments
start `CLAIM LAPSED:`, `CLAIM HELD:`, `CLAIM NOT READ:` or `MODEL REQUIREMENT:`; the same words from anyone else
change nothing.

## House rules for every comment

- Write "the title", never its name, and put no personal details and no machine paths in a comment.
- Post nothing from the game: no captures, assets, executable bytes, dumps or bulk extracted tables. A shader may be
  named beside its 16-hex id where that helps.
- Be honest about the model and the effort. They are how reviewers apply the model requirement, not a ranking.
- One task per pull request. If the work needs files of another task or area, say so on the task instead of
  widening it.

## The claim rules in one line, for each issue

Every issue ends with this line, which points here:

```text
Claim: one CHECK-IN comment holds it; check in every 60 minutes (48 hours for model: human); TASKS.md has the forms.
```
