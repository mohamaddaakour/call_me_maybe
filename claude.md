# Becoming a Highly Productive Senior Engineer with Claude & Claude Code

A practical, no-fluff guide. Every section is something you'll actually use this week, not AI theory.

---

## 1. Claude Fundamentals

### How Claude works (just enough to use it well)

Claude is a large language model: it predicts the next chunk of text given everything before it (your prompt + the conversation + any files/context). There's no persistent "understanding" between separate conversations unless you explicitly re-supply context (a file, a summary, memory). Practical implications:

- **Everything Claude "knows" about your task is in the current context window.** If it's not in there, Claude can't use it — even if you told it in a different session.
- Claude does **not** execute your code by "understanding" your intent like a human colleague would over months. It pattern-matches against the text you give it plus training data. The more precise and complete your input, the more precise the output.
- Claude Code (the CLI/agentic tool) closes this gap by letting Claude *read files, run commands, and see real output* instead of guessing — this is the single biggest productivity unlock over chat-only Claude.

### Context windows

The context window is the total amount of text (prompt + conversation history + files + tool outputs) Claude can "see" at once, measured in tokens.

- Current Claude models support very large context windows (hundreds of thousands of tokens), but **large ≠ free**. Performance and precision degrade as you cram in irrelevant material — this is sometimes called "context rot."
- Treat the context window like RAM, not like a hard drive. Load what's relevant to *this* task, not your entire repo history.
- In Claude Code specifically, context includes: your `CLAUDE.md` project file, recently read files, recent tool outputs (test runs, git diffs, file listings), and the conversation itself. All of this competes for space.

### Tokens

A token is roughly ¾ of a word (for English text; code tends to tokenize a bit less efficiently due to symbols/whitespace).

- Every message you send, every file Claude reads, every command output, every line of its response — all consumes tokens from the context window and costs money/time.
- Big generated files, verbose logs, and large pasted stack traces are token-expensive. A 10,000-line `package-lock.json` dumped into context is mostly wasted tokens.
- **Practical rule:** if you wouldn't paste something into a colleague's Slack DM to ask for help, don't paste it into Claude's context unedited.

### How Claude reasons about code

Claude does best when it can:
1. **See the actual code**, not your paraphrase of it (paraphrasing loses exact type signatures, imports, naming).
2. **See how it's used** (callers, tests, related modules) — reasoning about a function in isolation without callers often produces subtly wrong assumptions.
3. **Verify itself** — in Claude Code, Claude can run tests/linters/type-checkers instead of guessing if code is correct. Always let it do this rather than trusting a single-shot answer for anything nontrivial.
4. Work incrementally on **well-scoped chunks**. A single prompt asking for "refactor the whole auth system and add OAuth and fix the bug in signup" produces worse results than three focused prompts, because Claude (like a human) does better with one primary objective at a time.

### Model selection: Opus, Sonnet, Haiku

| Model | Best for | Speed/Cost |
|---|---|---|
| **Opus** | Hardest problems: novel architecture design, gnarly multi-file refactors, ambiguous debugging, security-critical reasoning, anything where being wrong is expensive | Slowest, most expensive |
| **Sonnet** | Default daily driver. Feature implementation, most debugging, code review, most Claude Code agentic work | Balanced — this is your 80% model |
| **Haiku** | High-volume, low-complexity tasks: simple boilerplate, quick syntax questions, bulk formatting/renaming, fast autocomplete-style tasks, sub-agent tasks in larger workflows | Fastest, cheapest |

**Practical selection heuristic:**
- Start on **Sonnet** by default for real engineering work.
- Escalate to **Opus** when: you've gone back and forth 2-3 times without a correct answer, the task spans architecture/design tradeoffs, or the blast radius of a wrong answer is large (auth, payments, data migrations, concurrency).
- Drop to **Haiku** for: mechanical/repetitive subtasks, or as the model powering background/parallel sub-agents in Claude Code where the task is narrow and well-defined.
- **Don't** default to the most expensive model for everything — it's slower and the quality gap is often irrelevant for routine work. Match model to task difficulty, exactly like you'd match a junior/senior engineer to a task.

---

## 2. Prompt Engineering for Software Engineers

Think of prompting Claude like writing a ticket for a very fast, very literal contractor who has no memory of your last conversation with them unless you show them the transcript.

### Writing effective prompts — the core structure

A good engineering prompt has four parts:

1. **Goal** — what should exist when you're done, in concrete terms.
2. **Context** — relevant files, constraints, existing patterns to follow.
3. **Constraints** — what NOT to do, style/architecture rules, out-of-scope items.
4. **Definition of done** — how you'll know it's correct (tests pass, lint clean, matches spec).

**Weak prompt:**
> "Add caching to the user service"

**Strong prompt:**
> "Add an in-memory LRU cache (max 500 entries, 5 min TTL) to `UserService.getUser()` in `src/services/user.ts`. Follow the same caching pattern already used in `src/services/product.ts` (see the `withCache` wrapper). Do not add a new dependency — use what's already in `package.json`. Cache should be invalidated on `updateUser()`. Add unit tests in `user.test.ts` covering hit, miss, TTL expiry, and invalidation. Run `npm test` when done and fix any failures."

### Giving clear requirements

- State **inputs/outputs explicitly** (types, shapes, edge cases) rather than "make it work correctly."
- If there's a spec, ticket, or design doc, **paste it or point Claude Code at the file** rather than summarizing from memory — summaries lose detail.
- Specify **non-functional requirements** up front if they matter: performance targets, backward compatibility, supported browsers/Node versions, etc. Claude won't infer these.

### Providing context

- Reference **existing code patterns**: "follow the same error-handling pattern as `orders.ts`" is more reliable than describing the pattern in prose.
- In Claude Code, let Claude actually read the relevant files (`grep`/`find`/open them) rather than pasting snippets from memory — stale pastes cause bugs.
- Mention **what's already been tried** when debugging — repeating a failed approach wastes a turn.
- For multi-file tasks, name the files you know are relevant; let Claude Code discover the rest via search rather than you manually pasting an entire directory tree.

### Prompt patterns for coding

**Planning prompt** (use before big changes):
> "Before writing any code: read `src/payments/` and propose a plan for adding Stripe webhook idempotency. List the files you'll touch, in order, and any risks. Wait for my approval before implementing."

**Debugging prompt:**
> "This test fails: [paste exact failure + stack trace]. Here's the relevant code: [file/path]. I've already checked X and Y, they're not the cause. Find the root cause — don't just make the test pass, explain *why* it was failing."

**Refactoring prompt:**
> "Refactor `OrderProcessor` to remove the duplicated validation logic across `validateOrder`, `validateReturn`, and `validateExchange` into a shared function. Behavior must be identical — run the existing test suite before and after and confirm no regressions. Don't change public method signatures."

**Code review prompt:**
> "Review this diff as a senior engineer would: [paste diff or point to branch]. Flag correctness bugs, security issues, missed edge cases, and anything that violates patterns elsewhere in this repo. Ignore pure style nits unless they affect readability significantly. Rank issues by severity."

**Architecture prompt:**
> "I need to add multi-tenant support to this SaaS app (currently single-tenant, [stack]). Propose 2-3 architectural approaches (e.g., schema-per-tenant vs row-level `tenant_id`), with tradeoffs for our scale (~500 tenants, Postgres). Recommend one and outline the migration path from current state."

**Documentation prompt:**
> "Write API documentation for the endpoints in `routes/orders.ts` in the same style as `docs/api/products.md`. Include request/response examples pulled from the actual TypeScript types, not invented ones."

**Test generation prompt:**
> "Write unit tests for `calculateShipping()` in `shipping.ts` using [test framework already in repo]. Cover: happy path, zero-weight edge case, international vs domestic, and the error path when address is invalid. Match the existing test file conventions in this directory."

**Performance optimization prompt:**
> "This endpoint (`GET /api/reports`) takes 3-4s under load. Here's a profile/trace: [paste]. Find the bottleneck before proposing a fix — don't guess. Once identified, propose the smallest change that fixes it, and note any tradeoffs."

**Security review prompt:**
> "Review `auth/` for security issues: injection, auth bypass, insecure token handling, missing rate limiting, secrets in code. For each finding, give severity, exploit scenario, and fix. Don't flag theoretical issues with no realistic exploit path — focus on what matters."

### Common mistakes

- **Vague asks**: "make this better" / "optimize this" without defining better/optimized for what metric.
- **Dumping the whole repo** into context "just in case" — wastes tokens and dilutes relevance.
- **Not specifying constraints**, then being surprised Claude added a new dependency, changed a public API, or used a different style than the rest of the codebase.
- **Accepting output without verification** — treating Claude's code like it's automatically correct, especially for concurrency, security, or math-heavy logic.
- **One giant multi-objective prompt** instead of sequential, scoped requests.
- **Not giving feedback when wrong** — restating the ask instead of pointing at exactly what's wrong wastes cycles.

### Best practices

- Be as specific as you would be writing a ticket for a new hire who's smart but has zero tribal knowledge of your codebase.
- Prefer **iterative, checkpointed** work over one giant prompt for complex tasks — plan → implement → verify → next chunk.
- Always give Claude a way to **self-verify** (tests, linters, type-checkers, build commands) rather than asking it to eyeball correctness.
- Explicitly state what's **out of scope** ("don't touch the database schema") to prevent scope creep.
- Reuse good prompts — build yourself a personal prompt library (see Section 7).

---

## 3. Claude Code

Claude Code is Anthropic's agentic CLI (also available in desktop/VS Code/JetBrains) that lets Claude read your filesystem, run shell commands, edit files, use git, and iterate — not just chat.

### Installation

```bash
npm install -g @anthropic-ai/claude-code
```

Requires Node.js. Run `claude` from inside a project directory to start. Authenticate via your Anthropic account/API key on first run.

*(Since install steps can change, always check `https://docs.claude.com` for the current instructions if something here looks off — Claude Code ships updates frequently.)*

### Configuration

- **`CLAUDE.md`** — the most important config file. Place it at your repo root (and optionally in subdirectories for monorepos). Claude Code automatically reads it at session start. Put in it:
  - Project overview and architecture summary
  - Build/test/lint commands
  - Coding conventions and patterns to follow
  - Things to never do (e.g., "never modify `migrations/` directly, always use the CLI generator")
  - Directory structure notes for non-obvious layouts
- Keep `CLAUDE.md` **concise and current** — it's loaded into every session's context, so bloat costs tokens on every single run. Treat it like onboarding docs for a new hire, not a full wiki.
- Settings (permissions, allowed tools, hooks) are configurable via `.claude/settings.json` in the project or user-level config — check current docs for exact schema since this evolves.

### Commands

Core interaction patterns (exact command names/flags evolve — check `docs.claude.com` for the current reference):

- Slash-style in-session commands for things like clearing context, compacting conversation history, switching models, and reviewing permissions.
- You can pipe things into Claude Code or point it at specific files/directories to scope its attention.
- Non-interactive/scripted invocation is supported for automation (CI, hooks, batch tasks) — useful for scripting repetitive review/lint tasks.

### Permissions

Claude Code asks for permission before taking actions that modify your system (writing files, running shell commands, git operations) unless you've pre-approved them.

- You can allow-list specific safe commands (e.g., `npm test`, `git status`) so Claude doesn't stop to ask every time.
- **Never blanket-approve destructive commands** (`rm -rf`, force-push, `git reset --hard`, database drops) — keep these behind manual confirmation always.
- In CI/automated contexts, scope permissions tightly to only what that pipeline needs.

### Project awareness

- Claude Code builds situational awareness by reading `CLAUDE.md`, exploring the directory structure, and searching/grepping as needed — it doesn't need the whole repo pasted in.
- For large repos, guide it: "the relevant code is under `services/billing/`" saves it from searching broadly and burning tokens/turns.
- It respects `.gitignore`-style exclusions for what's relevant — keep generated files, build artifacts, and `node_modules` out of its search path.

### Working with Git

Claude Code can run git commands directly: status, diff, branch, commit, log, blame, etc.

- Good workflow: ask Claude to **review its own diff** before committing ("show me `git diff`, does this match what we intended?").
- Have it write commit messages summarizing the actual diff, not a generic message — this produces much better commit history than "fix stuff."
- For risky changes, ask it to create a branch first rather than committing to `main`.
- **You still own git hygiene** — review diffs yourself before pushing, especially for anything touching shared/critical code.

### Working with terminals

- Claude Code can run arbitrary shell commands (with permission) — tests, builds, linters, package installs, curl requests against local servers.
- This is what separates it from chat-only Claude: it can **verify its own work** by actually running things instead of guessing whether code compiles.
- Practical pattern: "implement X, then run the test suite and fix any failures" lets it iterate autonomously instead of you manually copy-pasting errors back.

### Working with large repositories

- Don't try to load an entire large repo into context. Point Claude Code at the specific subdirectory/module relevant to the task.
- Use `CLAUDE.md` files at subdirectory level for monorepos with genuinely different conventions per package.
- For codebase-wide tasks (e.g., "rename this pattern everywhere"), consider scripted/batch approaches (grep + targeted edits) over asking Claude to hold the whole repo in context at once.
- Compact/clear conversation history between unrelated tasks in a long session — stale context from an earlier task can bias later reasoning and wastes tokens.

### Code editing

- Claude Code edits files directly (not just printing a diff for you to copy-paste), and you review the actual change.
- For precise, surgical edits, ask for a specific change ("rename this function and update all call sites") rather than "improve this file," which invites unwanted scope creep.
- Always review the diff — treat AI-written code exactly like a PR from a competent but new teammate: read it before merging.

### Refactoring

- Best practice: **have tests passing before you start**, so Claude Code (and you) can confirm behavior didn't change.
- Ask it to refactor in small, verifiable steps rather than one giant rewrite — easier to catch regressions.
- For large refactors, use the planning pattern first (see Section 2) so you agree on the approach before it touches code.

### Debugging

- Give it the real error/stack trace and let it read the actual failing code and its callers — don't summarize the bug from memory.
- Let it add temporary logging/print statements and re-run to gather evidence rather than guessing at root cause — this mirrors how a good engineer debugs.
- Ask explicitly for root-cause explanation, not just a patch — patches without understanding often mask the real bug.

### Planning

- For anything nontrivial, use a two-phase flow: **plan first, implement second.** Ask Claude to lay out an approach and files affected, review/edit that plan, then say "go implement this."
- This catches misunderstandings when they're cheap (a paragraph) rather than expensive (500 lines of wrong code).

### Automation

- Claude Code can be invoked non-interactively/scripted for things like: automated PR description generation, changelog drafting, lint/style fix passes, dependency upgrade triage.
- Good automation candidates: repetitive, well-specified, low-risk-if-wrong (easily reviewed) tasks.
- Bad automation candidates: anything where a wrong autonomous action is expensive and hard to reverse (auto-committing to main, auto-deploying, modifying production data).

### Productivity workflows

- Keep a **running `CLAUDE.md`** you improve over time as you notice Claude repeatedly getting something wrong or needing the same context explained twice — bake it into the doc instead.
- Use the **plan → implement → verify → commit** loop as your default rhythm for anything beyond trivial changes.
- Batch small independent tasks (e.g., "add these 3 unrelated small fixes") is fine; batch large interdependent tasks is not — split those into sequential prompts.

### Best practices

- Treat Claude Code output like a PR from a real teammate: **you are still accountable for what ships.**
- Keep `CLAUDE.md` accurate — stale instructions actively hurt you (Claude will follow outdated conventions confidently).
- Use version control as your safety net — commit working states often so you can always roll back an AI-driven change that went sideways.
- Don't hand off ambiguous, high-stakes decisions (architecture with long-term lock-in, security-critical design) without your own review — use Claude to generate options, but make the call yourself.

---

## 4. Token & Context Optimization

### How tokens work (practical recap)

Tokens are the unit Claude reads/writes in, roughly ¾ of a word for English prose, less efficient for dense code/symbols. Every part of a session — your prompt, files Claude reads, command output, its response — draws from the same context budget. Cost and latency scale with tokens processed.

### Reducing token usage

- **Don't paste what Claude Code can read itself.** Point it at a file path instead of pasting the file contents — it'll read exactly what it needs.
- **Trim logs/stack traces** to the relevant frames before pasting into chat-only Claude; keep only what points at the bug.
- **Avoid re-pasting the same file** multiple times in one session — it's already in context from the first read.
- **Summarize instead of dumping** for background context that doesn't need verbatim fidelity (e.g., "this service handles billing reconciliation, ~2k LOC, key files are X and Y" beats pasting the whole directory).

### Context management

- **Clear/compact context between unrelated tasks.** Old task context lingering in a long session can bias new reasoning and wastes budget. Claude Code supports clearing/compacting conversation history — use it when switching tasks.
- **Scope tasks narrowly.** "Work in `services/billing/`" is cheaper and more accurate than "work somewhere in this 50k-file monorepo."
- Keep `CLAUDE.md` **lean** — it's loaded every session. Prune stale sections regularly.

### Prompt compression

- Say the constraint once, precisely, instead of three paragraphs of hedging.
- Use structured lists over prose paragraphs for multi-part requirements — easier for Claude to parse, uses fewer tokens than flowing prose saying the same thing.
- Reference existing code ("follow the pattern in `x.ts`") instead of re-explaining a pattern in words.

### Working with large codebases

- Break work into **module-scoped sessions.** Don't ask Claude to reason about the entire codebase's architecture in one shot unless you genuinely need a birds-eye analysis.
- Use `grep`/search-first workflows: let Claude Code search for relevant files rather than you guessing and pasting the wrong ones.
- For cross-cutting changes (rename a widely-used function, upgrade a dependency across many files), consider a **plan-once, then batch-execute** pattern — agree on the exact change, then let Claude Code (or a script) apply it repeatedly rather than reasoning from scratch on every file.

### Cost optimization

- Match model to task (Section 1) — this is the single biggest cost lever. Don't run Haiku-tier tasks on Opus.
- Kill sessions / start fresh rather than letting one conversation balloon indefinitely — long stale sessions cost more per turn as history accumulates.
- For repetitive automated tasks (lint fixes, doc generation), script them to run at appropriate model tiers rather than manually invoking the most expensive model each time.

### Efficient project organization

- One clear `CLAUDE.md` per logical project/package, not one giant file trying to describe an entire monorepo.
- Keep generated artifacts (build output, lockfiles, coverage reports) out of Claude's default search path — they're rarely relevant and are token-expensive if ever read.
- Consistent naming/structure across your codebase reduces how much context Claude needs to correctly infer conventions — this is a case where good engineering practice directly reduces AI cost too.

---

## 5. Real Software Engineering Workflows

### Building a feature

1. **Plan**: "Read [relevant files/ticket]. Propose an implementation plan: files touched, new functions/types, edge cases, and testing approach. Don't code yet."
2. **Review the plan** — adjust scope, catch missed edge cases, confirm it matches existing patterns.
3. **Implement**: "Implement the plan. Run tests as you go."
4. **Verify**: review the diff, run the full test suite, manually sanity-check critical paths.
5. **Document/commit**: have Claude draft the commit message and any doc updates from the actual diff.

### Fixing bugs

1. Reproduce the bug (or get the exact repro steps/error/stack trace).
2. Give Claude the failure evidence + point it at the suspected code — let it read callers too.
3. Ask for **root cause**, not just a patch: "explain why this happens before proposing a fix."
4. Have it add a regression test that would have caught this bug.
5. Verify the fix doesn't break other tests; review the diff for scope creep.

### Code reviews

- Use Claude as a **first-pass reviewer** before human review: "Review this diff for correctness bugs, security issues, and edge cases. Rank by severity."
- Good for catching: missed null checks, inconsistent error handling, security anti-patterns, deviation from repo conventions.
- Not a replacement for human review on **architectural fit, business logic correctness, and team-specific tribal knowledge** — Claude doesn't know your team's unwritten priorities.

### Refactoring

- Confirm tests exist and pass **before** starting — this is your regression safety net.
- Refactor in small, independently verifiable steps.
- Ask Claude to confirm public API/behavior is unchanged, and to run tests after each step.

### Learning a new codebase

- "Give me a high-level map of this repo: main modules, how data flows through the system, and where I'd look to make a change to [specific feature]."
- "Walk me through what happens from `[entry point]` to `[end result]`, referencing actual file/line locations."
- Ask it to identify **non-obvious conventions** ("what patterns does this codebase use that a newcomer might not expect?") — surfaces tribal knowledge fast.

### API development

- Give the spec/contract (OpenAPI, existing endpoint conventions) and have Claude implement to match it exactly, not invent shapes.
- Ask for input validation, error responses, and status codes to match existing patterns in the repo.
- Have it write both the implementation and the corresponding request/response tests together.

### Backend

- Point it at the existing service/module patterns (DI setup, error handling middleware, logging conventions) so new code fits in rather than introducing a new style.
- For data layer changes, explicitly flag migration requirements and ask it to write the migration alongside the code change.

### Frontend

- Reference existing component patterns and design system/style guide so new UI is consistent.
- Ask it to consider loading/error/empty states explicitly — these are commonly skipped if not requested.
- For accessibility-sensitive work, ask explicitly ("ensure this is keyboard-navigable and has correct ARIA roles") — it won't always volunteer this unprompted.

### Full-stack

- Plan the contract first (API shape/types) before implementing both sides — prevents backend/frontend drift.
- Implement backend, verify with tests/curl, then implement frontend against the verified real API rather than an assumed shape.

### Writing tests

- Point at existing test files for framework/convention/style.
- Ask explicitly for edge cases: empty inputs, boundary values, error paths, concurrency where relevant — these are the ones most often missed.
- For bug fixes, always request a regression test tied to the specific bug.

### Documentation

- Generate docs **from the actual code/types**, not from a verbal description — prevents docs drifting from reality.
- Ask it to match existing doc style/structure in the repo.
- For public APIs, request real request/response examples derived from actual types/schemas.

### Pull requests

- Have Claude draft the PR description from the actual diff: what changed, why, how it was tested, any risks/rollback notes.
- Ask it to flag anything in the diff that reviewers should pay special attention to.
- You still write/edit the final PR description — Claude's draft is a strong starting point, not the final word.

### Git workflows

- Use feature branches for anything Claude Code is actively iterating on — easy to discard if it goes wrong.
- Commit at working checkpoints, not just at the end — gives you rollback points mid-session.
- Review `git diff` before every commit, especially after autonomous multi-file changes.

---

## 6. Senior Engineer Best Practices

### How senior engineers interact with Claude

- Treat it like **delegating to a capable but context-free contractor**: give clear scope, constraints, and a definition of done — then verify the output like you would any other contributor's work.
- Use it to **accelerate exploration** (design options, tradeoffs, "what would break if...") more than to replace your own judgment on final decisions.
- Iterate — first output is a draft, not a final answer, especially for anything nontrivial.

### What to automate

- Boilerplate generation (CRUD scaffolding, test scaffolding, config files)
- First-pass code review flagging obvious issues
- Documentation generation from existing code
- Repetitive refactors with a clear, verifiable pattern
- Commit messages / PR description drafts from real diffs
- Debugging legwork (tracing through code, forming hypotheses, running experiments)

### What NOT to automate

- **Final architectural decisions** with long-term lock-in — use Claude to generate options, you make the call.
- **Security-critical logic sign-off** — always have a human review auth, crypto, payment, and permission code, even if Claude wrote it well.
- **Unreviewed production deployments/migrations** — never let autonomous execution touch prod data without a human check.
- **Business-logic correctness** — Claude doesn't know your product's edge-case business rules unless you tell it; verify against actual requirements, not just "does it look reasonable."

### Verifying AI-generated code

- Run it. Don't just read it and assume correctness — execute tests, and manually exercise non-trivial paths.
- Check it against the **actual requirement**, not just "does this look like reasonable code."
- For anything concurrent, cryptographic, or involving floating-point/money math, review with extra scrutiny — these are common silent-failure zones.
- Diff review discipline: read every line of a nontrivial AI-generated diff before merging, same as you would a human PR.

### Avoiding hallucinations

- Hallucinations in code most often show up as: **invented APIs/library functions that don't exist**, **incorrect assumptions about a function's behavior it didn't actually read**, or **confidently wrong explanations of unfamiliar library internals**.
- Mitigation: let Claude Code actually read the library source/docs or run code to confirm behavior, rather than answering from memory for anything you're unsure about.
- If Claude cites a specific API/method you don't recognize, **verify it exists** (check docs, autocomplete, or have it grep for the actual import) before trusting it.
- Ask "are you sure this function/API exists, or are you inferring it?" — Claude will often self-correct when explicitly asked to check.

### Security considerations

- Never let Claude (or yourself) commit secrets/credentials — enforce via `.gitignore`, secret scanning, and explicit "never hardcode credentials" rules in `CLAUDE.md`.
- Explicitly request security review passes on auth, input handling, and anything touching user-supplied data — don't assume it's covered by default.
- Treat AI-suggested dependencies with the same scrutiny as a human PR suggesting a new dependency — check maintenance status, license, and necessity.

### Performance considerations

- Profile before optimizing — ask Claude to identify the actual bottleneck from real data (profiler output, timing logs) rather than guessing.
- Be explicit about performance constraints up front (latency budget, expected scale) — Claude won't infer your scale requirements.
- Watch for common AI-introduced inefficiencies: N+1 queries, unnecessary re-renders, missing indexes, overly broad caching invalidation.

### Maintaining code quality

- Keep `CLAUDE.md` as a living style guide — codify conventions once, benefit every session after.
- Use linters/formatters/type-checkers as an objective quality gate Claude Code can run against itself — don't rely purely on subjective review.
- Periodically audit for consistency across AI-assisted and human-written code — drift happens if conventions aren't reinforced.

---

## 7. Reusable Prompt Library

Copy, adapt, and keep these in your own notes/`CLAUDE.md` as starting templates.

### Feature Implementation
```
Read [relevant files/ticket/spec]. Before coding, propose an implementation plan:
files to touch, new functions/types, edge cases, and testing approach. Wait for
my confirmation before implementing.

Constraints:
- Follow existing patterns in [reference file]
- Do not add new dependencies without asking
- Out of scope: [anything explicitly excluded]

Definition of done: [tests pass / matches spec / etc.]
```

### Debugging
```
This is failing: [exact error/stack trace/failing test name].
Relevant code: [file paths].
Already ruled out: [what you've checked].

Find the root cause before proposing a fix — explain why this happens.
Once found, propose the minimal fix and add a regression test.
```

### Refactoring
```
Refactor [target] to [specific goal, e.g., "remove duplicated validation logic"].
Behavior must be identical — confirm existing tests pass before and after.
Do not change public method signatures / API contracts.
Do this in small, independently verifiable steps, not one large rewrite.
```

### Architecture
```
I need to [goal] in this system (current stack: [stack], current scale: [scale]).
Propose 2-3 architectural approaches with tradeoffs (complexity, performance,
migration cost, team familiarity). Recommend one and outline a migration path
from the current state. Flag any approach that would require a hard cutover
vs one that allows incremental rollout.
```

### Code Review
```
Review this diff/branch as a senior engineer would: [diff or branch ref].
Flag, in order of severity:
1. Correctness bugs
2. Security issues
3. Missed edge cases
4. Deviations from existing repo conventions
Ignore minor style nits unless they materially affect readability.
```

### Documentation
```
Write documentation for [module/API] in the same style/structure as
[reference doc]. Base all examples and type info on the actual code in
[file paths] — don't invent shapes or behavior not present in the code.
```

### Testing
```
Write tests for [target function/module] using [test framework already in repo],
matching conventions in [reference test file]. Explicitly cover:
- Happy path
- Boundary/edge cases: [list if known, e.g., empty input, max size]
- Error paths
- [Concurrency/async edge cases if relevant]
```

### Security Auditing
```
Review [module/directory] for security issues: injection, auth bypass,
insecure token/secret handling, missing input validation, missing rate
limiting, and improper access control. For each finding, give:
severity, realistic exploit scenario, and a concrete fix.
Skip theoretical issues with no realistic exploit path.
```

### Performance Optimization
```
[Endpoint/function] is slow: [metric, e.g., "3-4s under load"].
Profile/trace data: [paste actual profiling output — don't skip this step].
Identify the actual bottleneck before proposing a fix.
Propose the smallest change that resolves it, and note any tradeoffs
(memory, complexity, correctness risk).
```

### Learning Unfamiliar Code
```
Give me a map of [module/repo]: main components, how data flows through
the system end-to-end, and where I'd look to change [specific feature].
Then walk me through the actual code path from [entry point] to
[end result], referencing real file/line locations.
Flag any non-obvious conventions a newcomer wouldn't expect.
```

---

## Quick Reference: The Core Loop

For almost any nontrivial task, this is the rhythm that works:

**Plan → Confirm → Implement → Verify (run it) → Review diff → Commit**

Skipping "plan" on complex tasks or "verify" on any task is where most AI-assisted engineering mistakes come from — not the model being wrong, but skipping the checkpoints a senior engineer would naturally use with any collaborator.
