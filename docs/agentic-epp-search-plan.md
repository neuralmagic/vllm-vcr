# Agentic, GPU-free EPP hyperparameter search

Plan (early). Goal: an agentic-CI pipeline that searches EPP (inference-gateway
Endpoint Picker) plugin settings for the best performance, iterating entirely on
GPU-free `inference-sim` replicas driven by a once-captured hardware profile. The
agent runs in an `agentic-ci` openshell sandbox in GitHub Actions, edits the
`llm-d-infra` config, and opens a PR with the winning settings.

## Why this works (the core idea)

EPP decides which replica each request lands on. Its plugins (scorers:
kv-cache-aware, precise-prefix-cache, load/queue-aware, ...) and their weights are
the hyperparameters. They change the *distribution* of work across replicas, not
the per-token compute of any one engine.

The replay cache freezes per-token engine behavior into a GPU-free model. So if we
hold the engine model fixed and only vary EPP, the simulation isolates exactly the
thing we're tuning: routing, queueing, batching, KV reuse across replicas. The sim
already enforces real `max-num-seqs` / token-budget queueing and backpressure, so
EPP decisions have real consequences in the sim. Run N sim replicas (no GPU,
cheap), let EPP route across them, measure aggregate SLO attainment. Iterate
hundreds of times for the cost of CPU minutes.

Two properties the sim gives us that real GPUs cannot:
- **Noise-free A/B**: replay the *same* arrival schedule + prompts across every
  iteration (direct workload replay) with a fixed seed, so a metric delta is a
  pure EPP effect, not workload RNG. This is what makes a search loop converge.
- **Throughput at the cost of CPU**: a full benchmark iteration is minutes, not a
  GPU reservation, so an autoresearch loop of hundreds of trials is affordable.

## The crux: the profile cache key must be EPP-independent

Today `hash_config.py` hashes the full rendered manifests, which include EPP. If
the search used that key, every EPP tweak would be a cache MISS and re-capture on
a GPU, defeating the whole point.

**Required change:** split the hash. The *engine profile* cache key covers only
what determines engine timing (model, engine image, hardware/topology, workload:
prompt/output distributions + arrival rate + concurrency envelope). EPP plugin
settings are explicitly excluded. Then every EPP variant the agent tries is a
cache HIT against the one captured profile → GPU-free replay. Capture happens once
(or once per engine/model/hardware/workload change), search runs forever without a
GPU.

## Components

### A. Performance-gated tests (the fitness function) — prerequisite
`analyze.sh` today asserts traffic-served / no-failures / tpot-reasonable
(pass/fail). Search needs a *scalar objective* plus *hard SLO gates*:
- Objective (maximize): e.g. throughput at fixed rate, or SLO attainment =
  fraction of requests under a TTFT/e2e target; or a composite.
- Hard gates (must hold or the trial is rejected): correctness/traffic-served, no
  dropped requests, replicas stay inside the profiled concurrency envelope (see
  Risks).
Emit the objective as a machine-readable number (JUnit property / JSON) the agent
reads each iteration.

### B. Profile-once, possibly a sweep
We have single-point capture. EPP rebalancing moves per-replica concurrency, so
capture a small **concurrency sweep** (a few load points) so the trace-fitted
model interpolates across the regime the search explores, instead of extrapolating
off one operating point. Store each profile under the EPP-independent key.

### C. Multi-replica GPU-free sim topology — new
EPP picks among endpoints, so we need >=2 replicas. Extend the disaggregated
overlay to N GPU-free `inference-sim` replicas behind one EPP/InferencePool. Each
replica = frontend + sim (no GPU), all replaying the same cached profile. This is
the main new infra piece; it's cheap precisely because there are no GPUs.

### D. The agentic loop (agentic-ci openshell + autoresearch)
GitHub Actions step runs `agentic-ci run --backend openshell` with the Claude Code
harness. The agent executes an autoresearch loop (modify → verify → keep/discard →
repeat):
1. Propose an EPP config edit within a bounded, whitelisted surface (the
   epp-plugins values: scorer set + weights + thresholds).
2. Deploy the N-replica sim topology in replay mode (cache HIT, no GPU).
3. Run guidellm against the fixed replayed workload; `analyze.sh` emits the
   objective + gate verdicts.
4. Keep/discard vs best-so-far; append to a search ledger (config, metrics, seed).
5. Repeat until token/time budget or convergence.
Output: a PR to `llm-d-infra` with the winning epp-plugins values, the ledger, and
the measured delta vs the baseline config.

### E. Safety / cost envelope
- openshell sandbox: Landlock FS isolation + network policy
  (`.agentic-ci/openshell-policy.yml`) locked to GitHub + the bench cluster API +
  S3 (trace fetch) only.
- A **diff-scope gate**: the agent may only modify the whitelisted epp-plugins
  path; reject any trial whose diff touches anything else (no infra escape).
- Token budget cap on the loop; per-trial wall-clock cap.
- GPU-free: cost is CPU minutes per trial, the economic unlock.

## agentic-ci integration notes (from the openshell docs)

- Agent invocation is `claude --permission-mode bypassPermissions --model <M>
  --output-format stream-json -p "<PROMPT>"` inside the sandbox; we provide the
  prompt/skill and model.
- `agentic-ci` primitives map onto our needs: **gates** = the SLO/diff-scope
  checks, **skills** = the reusable "run one benchmark trial" + "propose EPP edit"
  steps, **verdict** = best config selection, **telemetry** (OTEL) = token/cost
  tracking for the budget cap.
- Network policy needs explicit binary paths (`--binary /usr/local/bin/claude`)
  and endpoint allow-list; add the cluster API + `s3.amazonaws.com` +
  `oidc.cks.coreweave.com` (IRSA) to `.agentic-ci/openshell-policy.yml`.
- GitHub Actions wiring isn't documented on the site; treat agentic-ci as a CLI
  step in a workflow (provider creds via service-account JWT for CI, then
  `provider refresh rotate` before the run).

## Auth / OIDC composition (reuse what we have, no static creds)

The pipelines already authenticate to the remote cluster with **zero static
credentials**: `setup-waldorf-auth` writes a kubeconfig whose user is an **exec
credential plugin** (`get-oidc-token.sh`). On every API call kubectl runs the
plugin, which mints a fresh **GitHub Actions OIDC token** (from
`ACTIONS_ID_TOKEN_REQUEST_URL` + `..._TOKEN`) and presents it to the cluster; the
CoreWeave/CKS API trusts GitHub's OIDC issuer via Workload Identity Federation.
Tokens are short-lived and re-minted per call, scoped to the GitHub identity's
cluster RBAC.

This travels into the openshell sandbox almost verbatim, the clean composition:

- **Carry the plugin, not a credential.** Inject the two job-scoped env values
  (`ACTIONS_ID_TOKEN_REQUEST_URL`, `ACTIONS_ID_TOKEN_REQUEST_TOKEN`), ship
  `get-oidc-token.sh` into the sandbox, point the sandbox kubeconfig at it.
  kubectl/helmfile inside the sandbox then auto-refresh exactly like on the runner
  — no token-expiry problem across a long autoresearch loop, because the plugin
  re-mints on demand for the job's lifetime. No static kubeconfig is ever written.
- **Network policy** (`.agentic-ci/openshell-policy.yml`) must allow: the GitHub
  OIDC token endpoint (so the plugin can mint), the **bench cluster API** (confirm
  which: `waldorf` `api.6787d4-...` vs the CKS bench cluster whose OIDC issuer is
  `a3fee6a3-...`), and `s3.amazonaws.com` for result upload. Lock everything else.
- **In-pod S3 (IRSA) is independent and already composes.** The sim replicas fetch
  their trace from S3 via projected-SA-token → CKS-OIDC → AWS-role
  (`metrics-uploader` / `llm-d-cluster-workload`), regardless of how the *agent*
  authenticated to deploy them. The agent never handles AWS trace-fetch creds; the
  pods assume the role themselves. The search loop inherits that auth unchanged.
- **Scope = blast radius.** Cluster access is the GitHub-OIDC identity's RBAC
  (bench namespaces only) and egress is pinned by the network policy, so a
  `bypassPermissions` agent is bounded to: edit whitelisted epp-plugins files,
  deploy into bench namespaces, read S3 results. Pin the GitHub-OIDC subject's
  cluster RBAC and the AWS trust policy to this repo/workflow so the injected
  request token can't mint broader access.
- **openshell's own provider creds** (Anthropic/GCP for the agent model) are a
  separate axis via the agentic-ci provider (`provider refresh rotate` for CI
  service accounts), orthogonal to cluster auth above.

## Phased implementation

Revised ordering (per decision): stand up the **agentic-ci/openshell harness
first** as a walking skeleton, then deepen the search loop underneath it. The
auth/sandbox integration is the riskiest unknown, so de-risk it before investing
in fitness functions and multi-replica topology.

1. **openshell walking skeleton.** Get `agentic-ci run --backend openshell` going
   in a GitHub Actions job with the Claude Code harness and a trivial prompt
   ("clone llm-d-infra, edit one whitelisted file, open a PR"). Prove: provider
   creds (Anthropic via CI service-account JWT + `provider refresh rotate`), the
   network policy allow-list, and the **cluster-auth bridge** (exec-plugin +
   injected `ACTIONS_ID_TOKEN_REQUEST_*` → `kubectl cluster-info` succeeds from
   inside the sandbox). This validates the whole auth/sandbox composition with no
   ML in the loop.
2. **One agent-driven benchmark trial.** Have the sandboxed agent deploy the
   existing single-replica replay topology (cache HIT, GPU-free) and run one
   benchmark + `analyze.sh`, reading back the result. Reuses the replay-cache
   reusable workflow; proves the agent can drive the existing pipeline end-to-end.
3. **Split the hash** (`hash_config.py`): EPP-independent engine-profile key so EPP
   edits stay cache HITs. Unit-test EPP-only change → HIT, engine/model/workload
   change → MISS.
4. **Fitness gates** in `analysis/` + `analyze.sh`: emit a scalar objective + hard
   SLO gates; keep existing pass/fail as one gate.
5. **Multi-replica sim overlay**: N GPU-free sim replicas behind EPP; validate EPP
   routes across them and the gate reads aggregate metrics. (Builds on the
   replay-cache-rollout reusable workflow.)
6. **Profile sweep capture**: capture a few concurrency points; verify model
   interpolation across the envelope.
7. **Agentic search loop**: give the (already-wired) sandboxed agent the
   autoresearch loop over the bounded EPP surface against the multi-replica sim,
   with the diff-scope gate, budget cap, and PR-out verdict. By this point the
   harness (1-2), the EPP-independent cache (3), the fitness (4), and the topology
   (5) all exist, so this step is the loop logic + the search prompt/skill, not new
   plumbing.

## Skills / tools the agent needs

Design principle: expose a small set of **thin, deterministic verbs** that wrap
the pipeline mechanics, so the agent spends its reasoning on *which EPP config to
try next*, not on orchestrating kubectl/helmfile/guidellm. Each verb is a scripted
command (one auditable CLI surface, e.g. `epp-search <verb>`), validated and
side-effect-bounded. The LLM's value is hypothesis-forming over the EPP surface
(reading scorer semantics, spotting interactions), so the read/context verbs
matter as much as the action ones.

**Context / read-only (the agent learns the space):**
- `search-space` — the bounded, whitelisted tunable surface as a machine-readable
  schema: each EPP knob (scorer set, weights, thresholds) with type + valid range.
  The single most important tool; without it the agent can't search legally.
- `explain-plugins` — semantic docs of each scorer/knob (what it optimizes, known
  interactions), pulled from the guides, so the agent reasons with domain
  knowledge instead of blindly perturbing numbers.
- `describe-workload` — the captured workload's characteristics (prompt/output
  distributions, arrival pattern, prefix structure) so the agent can reason about
  which scorers matter for *this* trace.
- `ledger` / `best-so-far` — the durable search history (every trial's config +
  metrics + seed) and the current champion. The agent's memory across iterations
  and what makes the loop resumable; read it to avoid repeats and follow gradients.

**Action (the only writes):**
- `propose <key=value…>` / `apply-config <file>` — write a candidate into the
  whitelisted epp-plugins values file ONLY, after schema-validating it (reject
  unknown/out-of-range before writing). The diff-scope gate enforces nothing else
  changed.
- `run-trial` — the fitness function and workhorse: deploy the N-replica GPU-free
  sim topology in replay mode (cache HIT), run guidellm against the **fixed
  replayed workload + pinned seed**, run `analyze.sh`, return structured JSON: the
  objective scalar, hard-gate verdicts, and key metrics (throughput, TTFT
  p50/p95, ITL, queue depth, per-replica balance). Idempotent and deterministic so
  trials are directly comparable. Appends to the ledger.

**Control / verdict:**
- `budget-status` — tokens spent (agentic-ci OTEL), trials run, wall-clock left;
  drives the stop decision.
- `finalize` — apply the champion, write the ledger + a summary (delta vs
  baseline), open the PR. The pipeline's verdict output.

**Gates (agentic-ci post-gates + custom):**
- diff-scope (only the epp-plugins file changed), envelope (reject trials pushing
  any replica outside the profiled concurrency envelope — the extrapolation guard),
  plus the stock `sensitive-files,commit-author,gitleaks`.

Why these and not "let the agent run raw kubectl": deterministic verbs make every
trial reproducible and comparable (same seed/workload), keep the blast radius to
one file, and let `run-trial` enforce the envelope + fairness invariants the search
depends on. The agent picks configs; the harness guarantees the measurement.

## Risks / open questions (be honest)

- **Profiled envelope / extrapolation.** EPP can drive a replica to a concurrency
  the capture never saw; the model would extrapolate. Mitigate with the sweep (B)
  and a hard gate that rejects trials pushing replicas outside the profiled range.
- **TTFT-under-load fidelity.** Our known soft spot (~25 vs 13 ms saturated). EPP
  search that optimizes TTFT percentiles inherits that gap. Either improve the
  TTFT-under-load model first, or target metrics the sim nails (throughput, ITL,
  completion count, queue depth) for the first iteration of this feature.
- **Does EPP's effect even show up in the sim?** It must, via per-replica
  queueing/batching/KV — but validate on real GPUs once: run the agent's winning
  config on real hardware and confirm the predicted ranking holds (a single
  confirmation run, the same way we validated replay fidelity).
- **EPP config surface.** Enumerate the bounded, whitelisted tunables
  (epp-plugins scorers/weights/thresholds) so the agent searches a real space, not
  arbitrary infra. The precise-prefix-cache helmfile shows the shape
  (scorer + ZMQ + tokenizer sidecar config).
- **Determinism across replicas.** Per-request seeds must be stable so the same
  workload replays identically across trials; confirm multi-replica sim seeding is
  reproducible.

## Dependencies

- Replay-cache rollout (the reusable `_benchmark.yaml`) — the search topology is a
  multi-replica variant of it.
- The engine pin / fidelity work carried in the replay-cache handover.
