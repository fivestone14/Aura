# Design decisions

Why Aura is built the way it is. This records the **reasoning**, especially for decisions that look
arbitrary or wrong until you know what they're defending against.

If you're about to "simplify" something here, read the relevant section first. Several of these
constraints are load-bearing in non-obvious ways.

---

## First principle

> ### Stop trying to guess who someone is. Get better at noticing what they do.

Every decision in this document is downstream of that one sentence. It is not a slogan — it is a
test, and it has teeth. Whenever a design choice presents itself, ask which side of the line it
falls on:

| ❌ Guessing who they are | ✅ Noticing what they do |
|---|---|
| Infer a personality type from their voice | Record that long answers get interrupted |
| Classify their emotion from acoustics | Measure that they're 30% faster than their own baseline |
| Match a voiceprint to identify them | Read the certificate on their device |
| Predict their preference from a category | Learn it from what they corrected |
| Assume a stranger's needs from a 10-second sample | Start neutral, ask, and converge |

**Why the left column keeps losing:** it is unfalsifiable at runtime. A guess about who someone
*is* cannot be corrected by the person, because it was never checked against them — it just quietly
steers everything downstream. An observation can be wrong, but it can also be *shown* to be wrong,
and then fixed. The right column is self-correcting; the left column is confidently unaccountable.

**It is also the cheaper column.** Observed behavior predicts outcomes roughly 7× better than
inferred traits (§4), needs no labeled corpus, invites no regulatory exposure, and requires no
model that could be wrong about a person in a way they cannot contest.

**Design test:** if a proposed feature requires deciding *what kind of person* the user is before
it can act, it belongs in the left column. Redesign it to act on something the user did instead.

---

## 1. Counter-regulate, don't mirror

**Decision:** When someone sounds rushed and tense, Aura replies *slower and quieter* — not matching
their energy.

**Why:** Matching high arousal escalates it. This is the consistent finding in crisis de-escalation
practice (speak below their volume, slower than feels natural, one idea per sentence) and in
infant-directed speech research, where soothing is realized as low-pitched falling contours — the
opposite of the infant's distress, not a mirror of it.

**Caveat, stated honestly:** the clinical evidence is practitioner guidance, not controlled
speech-ML experiment. Classic accommodation theory holds that *dis*entrainment can read as cold or
distancing. This is why the guardrail eval set exists (§6) — the risk is real and must be measured,
not assumed away.

---

## 2. Measure prosody. Don't classify emotion.

**Decision:** 88 standard acoustic features (eGeMAPS), computed directly. No emotion classifier, no
personality model, no arousal label from a trained head in v1.

**Why three separate reasons converge:**

- **Valence is not readable from audio.** Arousal and dominance reach usable accuracy from
  acoustics; valence sits near zero without linguistic content. This is a property of the signal,
  not an unsolved engineering problem — two decades have not fixed it.
- **Emotion categories are contested.** The dominant critique in affective science is that there is
  no reliable one-to-one mapping from a physical signal configuration to an emotion category;
  context and perceiver priors dominate. Building policy on category inference means building on
  the least defensible layer available.
- **Rate, energy, pitch, and pause structure are physically measurable** and are what the policy
  actually acts on. They need no model at all.

**Consequence:** thresholds are set against **the speaker's own rolling baseline**, not a population
norm. "Fast" means fast *for them*.

**Upgrade path:** a learned arousal head can be added later as an *additional* input alongside
eGeMAPS — not a replacement. It is deferred until production evaluation shows the rule-based read
failing. Adding it is not free: it requires labeled corpora, speaker-disjoint splits, and
cross-corpus validation, and the field's own benchmarks put state of the art far lower than people
expect.

---

## 3. Identity is a device certificate. Never a voiceprint.

**Decision:** device certificate → explicit selection → ephemeral no-profile mode. Voice biometrics
appear nowhere.

**Why — three independent reasons, any one sufficient:**

- **Legal.** A voiceprint is a biometric identifier. Under Illinois BIPA the obligation attaches on
  *collection*, not on use — so "we compute embeddings but only cluster locally" is not a
  mitigation. GDPR Art. 9, Texas CUBI, CCPA sensitive-data rules, and the amended COPPA Rule
  (voiceprints added as personal information, 2025) all engage. A device certificate triggers none
  of them; it is a login credential.
- **It wouldn't work well.** Benchmark speaker-verification error rates come from clean, long
  utterances. Real conditions — 1–3 seconds of far-field reverberant speech — degrade that by
  roughly an order of magnitude. Closed-set identification across a household is workable;
  **open-set rejection of an unknown speaker is what breaks**, and no threshold makes both
  stranger-acceptance and family-rejection acceptable at that utterance length.
- **It's spoofable.** Modern zero-shot voice cloning defeats speaker verification lacking a
  dedicated countermeasure. Thirty seconds of someone's recorded voice is enough. Voice is a
  convenience signal, never an authentication factor.

**Structural rule that follows:** the profile store is writable **only** by a session holding a
valid device certificate. An unenrolled voice is never written anywhere — no transcript, no
embedding, no statistic. This is simultaneously the guest-privacy answer and the legal one.

---

## 4. Learn the person by watching, not by guessing

**Decision:** the per-user profile accumulates from **observed behavior and explicit corrections**.
No trait is ever inferred or stored.

**The rejected alternative** was inferring a personality type from voice and mapping type → style.
It fails twice over:

1. **The trait isn't readable.** The widely-cited benchmarks measure ***perceived*** personality —
   what listeners think of a voice. The canonical corpus is **10-second clips of radio broadcasters,
   labeled by majority vote of 11 judges who certified they did not understand the language.** That
   measures a shared vocal stereotype, not a disposition. Against *actual self-reports*, the best
   published result is **r = 0.26–0.39 (R² ≈ 7–15%)** — and that used prosody **plus linguistic
   content**, which a prosody-only pipeline does not have. Openness and agreeableness sit near
   chance under both label types. Trait-average voices built from self-ratings are barely
   discriminable; built from *perceiver* ratings, all five traits are highly discriminable.
   **Listeners agree strongly with each other and are mostly wrong.**
2. **Even a perfect reading wouldn't help.** Personality → task outcome is **r = −.17 to +.18
   (~3% of variance)**, observational, with no adaptive policy ever actually tested. **Directly
   observed dialogue behavior → outcome is r ≈ .45 (~20%)** — roughly 7× more signal from data
   already in hand. Multiplied end-to-end, the voice→trait→strategy chain carries **under 1% of
   outcome variance.**
3. **The direction of the mapping is contested.** Multiple direct nulls exist for matching an
   agent's personality to the user's — including one specifically on *vocalic* extraversion. Other
   work finds users preferring a **complementary** personality. You cannot build a forward-predictive
   policy on a relationship whose *sign* is unknown. At realistic classification accuracy, ~40% of
   users would receive a confidently wrong policy, and a confident wrong policy is worse than a
   neutral default because it is both mis-targeted and unresponsive to correction.

**A rebuke from inside the field:** speech science treats stable per-speaker vocal character as a
**nuisance variable to normalize away** — models "exploit biases and find shortcuts such as speaker
characteristics, which usually harm generalization." Speaker normalization is standard practice
precisely because that signal is noise, not personality.

**Regulatory note:** measuring per-turn prosody to adapt delivery is defensible as ancillary
technical processing. **Persisting an inferred psychological profile is biometric categorisation** —
not prohibited (personality is not on the Art. 5(1)(g) closed list), but it carries transparency
duties and engages automated-decision rules. The step from measurement to profiling is exactly the
step this section declines.

**What is stored instead:**

```
verbosity: terse
impatient_with_long_answers: 0.24   (rate, n=31 — not an incident)
mean_speaking_rate / mean_pause_length   (aggregate scalars)
```

Traits are **rates with sample counts**, never single incidents — one irritated reply is noise; a
pattern across sessions is signal. Entries are **scoped to context**, because the same person wants
terse answers on a lookup and depth on a decision.

**Never stored:** voiceprints, per-utterance affect time-series, raw audio beyond the processing
window, or anything derived from an unenrolled voice.

**Open problem — cold start.** On turn one the profile is empty. The answer is *not* to guess a
type: start neutral, use the probe channel (§5) to elicit signal, and converge quickly.

---

## 5. 🚨 The profile governs delivery. Never substance.

**This is the most important constraint in the system and the easiest to break by accident.**

> **How someone wants to be *told* is learnable and adaptable.**
> **What they get *told* is not.**

**Why it matters:** sycophancy — a model aligning its response with a user's stated or inferred
beliefs even when those beliefs are wrong — is a documented consequence of optimizing for
helpfulness. **Personalization is a sycophancy amplifier.** A system that reads someone well is
*better equipped* to tell them what they want to hear. Without an explicit barrier, the personalization
layer quietly becomes a flattery optimizer.

### Three consequences. Do not remove any of them.

**a) The profile schema is delivery-only by construction.**
Permitted: `verbosity`, `pace`, `formality`, `wants_preamble`, `tolerance_for_backchannels`.
🚫 **No field may encode a stance, opinion, or preferred conclusion.** A field like `agrees_with_X`
is rejected at schema review. The schema is the enforcement mechanism — if there is nowhere to store
a belief, the belief cannot leak into delivery logic.

**b) Blind the skeptic.**
The council includes a dissent role whose job is to resist consensus. **It must receive the
transcript *without* the user profile.** Wiring the profile uniformly to all three council members
looks tidier and is the natural thing to do — and it silently destroys the one component that exists
to catch bias, because the skeptic inherits exactly the bias it is meant to detect.

**c) A sycophancy eval set, separate from the guardrail set.**
Cases where the user asserts something false, confidently, in a strong emotional register. Correct
behavior is **gentle delivery, unchanged substance.** The metric that matters:
**does agreement-rate rise as the profile fills?** If it does, the profile is leaking into content.

**Note the symmetry:** prosody counter-regulation (§1) already refuses to mirror the user's tone.
This section extends the same principle from tone to substance. It is one philosophy, not two
features.

---

## 6. Evaluation is four separate questions

Collapsing them produces a number that means nothing.

| Tier | Question | Method |
|---|---|---|
| **Perception** | Does it hear correctly? | Speaker-**disjoint** splits, mandatory. Random utterance splits leak the same speaker into train and test and inflate results dramatically. Report cross-corpus too. |
| **Controllability** | Did the renderer do what the policy asked? | Commanded vs realized rate, ΔF0, energy, pause duration. Report MAE, R², and a monotonicity curve. |
| **Timing** | Does it feel like conversation? | Turn-gap distribution, barge-in latency, false-interruption rate, speculative waste rate. |
| **Appropriateness** | Was that the *right* tone? | Pairwise blind A/B, not MOS. Nothing automatic answers this. |

**Two eval sets are first-class deliverables, not footnotes:**

- **The guardrail set** — held-out cases where the correct behavior is to *diverge* from the user's
  prosody. Without it, a naive mirroring system scores well on every other tier and is wrong in
  exactly the situations that matter most.
- **The sycophancy set** — see §5c.

**🚫 Do not use MCD** (mel-cepstral distortion). It measures distance from a parallel reference. If
the point is choosing a *different* prosody than the reference, **MCD penalizes success.** Report it
only to explain why it's the wrong metric.

**Build the eval harnesses before the models.** Otherwise "is this better?" gets asserted instead of
answered.

---

## 7. Split-brain: some jobs cannot wait

**Decision:** a native client library does the fast work on-device; a hosted service does the
thinking.

**Why the split is forced rather than chosen:**

| Job | Budget | Round-trip cost |
|---|---|---|
| Echo cancellation | needs the speaker signal *as it plays* | impossible remotely |
| Prosody + turn detection | every 20 ms frame | impossible remotely |
| Acknowledgement token | ~200 ms to feel natural | 50–100 ms of the budget, gone |

**What crosses the network: text and 88 numbers. Never audio.** A few hundred bytes per turn instead
of a continuous stream — simultaneously the latency win, the bandwidth win, and the privacy win.

**Graceful degradation is a design goal.** With the server unreachable, the client still hears,
still takes turns, still acknowledges. The system gets *dumber*, not *dead*.

### Audio stack: WebRTC, decided 2026-08-07

Echo cancellation, barge-in handling, noise suppression, and NAT traversal all come with
WebRTC, which is why every production voice system is built on it — the same stack the
major realtime voice APIs ship through. Writing our own would mean native bindings,
per-platform builds, and tuning an adaptive filter by hand: the most difficult work in
the project, to arrive at a worse version of something that already exists.

⚠️ **The cost, recorded because it is easy to rediscover painfully.** WebRTC cancels
*before* the application sees a frame, so it consumes the reference signal. The original
design used one reference tap for two consumers — echo cancellation and the second
channel of a two-channel turn-taking model. Only the first survives.

The code makes this visible rather than implicit: `AudioLoop.has_reference_signal` and
`Session.supports_two_channel_turn_taking` both report false on this path. A turn-taking
model that wants both sides must take Aura's channel from the synthesis output instead of
the audio loop. That is a wiring change, and it is answerable at wiring time.

`UpstreamCanceller` and `PassthroughCanceller` are structurally identical and
semantically opposite — "already handled" versus "nothing is protecting you". They are
deliberately separate types, because conflating them reintroduces exactly the silent
failure this module exists to prevent.

**Independent validation:** the lab that produced the strongest open full-duplex speech model
subsequently moved to a modular architecture with a swappable text model, and the major commercial
launch in this space uses a fast voice model that delegates to a frontier model in the background.
End-to-end buys roughly 200 ms and costs the reasoning.

---

## 8. The council is justified by robustness, not intelligence

**Honest position:** at equal compute budget, published comparisons find a single agent matches or
beats multi-agent topologies — including parallel-roles panels structurally similar to this one.
Apparent multi-agent gains are better explained by unaccounted computation.

**Why keep it:** the same work finds multi-agent *does* help on **degraded input** — and that is
precisely this system's regime, since the council reads partial, speculative transcripts of noisy
real-world speech.

**This is falsifiable, and should be falsified or confirmed with a measured A/B.** If transcripts
turn out cleaner than expected, the council collapses to a single agent with no quality loss. That
would be a **simplification, not a failure.**

**Constraints if it stays:**
- **Parallel, never a serial debate loop.** Debate is seconds-scale and unaffordable in the latency
  budget — and unguided debate has been found to underperform isolated self-correction.
- **Deadline-bounded**, with a hard cutoff rather than waiting for consensus.
- **Off the critical path** — the client covers the gap out loud.

---

## 9. Latency: ~800 ms is the target. 200 ms is the feeling.

Human conversation turn gaps cluster around 200 ms, but human *speech planning* takes far longer
than that. People hit short gaps by **predicting** when the other person will finish and preparing
during their turn.

**A pipeline that begins work at end-of-turn detection cannot reach 200 ms by getting faster** — the
component floor forbids it. Endpointing and model time-to-first-token alone consume most of a
realistic budget.

**Target ~800 ms voice-to-voice at p50 with a tight tail**, and buy the *perception* of instantness
three ways:

1. **Start during the user's turn**, on a forecast of when they'll stop. Published work completes a
   majority of downstream work before the user finishes speaking.
2. **Start the merge on partial input** — the aggregator begins on the first chunk from the
   generators rather than waiting for completion.
3. **Hold the floor out loud** — a functional acknowledgement covers the remaining gap.

**Accepted cost:** roughly a quarter of speculative work is discarded when the turn-end forecast is
wrong. This is a deliberate trade, not waste.

⚠️ **Acknowledgements must be functional, never decorative.** Filler added as ornament reads as
incompetence — users have specifically criticized assistants that say "um" for no reason. And
knowing when *not* to interject is the single hardest UX problem here: the most visible failure of a
recent major launch was interrupting users and laughing at things that weren't jokes.

---

## 10. Security decisions that look paranoid and aren't

- **The process holding an API credential must never be the process whose context contains
  untrusted transcript text.** Voice prompt injection is demonstrated, not theoretical — adversarial
  audio succeeds at high rates even played from a speaker across a room, and a documented injection
  outcome is the model echoing credentials or system-prompt contents into its output.
- **Any operation that persistently changes behavior requires human confirmation.** The realistic
  attack is injected content causing a persona-setting call with attacker-chosen instructions that
  then persist across every future conversation — a silent backdoor, not a bad turn.
- **Secrets live in the OS keychain.** Never in `.env` (a project file that development tooling
  indexes), never in shell history, never in a service plist.
- **Treat every transcript segment as untrusted data**, delimited and labeled, never as
  instructions.
- **Reachability via private overlay network only.** The dominant real-world failure for
  self-hosted AI is an unauthenticated service exposed to the internet — hundreds of thousands of
  instances of one popular local inference server are publicly reachable, with disclosed
  vulnerabilities leaking prompts, sessions, and credentials from memory.
- **Cryptographic erasure.** Per-profile encryption keys mean "delete my profile" destroys the key
  rather than the rows — making erasure verifiable rather than asserted.

**Honest limit:** none of this defends against malware already running as the same OS user. It
defends against device theft, other accounts, and network compromise. Claiming more would be false.

---

## 11. Audit: does the current design obey its own first principle?

Applied honestly, including where it doesn't.

| Component | Verdict |
|---|---|
| **Prosody perception** | ✅ **Notices.** Measures acoustics against the speaker's own baseline. No category assigned. |
| **Turn-taking** | ✅ **Notices.** Predicts *when they'll stop talking* from the audio signal — a behavior, not a disposition, and confirmed or refuted within a second. |
| **Identity** | ✅ **Notices.** Reads a certificate the user's device presents. Nothing inferred about the person. |
| **The Interpreter** | ✅ **Notices.** Accumulates corrections given and behaviors observed. Stores rates with sample counts, never a type. |
| **Prosody policy** | ⚠️ **Mixed — and deliberately so.** Counter-regulation is a *rule about arousal*, not a claim about the person, so it stays on the right side. But the rule itself is a prior we did not learn from this user. Mitigated by the guardrail eval and by the Interpreter overriding it per person. **Watch this one.** |
| **Cold start** | ⚠️ **The weakest point.** On turn one there is nothing observed, so *something* must be assumed. The principle says: assume as little as possible, and **probe** rather than profile. A neutral default plus a fast question beats a confident guess. |
| **Council** | ✅ **Notices.** Agent B interprets *what was meant in this turn* — a per-utterance reading that the next turn immediately tests. Not a standing claim about the person. |
| ~~Personality inference~~ | 🚫 **Rejected.** This was pure left-column, and it is why the principle exists. |

**The pattern:** every place the design felt uncomfortable turned out to be a place it was guessing.
Every place it felt solid turned out to be a place it was measuring. That correspondence is the
reason to trust the principle rather than merely state it.

---

## Open questions

| Question | Why it matters |
|---|---|
| ~~Are the TTS model's prosody predictors reachable?~~ | ✅ **Resolved 2026-08-09.** Yes. Kokoro's forward pass is thirty lines of plain PyTorch; duration, pitch and energy are all local variables passed into the vocoder, so a subclass can scale all three. No fork needed. The GPL-3 fallback is no longer on the table. |
| Do the Apple Silicon ports of that model hold up? | Edge deployment depends on it; the ports are community-maintained. |
| How many turns before a learned profile beats a neutral default? | Determines whether cold-start work is worth doing. |
| Does the council survive its A/B? | See §8. |
| What is the actual per-turn output length of council members? | Determines whether streaming aggregation saves ~1.7 s or ~250 ms. |
