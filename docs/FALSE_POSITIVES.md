# False-Positive Intuition

The most important question for any perimeter system is not "what does it
detect?" but "what does it falsely fire on?" Every modality in TRITON's
perimeter stack has well-known failure modes. This document is the
honest catalog of them, the in-code mitigations, the mitigations not yet
implemented, and the operational policy for living with the residual
false-positive rate.

## Person detection (M48)

### YOLOv8n on cameras

| FP cause | Why it happens | Frequency |
|---|---|---|
| Mannequins, posters, billboards | Trained on COCO; mannequins look like people | Common in retail / urban |
| Reflections in windows or polished floors | The reflection is a literal person image | Common |
| Strong shadows on textured walls | Shape similarity to a tall thin object | Common at dawn / dusk |
| Statues, garden gnomes | Same as mannequins | Site-specific |
| People in unusual poses (crouched, lying) | Underrepresented in COCO | Sporadic |
| Animals (dogs, deer) | Mostly handled, but partial-view animals confuse the head detector | Sporadic in rural sites |

### OpenCV HOG (sandbox / fallback)

| FP cause | Why it happens |
|---|---|
| Vertical objects of similar aspect ratio (lampposts, narrow signs, tree trunks) | HOG's pretrained SVM was trained mostly on full-body upright humans |
| High-contrast vertical patterns | The gradient signature loosely matches a human silhouette |
| Anything tall and dark against a bright background | Contour edges fool the descriptor |

### What's mitigated in code

- **`confirm_frames=2`** in `VisionPerimeterEngine` — a single-frame YOLO
  fire on a poster won't propagate to an alert. This kills probably 70%
  of single-frame false positives.
- **Centroid tracker continuity** — a flickering one-frame detection
  doesn't get a stable track, so it never accumulates the consecutive
  inside-fence frames needed for an entry alert.
- **Foot-point semantics** — the fence test uses `(cx, y2)` (the bottom
  of the bounding box, where the person's feet would be) rather than the
  bbox center. This makes hanging signs and elevated reflections less
  likely to cross ground-plane fences.

### What's NOT mitigated

- **Persistent false objects** (a poster that's always in frame). These
  fire a track that never moves. Fix: a "static-object suppression list"
  per camera, populated by an operator review session in the first week
  of deployment.
- **Domain shift** — if YOLO was trained mostly on daylight outdoor
  scenes and you deploy on an indoor warehouse at night, accuracy
  degrades. Fix: fine-tune YOLO on ~500 labelled in-scene frames before
  go-live.

---

## ReID embedder (M50)

### Color-texture (sandbox)

The fundamental failure mode is **two people in similar clothing**.
Cosine similarity peaks at 0.9+ for identical clothing regardless of
who's wearing it. Examples:

- Event venue with uniformed staff
- Schools, hospitals, factories with dress codes
- Two visitors who happen to wear black coats in winter

### OSNet (production)

OSNet trained on Market-1501 / DukeMTMC handles uniform similarity
better — it learns face / body-shape cues — but still struggles with:

- **Top-down camera angles** (rooftop / ceiling cams). The training data
  is mostly waist-height; rooftop cameras get poor accuracy.
- **Severe lighting domain shift** between cameras (indoor fluorescent
  → outdoor sunlight). Same person, ~0.45 cosine similarity.

### What's mitigated in code

- **Three-zone ambiguity gate** in `CrossCameraTracker`:
  - sim ≥ 0.78 → match
  - sim < 0.55 → spawn new ID
  - in between → DEFER, no commit, retry next frame
  
  Defer-instead-of-guess is the right tradeoff in this domain because
  a wrong handoff is worse than a missed handoff (a wrong handoff
  attributes one person's path to another, which corrupts the audit
  log).

- **Per-camera embedding cache** — the matcher prefers same-camera
  embedding history when re-identifying within one camera, so lighting
  domain shift between cameras doesn't break local re-identification.

### What's NOT mitigated

- **Uniform collisions in the same camera.** The local centroid tracker
  catches this if motion is continuous, but not if two uniformed people
  cross paths. Fix: face-recognition module (out of scope for v1, would
  add `face_embedder.py` parallel to `reid_embedder.py`).
- **Cross-camera lighting.** Production fix is dual-domain training data
  for OSNet. We do not yet collect this.

---

## Audio anomaly (M49)

### Rule-based classifier

Each rule has a matched failure mode:

| Class | False-positive cause | What it sounds like |
|---|---|---|
| `gunshot` | Door slams, dropping heavy objects, fireworks | High crest factor + broadband |
| `glass_break` | Coins on tile, dropped silverware, cymbal hits | Saturated 2-8kHz + high ZCR |
| `breaking_door` | Bass drops in nearby music, vehicle backfire, HVAC compressor cycling | Low-band dominant + high peak |
| `scream` | Loud singing, child laughter, opera, sustained car horn | Harmonic 800-3500Hz |
| `alarm` | Music with strong tonal content, kettle whistle, smoke detector test | Very low spectral flatness |

Estimated baseline FP rate: **5-15 per camera per day** in a moderately
noisy environment (urban building, retail). Almost all are dismissable
within a 5-second clip review.

### CNN (when trained)

YAMNet-class CNNs trained on AudioSet generalize across most environments
but have their own pathologies:

- **Distribution shift** — a CNN trained on Western indoor audio fires
  oddly in different contexts (e.g. industrial floor tones interpreted
  as alarms).
- **Adversarial-ish audio** — looped music can trigger harmonic-pattern
  rules; very rare in legitimate operation but happens in retail.

### What's mitigated in code

- **Refractory window (2 s per class)** — one event spanning multiple
  chunks fires once, not five times.
- **Severity gating** — `gunshot` is severity 5, `alarm` is severity 3.
  Operators can filter the bus by severity to triage.
- **Dual-path fusion** (in `audio_cnn.py:DualPathAudioDetector`) — when
  the trained CNN is deployed, an alert fires only if rules and CNN
  agree, OR if one path crosses a high confidence threshold. This is the
  primary FP mitigation lever in production.

### What's NOT mitigated

- **Per-scene calibration** — the rule thresholds are global. A loud
  factory wants higher RMS thresholds; a quiet office wants lower. Fix:
  a calibration script that runs the extractor over 30 minutes of
  ambient audio per microphone and learns scene-specific thresholds.
- **Direction of arrival** — a single mic can't distinguish a glass
  break inside the perimeter from one in the parking lot. Fix: stereo
  microphone + cross-correlation, or two mics with TDOA. Out of scope
  for v1.

---

## Cross-modal correlation (not yet implemented)

The **highest-leverage** FP mitigation is correlating events across
modalities with time + space windows:

- A `SCREAM` audio alert *and* a `GEOFENCE_ENTRY` from a nearby camera
  within 3 seconds is far more credible than either alone.
- A `GEOFENCE_ENTRY` from the camera with no concurrent audio anomaly
  is more likely a person walking through, while one with a `BREAKING_DOOR`
  audio alert at the same moment is more likely a forced entry.

This belongs in M26 AIP (the orchestrator) — it gets all three sensor
streams already and is the right place to do confirmation logic. The
M26 implementation is not in this repo.

---

## Operational policy

The right way to live with the residual false-positive rate is:

1. **Tier the alerts.** Severity-5 wakes someone at night; severity-3
   queues for morning review. This is the bus filter in `integration.py`.
2. **Review weekly.** First two weeks of deployment, an operator reviews
   100% of severity-≥4 alerts and labels them TP / FP. The labels feed
   back into per-camera threshold tuning.
3. **Track the FP rate per detector**, not just overall. If `LOITERING`
   is firing 3x more than `GEOFENCE_ENTRY` and most are FP, the
   loitering threshold is wrong for this site, not the whole system.
4. **Measure the cost of misses** before tuning down. A single missed
   intrusion at a sensitive site can cost more than 1000 false alarms;
   a single missed gunshot can cost a life. Set the operating point
   accordingly.
5. **Don't optimize for zero FPs.** A perimeter system that never fires
   incorrectly is also a perimeter system that's missing too much. The
   right operating point is where review-cost-per-FP × FP-rate equals
   the marginal cost of the next missed detection.

This policy is what Verkada-class systems get right — they ship with FP
rates, not FP-free systems, and they invest in the operator review UX
because that's the actual product.
