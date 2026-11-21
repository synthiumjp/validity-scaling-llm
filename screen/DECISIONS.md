# validity-screen v0.4: decision record

*Corrections made in 0.4.1 after an independent review are marked. Decision rules are unchanged.*

Each decision states what changes, the clinical rationale (how validity assessment in psychological testing handles the same problem), the statistical rationale, and its effect on the paper's results. The clinical analogy is procedural: it borrows the order and logic of validity assessment, not the scales themselves.

The screen classifies a binary confidence signal from four cells: correct and kept (a), wrong and kept (b), correct and withdrawn (c), wrong and withdrawn (d).
- Kept-error rate L = b/(b+d).
- Withdrawn-correct rate Fp = c/(a+c).
- Item sensitivity J = (1 − Fp) − L (Youden's J).

---

## D1. Invalid means a fixed response set, judged the same way in both directions

**Change.** Invalid if L ≥ .95 or Fp ≥ .95, each certified by its interval (Wilson lower bound > .90). v0.3 called a model Invalid at Fp ≥ .50.

**Clinical rationale.**
- A validity indicator asks one question: does the response style leave the content scales interpretable?
- TRIN (MMPI-3; TRIN-r in the MMPI-2-RF) detects fixed responding in either direction, acquiescence (all True) and non-acquiescence (all False), against the same cut. Neither direction is privileged.
- Blanket confidence is acquiescence: KEEP on everything. Blanket withdrawal is nay-saying: WITHDRAW on everything. They should be judged by the same standard.
- The v0.3 rule invalidated a model for withdrawing half its correct answers. That is the equivalent of invalidating a cautious respondent who still answers in line with item content. Clinically we would not do that.

**Statistical rationale.**
- At L ≥ .95 or Fp ≥ .95 the *observed* J is at most .05, so the threshold means the same thing in both directions. *Corrected 0.4.1:* certification establishes only that the true rate exceeds .90, so the population bound on J is .10, and an Invalid signal can still show a small but significant J (two cells in the held-out study do).
- Fp ≥ .50 bounds J only at .50. That is why v0.3 called strong signals Invalid, for example Haiku at cut 95 (J = .33, AUROC .77).

**Effect.**
- DeepSeek-R1 changes from Invalid to Uncertifiable. It has 6 errors, and its Fp of .946 is just below .95. It also carries the low-coverage flag (D2).
- Strong signals at conservative cut points (Haiku at 95, Qwen 80B Thinking at 100, GPT-4o on ARC-Easy at 100) become Valid.
- The other 19 battery verdicts are unchanged.

## D2. Withdrawing many correct answers is a caution flag, not invalidity

**Change.** Fp ≥ .50 raises a "low coverage" flag and is reported beside the tier.

**Clinical rationale.** MMPI-3 interpretation separates "invalid: do not interpret" from "interpret with caution", and moderate under-reporting gets the caution, not invalidation. A model that withdraws half its correct answers may still discriminate well; whether it is usable depends on how much coverage a deployment needs. That is a utility judgement, separate from validity.

**Statistical rationale.** Coverage and discrimination are different quantities. Merging them is what made the v0.3 tiers out of order with respect to J.

## D3. Too few scorable responses means no verdict (Uncertifiable), and invalidity must be certain

**Change.** Unchanged in substance. Fewer than 10 errors or 10 correct answers gives Uncertifiable. A response set counts as established only if its interval excludes the non-invalid range.

**Rationale (corrected 0.4.1).** This rule is statistical, not clinical. With fewer than 10 events per denominator, rates are unstable and intervals uninformative, and the Wilson guard stops a premature Invalid.
- *Withdrawn analogy 1:* the draft linked Uncertifiable to Cannot Say. That was wrong: Cannot Say counts unanswered items, which is Step 1 (exclusions), whereas Uncertifiable is triggered by too few *errors*, a property of accuracy in a protocol where every item was answered.
- *Withdrawn analogy 2:* the draft said clinicians do not report a validity elevation whose error band reaches the normal range. There is no such manual practice; validity cut scores are applied to the T score.

## D4. Valid requires evidence that confidence follows content

**Change.** Valid requires J > 0 by a one-sided test at .025, after D1 to D3. Otherwise the verdict is Indeterminate, reason "no detectable item sensitivity".

**Clinical rationale.**
- A protocol is interpretable only if responses are driven by item content.
- In personality inventories, content non-responsiveness is detected indirectly, through inconsistency on paired items (MMPI-3 VRIN and CRIN). For a confidence signal it can be tested directly, because correctness is known.
- This is a procedural analogue, not the same mechanism. The paper will say so, and VRIN will no longer be called a criterion test.

**Statistical rationale.** Absence of a response set does not imply presence of a signal. A model can avoid both blanket patterns and still keep and withdraw at random.

## D5. The test behind "Valid" must hold its stated error rate (mid-p instead of Newcombe)

**Change.** The J > 0 test becomes Fisher's exact test with the mid-p correction (one-sided, .025). The Newcombe interval is still reported for J.

**Rationale (corrected 0.4.1).**
- *Statistical (unchanged):* the test behind "Valid" should hold its stated level. It is a significance test, and a significance test should keep its false-positive rate at the level it states.
- *Clinical (corrected):* validity indicators are accepted on the strength of known error rates (Larrabee, 2012).
  - The performance-validity convention of keeping false positives at or below about 10% (Boone, 2013) refers to genuine examinees wrongly called *invalid*.
  - The screen's analogue of that error is false Invalid: certifying a response set that is not one. It is reported separately, and it is not small near the threshold: a true kept-error rate of .93 is certified up to about 23% of the time.
  - The draft wrongly attached the 10% convention to false Valid.

**Statistical rationale.** Exact enumeration under no signal (452 items, 10 to 100 errors, KEEP rates .50 to .99) gives these false-Valid rates against a nominal .025 (*0.4.1 note:* over a full grid of keep rates down to .02 the mid-p test reaches about .030 with only 10 to 12 errors at keep rates of .50 or more, and about .037 at keep rates of .10 to .30; see the operating characteristics):

| Test | False-Valid rate |
|---|---|
| Newcombe interval | up to .088 (3.5 times nominal) |
| Fisher exact | .002 to .019 (holds the level, but too cautious, costing power) |
| Mid-p | .003 to .024 (stays at or below the nominal level in every setting tested) |

Mid-p is the standard compromise for discrete tests (Lancaster, 1961; Agresti, 2013). It will be checked again in the operating-characteristics analysis across item counts and accuracies.

**Effect.** Small. Battery verdicts are re-checked in the rerun.

## D6. Every Indeterminate says why, and a Valid that cannot exclude a response set says so

**Change.**
- Indeterminate carries a reason code: response set suspected but not certified, inversion suspected but not certified, or no detectable sensitivity.
- A Valid verdict whose L or Fp interval reaches .95 carries a flag: "response set not excluded".
- Figures report the three kinds of Indeterminate separately.

**Clinical rationale.** A validity report says why a protocol could not be interpreted, and caution bands sit beside clear results.

**Statistical rationale.** The guard otherwise works in one direction only. Gemini 2.5 Flash and Pro are Valid with L intervals reaching .953 and .965.

## D7. One cut point chosen in advance; the others are sensitivity analyses

**Change.**
- For 0–100 confidence, the tool has no default cut point; the user must state one (consistent with D8).
- The guidance tells users to choose it before seeing results, from the deployment threshold they actually intend to use.
- This paper's analyses were not pre-specified. It will report all six cut points descriptively, name none as primary, and report the family-wise false-Valid rate.
- "Never Valid at any cut point" summaries are dropped.

**Clinical rationale (corrected 0.4.1).** Whether adding validity indicators inflates false positives has been argued (Berthelson et al., 2013; Bilder, Sugar and Hellemann, 2014) and disputed (Davis and Millis, 2014; Larrabee, 2014); the draft cited the rebuttals as support. The statistical point does not depend on that debate: trying several cut points until one gives the wanted verdict inflates the false-Valid rate, which we measure directly (6.3% on average across six cuts against 1.3% at one).

**Statistical rationale.** In the atlas, a null signal is Valid at one or more of six cut points in 7% of permutations on average, and up to 13%.

**Why no default.** A default would be a choice made for the user after seeing these data. Clinically, a cut score is set for the referral question and the population; here it is set by the deployment threshold, which only the user knows.

## D8. Missing or malformed responses are excluded and counted, never guessed

**Change.** `screen()` rejects non-binary confidence and missing values. `binarise()` requires an explicit cut point. The CLI routes unparseable cells to the exclusion count and reports them.

**Clinical rationale.** An unanswered or double-marked item counts toward Cannot Say. It is never scored as True or False on the examinee's behalf. Step 1 of the screen already applies this to model output; the tool must apply it to its own input.

**Statistical rationale.** v0.3 coerced NaN to high confidence and treated raw 0–100 values as binary. Both produce a certified "blanket confidence" verdict from a data error.

---

## Open decision for you

**D9. Index names.**
- "L", "Fp" and "RBS" are borrowed MMPI scale names that mean something different in the MMPI: Lie, Infrequent Psychopathology, Response Bias Scale.
- Clinical reviewers will read them as claims of equivalence, and RBS is the plainest mismatch.
- Proposal: in the paper, use plain names (kept-error rate, withdrawn-correct rate, Youden's J) and mention the clinical analogues once. Keep `L` and `Fp` as attribute names in the package for backward compatibility. Drop "RBS" as a name.
