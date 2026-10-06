# Reviewer letter (round 2, MDPI submission "A Hybrid AI Architecture...")

Received via paste on 2026-10-02. Full verification in REVIEWER_CHECK.md; revision
workspace at ../MDPI_Hybrid_Revision/ (extracted from MDPI_Submission.zip).

The architecture is clearly specified and the paper is candid about its limits. However, only the learned branch is tested, the synthetic test rests on two patients, and the one real-data test is negative in a setting the system was not built for. I would hold submission until points 1 to 5 are addressed.

Strengths
- Candid scope. The paper calls itself a research architecture, not a validated system (abstract, Sections 4.5 and 6), and reports the failed Holter transfer in full.
- Clear decision authority. Evidence generation, the decision agent and delivery are separated (Table 1). Language models cannot raise or suppress a warning. Each warning is stored before delivery, so bedside and remote users see the same record (Section 3.11). Figure 1 shows this layering clearly.
- Patient-level evaluation design. The paper uses patient-level splits, grouped cross-validation, chronological replay with a refractory interval, and alert-level measures such as lead time and alerts per patient-day. These and the labelling rules in Section 3.3 suit the problem.
- Shortcut checks. The identity probes and ablations in Section 4.2 test whether the embeddings encode the patient rather than the deterioration. They place the 1 h signal in the engineered HRV and ectopy features.
- Specification detail. The algorithms, equations and explicit rules in Table 2 would let someone rebuild the pipeline once the missing settings in point 7 are added.
- Related work organised by task. Section 2 separates prediction from detection (2.4) and notes that AUROC must be read with precision when events are rare (2.2).

Improvements (in priority order; points 1 to 5 are the substantive ones)
1. Test the architecture the title claims. Only the learned branch is evaluated (Section 3.12). Extend the generator to produce every NEWS2 input and replay the full decision agent, comparing alerts per patient-day for learned-only, rule-only and hybrid setups. Section 3.10 says all active triggers are combined, but Algorithm 8 adds only the highest-priority rule. Also, a rule that keeps firing with a 180 s cooldown can alert about 20 times an hour, which undercuts the alarm-burden claim in Section 5.1.
2. The synthetic signal looks like a patient-level shortcut. The drift covers only the final 6 h (Section 3.2). Yet the case patient's 6 h risk passes 0.5 at 21.7 h before arrest and stays high (Figure 6). The replay also gives 0.946 at 24 h against 0.667 on validation (Table 3).
   - Check the generator for differences between case and control patients outside the drift, and confirm patients 37 and 52 were not in training.
   - An alert 21.7 h out is a false alert for a 6 h horizon.
   - The 6 h Brier score (0.315) is worse than a constant forecast at the observed prevalence, which scores at most about 0.19 here.
   - So 0.891 and 21.7 h should not headline the abstract.
3. The replay is too small, and its numbers disagree.
   - Table 4 reports intervals that Section 4.4 says two patients cannot support. With two patients, Algorithm 11 returns only two distinct values, so a point estimate cannot sit strictly inside its interval as 0.891 does.
   - Direct concatenation scores 0.769 to 0.859 on validation (Section 4.1, Tables 3 and 6) and 0.891 or 0.929 on the replay (Tables 4 and 8). That spread is larger than any effect in Table 8.
   - Table 6's 247 case alerts exceed what a 60 min refractory interval allows.
   - Table 4's positive counts (12, 72, 288) suggest the 15 min exclusion was not applied.
   - Fix: generate a few hundred synthetic patients, hold out a proper test set, run several seeds, and regenerate every table from one run.
4. Give the Holter analysis denominators and explain the inversion.
   - Records average about 7.6 h, so 6 h negatives come only from records with more than 6 h before VF. Report positives, negatives and contributing patients per horizon.
   - Several 6 h AUROCs fall below 0.5 with intervals that exclude 0.5 (Table 7). That needs a cause; pooling scores across folds is one candidate.
   - The identity probe should be scored on held-out windows, with the majority-class rate as chance, not 0.05.
   - Table 7 lacks a logistic regression or gradient boosting baseline, and both beat FEAN in Table 3.
   - The MIMIC-III analysis planned in Section 5.2 is the test the design needs. I would complete it before submitting.
5. State the clinical setting and ground the rule branch. The paper points to three settings at once. Arterial pressure waveforms suggest intensive care, NEWS2 and the ward studies in Section 2.2 suggest general wards, and medicine adherence, the obesity and cholesterol multiplier and Telegram suggest remote monitoring. The setting determines the inputs, event rate and comparator.
   - NEWS2 needs a consciousness level, and the paper does not say where it comes from.
   - The weights, thresholds and multiplier in Section 3.9 and Table 2 have no source, and the multiplier raises alert rates for one subgroup by design.
   - Four of the six components need ECG or PPG, so the branch is not the fallback for missing waveforms that Section 3.9 describes.
6. Define the evidence record. The "leading drivers", the confidence values and the time-to-event normalisation are never defined. The time-to-event head is trained only where an arrest follows, so it ignores censoring, yet its output appears in every warning. Describe and test the language-model explanation service, or remove it. Also define "agent": the components in Table 1 are deterministic services, and AI reviewers are likely to read them as LLM agents.
7. Reproducibility. Report the generator parameters, split sizes, the seven vital-sign features, model hyperparameters, stopping rule, seeds and number of runs, and release the code with a DOI. Compute HRV over the full 5 min window rather than one 30 s segment, as in the 1996 ESC/NASPE Task Force standard; these features carry the only real-data signal.
8. Governance of remote alerts. Telegram's standard cloud chats are not end-to-end encrypted. Describe how recipients are authenticated, what patient data leaves the hospital network and how delivery is audited, or present Telegram as a prototype channel only.
9. References.
   - Several sources are used without citation: NEWS2 (Royal College of Physicians, 2017), the Sudden Cardiac Death Holter Database and PhysioNet, MIMIC-III, the Ext CA annotation set, PCGrad (Yu et al., NeurIPS 2020) and the synthetic ECG beat model.
   - [30] names Hu as first author, but SiamQuality's first author is Cheng Ding.
   - [28] and [32] were published at ICLR 2025 and ICLR 2024; cite those versions.
   - [32] found that its wearable PPG and ECG models capture participants' demographic and health information, which supports the identity finding in Section 4.2.
10. Presentation.
   - Hyphens have been removed throughout, which breaks names (ECG-FM, PaPaGei-S, MIMIC-III, Martinez-Alanis) and compounds (time-to-event, in-hospital). Avoiding em dashes does not require this.
   - Spelling mixes British and American forms (behaviour and multicentre against generalize and labeling).
   - Move the last paragraph of Section 3.7 to the Discussion.
   - Drop Figures 12 and 13, which repeat Table 8, and consider moving the slow-risk layer and robustness sweep to supplementary material.
   - Add the MDPI template and back-matter statements.
