# RiboSphereModel_RNA-

RiboSphere-based RNA structure reconstruction using a modular deep learning pipeline.

## Final Test Results

The trained RiboSphere model was evaluated on **20 test RNA structures** using **Kabsch-aligned RMSD** to measure the difference between the reconstructed and target RNA structures.

### RMSD Statistics

| Metric      |    RMSD (Å) |
| ----------- | ----------: |
| **Mean**    |  **5.3192** |
| **Median**  |  **2.3451** |
| **Minimum** |  **1.7243** |
| **Maximum** | **43.4237** |

### Test Summary

| Evaluation        |              Result |
| ----------------- | ------------------: |
| Test structures   |                  20 |
| Evaluation metric | Kabsch-aligned RMSD |
| Median RMSD       |        **2.3451 Å** |
| Mean RMSD         |        **5.3192 Å** |
| Minimum RMSD      |        **1.7243 Å** |
| Maximum RMSD      |       **43.4237 Å** |

## Results Overview

The test set shows a **median reconstruction RMSD of 2.3451 Å**, indicating that at least half of the evaluated structures have reconstruction errors at or below this value.

The mean RMSD is higher at **5.3192 Å**, reflecting the influence of a small number of structures with substantially larger reconstruction errors.

Overall, most test structures fall approximately within the **2–3 Å RMSD range**, while a small number of challenging structures produce significantly higher errors.

---

## Evaluation Metric

### Kabsch-Aligned RMSD

RMSD is calculated after optimal rotational and translational alignment using the **Kabsch algorithm**.

This alignment removes differences caused by the global orientation and position of the structures, allowing the evaluation to focus on the geometric difference between the reconstructed and target RNA coordinates.

The RMSD is reported in **Ångström (Å)**.

---

## Interpretation of the Initial Test

The results demonstrate that the trained model can reconstruct RNA structures with relatively low RMSD for a substantial portion of the test set.

However, the difference between the median and mean RMSD indicates that reconstruction quality is not uniform across all structures.

| Observation                          | Result        |
| ------------------------------------ | ------------- |
| Median reconstruction error          | **2.3451 Å**  |
| Mean reconstruction error            | **5.3192 Å**  |
| Lowest observed RMSD                 | **1.7243 Å**  |
| Highest observed RMSD                | **43.4237 Å** |
| Main error range for most structures | **~2–3 Å**    |

The high maximum RMSD indicates that some structures remain challenging for the current model and require further investigation.

---

## Result

<img width="1228" height="698" alt="image" src="https://github.com/user-attachments/assets/e1d124f8-ebe7-4c6d-8e99-043c3501c872" />


## Future Work

Future development will focus on:

* Improving reconstruction accuracy.
* Investigating high-RMSD structures.
* Evaluating performance across different RNA lengths.
* Improving the geometric representation.
* Training with larger and more diverse RNA datasets.
* Evaluating additional structural metrics alongside RMSD.
* Comparing subsequent model versions against this test baseline.

---

## Current Baseline

The current test experiment establishes the following baseline:

> **Median Kabsch-aligned RMSD: 2.3451 Å**

> **Mean Kabsch-aligned RMSD: 5.3192 Å**

These values can be used as reference points for subsequent RiboSphere model improvements.

---

## Evaluation Status

| Stage                | Status        |
| -------------------- | ------------- |
| Model training       | Completed     |
| Test evaluation      | Completed     |
| Test structures      | 20            |
| Kabsch alignment     | Applied       |
| RMSD evaluation      | Completed     |
| Baseline established | **Completed** |
| Further optimization | Planned       |

