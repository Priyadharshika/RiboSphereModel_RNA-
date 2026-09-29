# RiboSphereModel_RNA-

## Final Test Results

The trained RiboSphere model was evaluated on 20 test RNA structures using **Kabsch-aligned RMSD**.

### RMSD Statistics

| Metric  | RMSD (Å) |
| ------- | -------: |
| Mean    |   5.3192 |
| Median  |   2.3451 |
| Minimum |   1.7243 |
| Maximum |  43.4237 |

Most test structures show RMSD values in the **~2–3 Å range**. A small number of structures have substantially higher reconstruction errors, including RNA 18 (14.08 Å) and RNA 20 (43.42 Å), which contribute strongly to the higher mean RMSD.

**Evaluation metric:** Kabsch-aligned RMSD

**Test set:** 20 RNA structures
