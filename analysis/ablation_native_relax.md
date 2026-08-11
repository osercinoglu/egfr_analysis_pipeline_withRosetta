# C6 / C8 ablation — native treatment and relaxation

15 decoys per condition, `identity=composition, placement=inplace`, `chen_literal`, `exclude_fa_rep`, thresholds 0.78 / -1.0.

Conditions A and B share a decoy ensemble and differ only in the native reference, so C6's effect is isolated with zero sampling noise between them.

`frac_*` are over all superset pairs; `lig_frac_*` over ligand-incident pairs only.

| structure   | condition           |   wall_s_per_decoy |   frac_minimal |   frac_highly |   frac_neutral |   mean_F |   median_sigma |   lig_frac_minimal |   lig_frac_highly |   lig_frac_neutral |   lig_mean_F |   lig_median_sigma |
|:------------|:--------------------|-------------------:|---------------:|--------------:|---------------:|---------:|---------------:|-------------------:|------------------:|-------------------:|-------------:|-------------------:|
| 5GMP        | A prototype         |             4.8736 |         0.3089 |        0.0677 |         0.6234 |   0.3306 |         5.5003 |             0.5745 |            0.0000 |             0.4255 |       0.9197 |             5.8793 |
| 5GMP        | B +C6 native repack |             4.8736 |         0.3792 |        0.0098 |         0.6111 |   0.7928 |         5.5003 |             0.5106 |            0.0000 |             0.4894 |       0.8665 |             5.8793 |
| 5GMP        | C +C6+C8 mc         |            23.2463 |         0.3822 |        0.0100 |         0.6078 |   0.7966 |         5.4264 |             0.4468 |            0.0000 |             0.5532 |       0.8379 |             5.4051 |
| 1XKK        | A prototype         |             5.1137 |         0.2886 |        0.0280 |         0.6834 |   0.6039 |         6.8517 |             0.1231 |            0.0000 |             0.8769 |       0.4136 |             6.7102 |
| 1XKK        | B +C6 native repack |             5.1137 |         0.3175 |        0.0153 |         0.6672 |   0.6891 |         6.8517 |             0.2154 |            0.0000 |             0.7846 |       0.6021 |             6.7102 |
| 1XKK        | C +C6+C8 mc         |            24.8442 |         0.3245 |        0.0135 |         0.6620 |   0.7069 |         7.0036 |             0.2462 |            0.0000 |             0.7538 |       0.6080 |             7.1460 |
| 3POZ        | A prototype         |             5.8612 |         0.3711 |        0.0379 |         0.5910 |   0.7358 |         5.1516 |             0.2388 |            0.0000 |             0.7612 |       0.6171 |             5.2029 |
| 3POZ        | B +C6 native repack |             5.8612 |         0.3920 |        0.0116 |         0.5964 |   0.8479 |         5.1516 |             0.1493 |            0.0000 |             0.8507 |       0.5521 |             5.2029 |
| 3POZ        | C +C6+C8 mc         |            23.3055 |         0.3818 |        0.0107 |         0.6075 |   0.8560 |         5.1882 |             0.2537 |            0.0000 |             0.7463 |       0.7098 |             4.5782 |
