# V2.2 Calibration and Critical Threshold

Calibration and threshold selection use CALIBRATION only. The predeclared rule retains thresholds with critical recall at least 0.95, then minimizes FPR plus 1.5 times residual critical risk and a penalty for falling below a 0.98 development safety margin; ties favor the higher threshold.

Selected calibrator: isotonic; threshold: 0.08333333333333333; provenance: {'rule': 'CALIBRATION recall >= 0.95 then minimize FPR + 1.5*residual-critical-risk + 10*max(0,0.98-recall); break ties by highest threshold', 'split': 'CALIBRATION'}.
