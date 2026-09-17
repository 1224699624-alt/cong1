# R244 GT-Free Merge Filter Progress

R244 was the original-val candidate-generation/oracle audit for TSRS Epiphysis. It is preserved
as mechanism evidence, not as a deployable GT-free selector.

## Oracle result

- 87 oracle images; 87/87 safe.
- Boundary IoU delta: `+0.001281313`.
- Boundary F1 delta: `+0.001695222`.
- gap-region FP delta: `-0.001125485`.
- component count MAE delta: `-0.563218391`.
- Recall delta: `0`.
- cut GT foreground fraction: `0`.

The result shows that correctly selected local cuts can reduce false bridges without cutting true
bone foreground. The GT-free candidate selector was not accepted as a final method because the
full candidate pool contained too many risky cuts. R245/R246/R248 retained the failure-audit
evidence; no clean-test-v2 tuning or model selection was performed.
