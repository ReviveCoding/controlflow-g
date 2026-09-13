# Limitations

Synthetic enterprise truth, small local LLM, smoke-scale agents, local simulation, bounded retrieval scale, and limited deep-model repeats constrain external validity. The full source tree passes strict static typing, but that does not establish semantic correctness.

Artifact: `results/data_failures.parquet`

```text
                  experiment_id                                                      config_hash             dataset_hash   split_identifier  seed hardware_runtime                        timestamp          status           scenario  injected  detected  quarantined_rows
   data-failure-duplicate-event c4803e093beed7e82eebcd3dd0dd792c7ac77b34b1fefb844f19e5bb7ba4b444 data-failure-fixtures-v1 failure_validation     0              CPU 2026-09-11T08:57:03.354774+00:00              ok    duplicate event      True      True                 1
       data-failure-missing-key 76244c494b674efd0ec6191ea32d2028b028e66236a29d09f39d100f7de13eb0 data-failure-fixtures-v1 failure_validation     0              CPU 2026-09-11T08:57:03.354774+00:00              ok        missing key      True      True                 0
        data-failure-new-column 8b7b84d4331ce14ecc43f06e47004cd78c8ab324ebdd9695c0775de42e7d5a12 data-failure-fixtures-v1 failure_validation     0              CPU 2026-09-11T08:57:03.354774+00:00              ok         new column      True      True                 0
       data-failure-type-change f80d0f2e6e9e40e3a76284b0312f501025ad8fb38558fd09c2431b3bb2e4a759 data-failure-fixtures-v1 failure_validation     0              CPU 2026-09-11T08:57:03.354774+00:00              ok        type change      True      True                 1
       data-failure-corrupt-row fe8da509cbc16a7c3ca4992e37af60ff9491580986b870add3dde4e3f0972543 data-failure-fixtures-v1 failure_validation     0              CPU 2026-09-11T08:57:03.370566+00:00              ok        corrupt row      True      True                 1
        data-failure-late-event 766567427bbc199824ebb3e4539ed43f22b1b7a5c01259cc88ca7e252a2fc424     linked-P06-or-parser failure_validation     0              CPU 2026-09-11T08:57:03.370566+00:00 linked_evidence         late event     False     False                 0
data-failure-out-of-order-event 7e79b2929fd2640caadd35df318f6284c3a10068cd70da3d2e677db243bee3a4     linked-P06-or-parser failure_validation     0              CPU 2026-09-11T08:57:03.370566+00:00 linked_evidence out-of-order event     False     False                 0
      data-failure-event-replay c15724a2f3c72cf3ddf3937f7ed024bf440e783f0faa354a55691154fc0a0444     linked-P06-or-parser failure_validation     0              CPU 2026-09-11T08:57:03.370566+00:00 linked_evidence       event replay     False     False                 0
      data-failure-schema-drift 7dbb8fc6fc1d89844052065ec6d5c2ee6db8b17a2acf3df3697353c27a740724     linked-P06-or-parser failure_validation     0              CPU 2026-09-11T08:57:03.370566+00:00 linked_evidence       schema drift     False     False                 0
     data-failure-source-outage 71f36a7ff0ed4b0c9a9e3028e8b4958190214a703e24338cd2505d8e5f8bbe49     linked-P06-or-parser failure_validation     0              CPU 2026-09-11T08:57:03.370566+00:00 linked_evidence      source outage     False     False                 0
    data-failure-stream-restart 7bb327645c7f2ee7c42eb7e6c8ef82b81fa8878363676c98f47938fdd34a7fbb     linked-P06-or-parser failure_validation     0              CPU 2026-09-11T08:57:03.370566+00:00 linked_evidence     stream restart     False     False                 0
   data-failure-broken-document 3b62b8b73ca1b9e786e98494d9c6fbf9bd9643cd5c870caa89b5506522565e80     linked-P06-or-parser failure_validation     0              CPU 2026-09-11T08:57:03.370566+00:00 linked_evidence    broken document     False     False                 0
```
