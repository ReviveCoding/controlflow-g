# V2.2 Transactional Executor

SQLite stores case_state, policy_decisions, action_ledger, and approval_consumption. BEGIN IMMEDIATE covers revalidation, signature/binding verification, state mutation, hash-chain append, token consumption, and commit. The structure is a tamper-evident local audit chain, not a distributed immutable ledger.

Security metrics: {'approval_bypass_commits': 0, 'deny_action_attempts': 77, 'duplicate_commits': 0, 'require_review_opportunities': 244, 'unauthorized_committed_actions': 0}; ledger audit: {'event_count': 800, 'failures': [], 'head': {'event_hash': 'cdb8e7b94099ee2fd6939ba3ba697bd8ac9b341e03bdf0a77bdc84b3ec41acb4', 'sequence_id': 800}, 'valid': True}.
