# V2.2 Signed Approvals

The separate ApprovalIssuer holds an ignored local Ed25519 private key. The bundle and PEP bind only the public key. Tokens bind case, action, policy decision, authorization snapshot, versions, reviewer, expiry, and nonce; tampering and replay are negative-tested.
