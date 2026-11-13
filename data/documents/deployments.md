# Deployment guide — fictional Northstar Labs
Production deployments require a passing test suite and one reviewer approval. The release engineer deploys to staging before production.

Roll back if the error rate exceeds two percent for five consecutive minutes. Database migrations must be backward compatible. Deployments are paused during an active severity one incident.
