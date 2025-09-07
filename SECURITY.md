# Foresight SAR Security Guidelines

## Overview

This document outlines security practices for the Foresight SAR application, with particular focus on secrets management, evidence integrity, and operational security.

## Secrets Management

### Environment Variables

**NEVER commit sensitive data to the repository.** All sensitive configuration must be managed through environment variables or external secret management systems.

#### Required Environment Variables

Copy `.env.example` to `.env` and configure the following:

```bash
cp .env.example .env
```

**Critical Variables:**
- `VAULT_ADDR` - HashiCorp Vault server address
- `VAULT_TOKEN` - Vault authentication token
- `DATABASE_URL` - Database connection string
- `REDIS_URL` - Redis connection string
- `EVIDENCE_PRIVATE_KEY_PATH` - Path to evidence signing key
- `KEY_PASSPHRASE` - Passphrase for private keys

### HashiCorp Vault Integration

The application uses HashiCorp Vault for:
- **Transit Engine**: Cryptographic signing of evidence packages
- **KV Store**: Secure storage of configuration secrets
- **PKI**: Certificate management for TLS

#### Vault Setup

1. **Enable Transit Engine:**
   ```bash
   vault auth -method=userpass username=admin
   vault secrets enable transit
   vault write -f transit/keys/evidence-signing
   ```

2. **Create Policies:**
   ```bash
   vault policy write foresight-sar - <<EOF
   path "transit/sign/evidence-signing" {
     capabilities = ["update"]
   }
   path "transit/verify/evidence-signing" {
     capabilities = ["update"]
   }
   path "kv/data/foresight/*" {
     capabilities = ["read"]
   }
   EOF
   ```

3. **Generate Application Token:**
   ```bash
   vault token create -policy=foresight-sar -ttl=24h
   ```

### GitHub Secrets

For CI/CD operations, configure these GitHub Secrets:

#### Required Secrets
- `VAULT_ADDR` - Vault server URL
- `VAULT_TOKEN` - CI/CD service token
- `POSTGRES_PASSWORD` - Test database password
- `CODECOV_TOKEN` - Code coverage reporting
- `STAGING_HOST` - Staging server hostname
- `STAGING_USER` - Staging deployment user
- `STAGING_SSH_KEY` - SSH private key for staging
- `PRODUCTION_HOST` - Production server hostname
- `PRODUCTION_USER` - Production deployment user
- `PRODUCTION_SSH_KEY` - SSH private key for production
- `SLACK_WEBHOOK_URL` - Slack notifications (optional)
- `TEAMS_WEBHOOK_URL` - Teams notifications (optional)

#### Setting GitHub Secrets

1. Navigate to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each required secret with appropriate values

## Evidence Integrity

### Digital Signatures

All evidence packages are cryptographically signed using:
- **Primary**: HashiCorp Vault Transit engine
- **Fallback**: Local private key with passphrase

### OpenTimestamps

Evidence packages are timestamped using OpenTimestamps for:
- **Proof of existence** at specific time
- **Tamper detection** through blockchain anchoring
- **Legal admissibility** in court proceedings

#### Timestamp Verification

```bash
# Verify timestamp
ots verify evidence-package.zip.ots

# Upgrade timestamp (after blockchain confirmation)
ots upgrade evidence-package.zip.ots
```

## Development Security

### Pre-commit Checks

Install pre-commit hooks to prevent accidental secret commits:

```bash
pip install pre-commit
pre-commit install
```

### Secret Scanning

The CI/CD pipeline includes:
- **TruffleHog**: Scans for committed secrets
- **Trivy**: Vulnerability scanning
- **SARIF**: Security findings integration

### Code Review Requirements

- All changes require review before merge
- Security-sensitive changes require additional review
- Never approve PRs containing hardcoded secrets

## Operational Security

### Access Control

- **Principle of least privilege** for all system access
- **Multi-factor authentication** required for production
- **Regular access reviews** and deprovisioning

### Logging and Monitoring

- All security events are logged
- Failed authentication attempts trigger alerts
- Evidence access is audited and tracked

### Incident Response

1. **Immediate**: Revoke compromised credentials
2. **Assessment**: Determine scope of compromise
3. **Containment**: Isolate affected systems
4. **Recovery**: Restore from clean backups
5. **Lessons Learned**: Update security measures

## Deployment Security

### Environment Separation

- **Development**: Local environment with mock data
- **Staging**: Production-like environment for testing
- **Production**: Live operational environment

### Secure Deployment

- All deployments use signed artifacts
- Production deployments require manual approval
- Rollback procedures are tested and documented

### Network Security

- TLS encryption for all communications
- VPN access required for production systems
- Firewall rules restrict unnecessary access

## Compliance

### Data Protection

- Personal data is encrypted at rest and in transit
- Data retention policies are enforced
- Right to erasure is supported where applicable

### Evidence Chain of Custody

- All evidence handling is logged
- Digital signatures ensure integrity
- Timestamps provide proof of existence
- Access controls prevent unauthorized modification

## Security Contacts

- **Security Team**: security@foresight-sar.com
- **Incident Response**: incident@foresight-sar.com
- **Vulnerability Reports**: security@foresight-sar.com

## Regular Security Tasks

### Weekly
- Review access logs
- Check for security updates
- Verify backup integrity

### Monthly
- Rotate service credentials
- Review user access permissions
- Update security documentation

### Quarterly
- Conduct security assessments
- Review and test incident response procedures
- Update threat models

## Security Tools

### Required Tools
- **HashiCorp Vault**: Secret management
- **OpenTimestamps**: Evidence timestamping
- **TruffleHog**: Secret scanning
- **Trivy**: Vulnerability scanning

### Recommended Tools
- **SOPS**: File encryption
- **Age**: Modern encryption tool
- **Cosign**: Container signing
- **Sigstore**: Software supply chain security

## Emergency Procedures

### Credential Compromise

1. **Immediately revoke** the compromised credential
2. **Generate new credentials** with different values
3. **Update all systems** that use the credential
4. **Audit logs** for unauthorized access
5. **Document the incident** for future reference

### System Compromise

1. **Isolate** the compromised system
2. **Preserve evidence** for forensic analysis
3. **Notify stakeholders** according to incident response plan
4. **Restore from clean backups** after threat removal
5. **Conduct post-incident review** and update security measures

---

**Remember**: Security is everyone's responsibility. When in doubt, ask the security team.