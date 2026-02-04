# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in Arbor, please open a private security advisory on GitHub or create a security issue with details:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

**Do not** disclose security vulnerabilities publicly until coordinated. We will work with you on a fix and responsible disclosure timeline.

## Security Considerations

### Trust Model

Arbor operates on a **"trust the operator"** model:

- The platform executes whatever orders your code submits
- The C++ engine does not validate trading strategies or risk decisions
- **You are responsible** for implementing proper controls

### Recommended Safeguards

#### Pre-Trade Risk Management
- Implement position limits before submitting orders
- Use the built-in `Risk.checkOrder()` function
- Validate order sizes and prices
- Monitor daily loss limits

#### Environment Security
- Never commit `.env.local` or API keys (already in `.gitignore`)
- Use separate API keys for development and production
- Rotate credentials regularly
- Restrict database access by IP

#### Code Security
- Review all custom strategies before deployment
- Sanitize external inputs (market data, user input)
- Use TypeScript for type safety
- Implement comprehensive logging

#### Operational Security
- Run on sandboxed or isolated systems
- Monitor for unusual trading patterns
- Implement kill switches for runaway algorithms
- Keep C++ libraries and dependencies updated

### Known Limitations

1. **No Built-in Rate Limiting** — Implement at the broker/API level
2. **No Persistent Encryption** — Supabase handles database encryption
3. **No Audit Logging** — Add to your deployment for compliance
4. **Test First** — Always backtest strategies on historical data before live trading

## Dependency Security

Arbor uses well-maintained open-source dependencies:

### Node.js/TypeScript
- **Next.js** — Actively maintained by Vercel
- **React** — Widely used and regularly audited
- **TypeScript** — Static typing catches many bugs

### C++
- **Google Test** — Industry standard testing framework
- **Standard Library** — Uses C++20 standard library primitives
- **No external C++ dependencies** — Core engine is self-contained

Run `npm audit` and check for updates regularly:

```bash
npm audit fix
pnpm update
```

## Compliance

Arbor is provided as-is for educational and research purposes. If you deploy for actual trading:

- Verify compliance with local financial regulations
- Implement required audit logging
- Set up monitoring and alerting
- Use under proper market supervision
- Consider liability insurance

## Further Reading

- [OWASP Security Principles](https://owasp.org/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

**Questions?** Open a GitHub Discussion or Issue for security-related questions.
