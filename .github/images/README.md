# Arbor - Production-Grade Trading Infrastructure

This directory contains logos and images for the Arbor project README and documentation.

## Logo Files

- `arbor-dark.svg` — Logo for light theme backgrounds
- `arbor-light.svg` — Logo for dark theme backgrounds

You can add your own SVG logos here or use placeholder images.

## Using in Documentation

Reference logos in markdown:

```markdown
<picture>
  <source media="(prefers-color-scheme: light)" srcset=".github/images/arbor-dark.svg">
  <source media="(prefers-color-scheme: dark)" srcset=".github/images/arbor-light.svg">
  <img alt="Arbor Logo" src=".github/images/arbor-dark.svg" width="80%">
</picture>
```

This provides theme-aware logos that adapt to user's system preferences.
