# Contributing to Arbor

Thank you for your interest in contributing to Arbor! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see README.md)
4. Create a new branch for your feature or fix

## Development Setup

### Web Application

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev
```

### C++ Engine

```bash
cd cpp
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
```

## Code Style

### TypeScript/JavaScript

- Use TypeScript for all new code
- Follow existing code formatting (Prettier)
- Use meaningful variable and function names
- Add JSDoc comments for public APIs

### C++

- Follow the existing code style
- Use modern C++20 features where appropriate
- Include header guards or `#pragma once`
- Document public APIs with Doxygen-style comments
- Keep functions focused and reasonably sized

## Pull Request Process

1. **Create a branch** with a descriptive name:
   - `feature/add-new-indicator`
   - `fix/order-book-race-condition`
   - `docs/update-api-reference`

2. **Write clear commit messages**:
   ```
   feat: add RSI technical indicator
   
   - Implement 14-period RSI calculation
   - Add unit tests for edge cases
   - Update documentation
   ```

3. **Ensure tests pass**:
   - Run `pnpm lint` for the web application
   - Run `ctest` for C++ code

4. **Update documentation** if needed

5. **Submit the PR** with a clear description of changes

## Testing

### Web Application

```bash
pnpm lint
```

### C++ Engine

```bash
cd cpp/build
ctest --output-on-failure
```

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, compiler version, Node.js version)
- Relevant logs or error messages

## Feature Requests

Feature requests are welcome! Please:

- Check existing issues first
- Describe the use case
- Explain why this feature would be valuable
- Consider if you could implement it yourself

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Keep discussions on-topic

## Questions?

Feel free to open an issue for questions or discussions.
