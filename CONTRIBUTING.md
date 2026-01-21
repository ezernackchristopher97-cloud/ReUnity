# Contributing to ReUnity

Thank you for your interest in contributing to ReUnity! This document provides guidelines and information for contributors.

## ⚠️ Important Disclaimer

Before contributing, please understand that ReUnity is **NOT a clinical or treatment tool**. It is a theoretical and support framework only. All contributions must maintain this distinction and include appropriate disclaimers.

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Special Considerations

Given the sensitive nature of this project (trauma-aware AI), we ask contributors to:

- Be mindful that users may have trauma histories
- Avoid language that could be triggering
- Prioritize user safety in all design decisions
- Respect privacy and consent in all features

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)

### Suggesting Features

1. Check existing issues and discussions
2. Create a new issue with:
   - Clear description of the feature
   - Use case and motivation
   - Potential implementation approach
   - Safety considerations (if applicable)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation as needed
7. Commit with clear messages
8. Push to your fork
9. Open a Pull Request

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add entropy visualization component
fix: correct JS divergence calculation for edge cases
docs: update API documentation for memory endpoints
test: add tests for pattern recognition module
```

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/reunity.git
cd reunity

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install pytest black ruff mypy

# Run tests
pytest

# Check formatting
black --check src tests
ruff check src tests
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Maximum line length: 88 characters (Black default)
- Use docstrings for all public functions and classes
- Include disclaimers in module docstrings

### Example Function

```python
def analyze_entropy(
    distribution: np.ndarray,
    normalize: bool = True,
) -> EntropyMetrics:
    """
    Analyze entropy of a probability distribution.

    DISCLAIMER: This is not a clinical tool.

    Args:
        distribution: Probability distribution to analyze.
        normalize: Whether to normalize the result.

    Returns:
        EntropyMetrics containing analysis results.

    Raises:
        ValueError: If distribution is invalid.
    """
    ...
```

## Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use descriptive test names
- Include edge cases and error conditions

```python
class TestEntropyAnalyzer:
    """Tests for entropy analysis."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution should have maximum entropy."""
        ...

    def test_invalid_distribution_raises_error(self):
        """Invalid distribution should raise ValueError."""
        ...
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public APIs
- Include examples in documentation
- Keep the disclaimer visible and clear

## Safety Considerations

When contributing to ReUnity, always consider:

1. **User Safety**: Could this change harm a vulnerable user?
2. **Privacy**: Does this respect user consent and data sovereignty?
3. **Clarity**: Is it clear this is not clinical treatment?
4. **Accessibility**: Is this accessible to users in various states?

## Questions?

Feel free to open an issue for any questions about contributing.

---

*Remember: This tool is meant to support, not replace, professional mental health care.*
