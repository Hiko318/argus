# Contributing to Foresight SAR System

Thank you for your interest in contributing to the Foresight Search and Rescue (SAR) System! This document provides guidelines for contributing to this project.

## Code of Conduct

This project is designed for search and rescue operations. All contributions must:
- Respect privacy and data protection laws
- Be intended for legitimate search and rescue use cases
- Follow ethical AI development practices
- Maintain the highest standards of security and reliability

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git
- CUDA-compatible GPU (recommended for ML inference)

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/foresight.git
   cd foresight
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   cd foresight-electron && npm install
   ```

5. Copy environment template:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Development Guidelines

### Code Style
- Python: Follow PEP 8, use `black` for formatting
- JavaScript: Use ESLint and Prettier
- Commit messages: Use conventional commits format

### Security Requirements
- Never commit API keys, tokens, or secrets
- Use environment variables for configuration
- Validate all user inputs
- Follow OWASP security guidelines

### Testing
- Write unit tests for new features
- Test with various video formats and resolutions
- Verify detection accuracy with test datasets
- Test cross-platform compatibility

### Documentation
- Update README.md for new features
- Document API changes
- Include inline code comments
- Update operational runbooks in `docs/`

## Contribution Process

### 1. Issue Creation
- Check existing issues first
- Use issue templates
- Provide detailed reproduction steps for bugs
- Include system information and logs

### 2. Pull Request Process
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Update documentation
5. Commit with descriptive messages
6. Push to your fork
7. Create a pull request

### 3. Pull Request Requirements
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No secrets or large files committed
- [ ] Security implications considered
- [ ] Performance impact assessed

## Areas for Contribution

### High Priority
- **Model Optimization**: Improve YOLO detection accuracy
- **Performance**: Reduce latency and memory usage
- **Integration**: DJI SDK integration for direct drone feeds
- **Documentation**: Operational procedures and troubleshooting

### Medium Priority
- **UI/UX**: Improve operator interface
- **Testing**: Automated testing framework
- **Deployment**: Docker containerization
- **Monitoring**: Enhanced telemetry and logging

### Low Priority
- **Features**: Additional detection models
- **Platforms**: Mobile app development
- **Analytics**: Historical data analysis

## Model Contributions

### Training Data
- Only use ethically sourced, properly licensed data
- Ensure data privacy compliance
- Document data sources and preprocessing

### Model Files
- Use Git LFS for model files
- Include model cards with performance metrics
- Provide validation datasets
- Document training procedures

## Reporting Security Issues

For security vulnerabilities:
1. **DO NOT** create public issues
2. Email security concerns to: [security contact]
3. Include detailed reproduction steps
4. Allow time for responsible disclosure

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

For questions about contributing:
- Create a discussion in GitHub Discussions
- Check existing documentation in `docs/`
- Review closed issues for similar questions

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Project documentation

Thank you for helping make search and rescue operations more effective!