# Python UV Template

A modern Python project template using UV, Ruff, and best practices for Python development. This template provides a streamlined, fully-configured starting point for new Python projects with focus on reliability, type safety, and modern development workflow.

## Features

- **Python 3.13 support** - Latest Python version with full typing and modern features
- **UV Package Manager** - Fast, reliable alternative to pip for dependency management
- **Ruff Integration** - Comprehensive Python linter and formatter
- **Type Checking** - MyPy configured with strict settings
- **Testing with Coverage** - Pytest configured with coverage reporting
- **Pre-commit Hooks** - Automated code quality checks
- **Conventional Commits** - Enforced commit message format
- **Security Scanning** - Gitleaks integration to prevent secrets in commits
- **Pure Python Package** - Proper package structure with `py.typed` support

## Quick Start

### Creating a New Project

This is a [Copier](https://copier.readthedocs.io/) template. To create a new project:

1. Generate a new project:
   ```
   copier copy gh:sonesuke/python-uv-template path/to/your-project
   ```

2. You'll be prompted for:
   - `project_name`: Your project name (can contain hyphens)
   - `package_name`: Python package name (will use underscores by default)

### Development Setup

Once your project is created:

1. Set up a virtual environment and install development dependencies:
   ```
   uv sync
   ```

2. Set up pre-commit:
   ```
   uv run pre-commit install
   ```

### Development Workflow

- **Format and lint code**:
  ```
  uv run ruff check --fix
  ```

- **Run tests with coverage**:
  ```
  uv run pytest --cov=src
  ```

- **Type checking**:
  ```
  uv run mypy
  ```

## Project Structure

```
your-project/
├── src/
│   └── your_package/
│       ├── __init__.py
│       └── py.typed         # Marker file for type checking
├── tests/
│   └── test_your_package.py
├── .pre-commit-config.yaml  # Pre-commit hook configuration
├── .python-version          # Python version specification
├── pyproject.toml           # Project configuration 
└── README.md                # Project documentation
```

## Configuration Details

### Package Configuration

The project uses `pyproject.toml` for all configuration, following modern Python packaging standards:

- **Build System**: Hatchling for reliable, modern packaging
- **Dependencies**: Minimal core dependencies by default, add what you need
- **Development Tools**:
  - MyPy for comprehensive type checking with strict settings
  - Pytest with coverage reporting
  - Ruff for fast, comprehensive linting and formatting

### Ruff Configuration

Ruff is configured with a comprehensive set of rules:

- C9: McCabe complexity checking to keep functions maintainable
- ANN: Type annotations enforcement for better code quality
- S: Security checks to catch common vulnerabilities
- E, F, W: Style and error checking (pycodestyle, pyflakes, warnings)
- I: Import sorting for consistent organization
- D: Documentation standards enforcement

The default configuration sets:
- Line length to 120 characters
- Double quotes for string literals
- Split imports on trailing commas
- Maximum complexity of 20

### MyPy Configuration

Type checking is configured with strict settings:

- Strict mode enabled for comprehensive type checking
- Files limited to the `src` directory
- Untyped imports are allowed to facilitate working with third-party libraries

### Testing Configuration

Tests are configured using pytest:

- Test discovery in the `tests` directory
- Python path configured to include the `src` directory
- Coverage reporting set up to track code coverage

## Using UV for Dependency Management

[UV](https://github.com/astral-sh/uv) is a fast, reliable Python package installer and resolver that significantly speeds up dependency management:

- **Install dependencies**: `uv sync`
- **Add a new dependency**: `uv add package-name`

## Benefits of This Template

- **Consistency**: Enforced code style and quality through automated tools
- **Reliability**: Type checking and comprehensive testing
- **Security**: Pre-commit hooks catch secrets and security issues
- **Speed**: UV package manager is significantly faster than pip
- **Modern Practices**: Configured for current Python best practices

## License

This template is available as open source under the terms of the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.