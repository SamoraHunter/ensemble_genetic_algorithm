# Contributing to ensemble_genetic_algorithm

Thank you for your interest in contributing to ensemble_genetic_algorithm! We welcome contributions from the community, whether it's fixing a bug, adding a feature, or improving the documentation. By contributing to this project, you agree to abide by the following guidelines:

## Code of Conduct

Please read and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to fostering a welcoming and inclusive community.

## Getting Started

Before you start contributing, please make sure you have:

1. **Forked** the repository to your GitHub account.
2. **Cloned** the forked repository to your local machine.


```bash
git clone https://github.com/YOUR_USERNAME/ensemble_genetic_algorithm.git
cd ensemble_genetic_algorithm
```

Created a new branch for your contributions.
3. **Set up the development environment**. This project uses `pyproject.toml` to manage dependencies. It is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
pip install -e .[dev]
```

4. **Created a new branch** for your contributions.

```bash
git checkout -b feature/your-feature-name
```

Making Changes
Ensure that you are working on the latest code from the main branch.

git checkout main
git pull origin main

Make your changes, following the project's coding style and guidelines.
Write clear and concise commit messages describing your changes.

Testing
Make sure to test your changes thoroughly before submitting a pull request. A synthetic dataset created with make classification is made available in unit tests. 

Submitting a Pull Request
When you're ready to submit your changes:

Push your changes to your fork on GitHub.

git push origin feature/your-feature-name
Open a Pull Request against the main branch of the original repository at https://github.com/SamoraHunter/ensemble_genetic_algorithm/.

Review Process
Your Pull Request will be reviewed by maintainers and other contributors. Be prepared to address any feedback and make necessary changes.

Code of Conduct
Please note that all contributors are expected to adhere to our Code of Conduct. 

Licensing
By contributing to this project, you agree that your contributions will be licensed under the project's LICENSE file.

Thank you for contributing to ensemble_genetic_algorithm!
