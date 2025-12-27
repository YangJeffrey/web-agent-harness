# Web Agent Harness

Uses [Webarena](https://arxiv.org/pdf/2307.13854), [Playwright](https://playwright.dev/), and [Claude Computer use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/computer-use-tool)

## Setup (macOS)

### Prerequisites

- Python 3.8+
- Chromedriver: `brew install chromedriver`

### Installation

1. **Create and activate virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

3. **Configure API key**

   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
   ```

### Run the application

```bash
python index.py
```

**Command-line Options**

- `--environment` / `-e`: URL of the website/environment to interact with (default: https://www.saucedemo.com/)
- `--task` / `-t`: Task description for the agent to complete (default: Sign in, add a product to the cart, and checkout)
- `--max-iterations` / `-i`: Maximum iterations per attempt (default: 50)
- `--attempts` / `-a`: Number of attempts (pass@k rate) (default: 5)

**Examples**

```bash
# Use default configuration
python index.py

# Custom task with different site
python index.py -e "https://example.com" -t "Navigate to the about page"

# Increase max iterations and attempts
python index.py -i 100 -a 10

# Full custom configuration
python index.py --environment "https://mysite.com" --task "Complete the form" --max-iterations 75 --attempts 3
```

**Default Demo Configuration**

- Site: [Sauce Demo Shop](https://www.saucedemo.com/)
- Task: Sign in, add a product to the cart, and checkout
- Max iterations/attempt: 50
- Attempts: 5
