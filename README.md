# RL Env

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

**Demo Configuration**

- Site: [Sauce Demo Shop](https://www.saucedemo.com/)
- Task: Sign in, add a product to the cart, and checkout
- Max iterations/attempt: 50
- Attempts: 5
