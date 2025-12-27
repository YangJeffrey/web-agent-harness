import os
import asyncio
import json
import base64
import io
import re
import argparse
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np
from anthropic import Anthropic
from playwright.async_api import async_playwright
import gymnasium as gym
from gymnasium import spaces

class WebEnvironment(gym.Env):
    """Gym environment that wraps web browser interactions for RL"""

    def __init__(self, environment_url: str = "https://www.saucedemo.com/", headless: bool = False):
        super().__init__()

        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(8),
            'x': spaces.Box(low=0, high=1024, shape=(1,), dtype=np.float32),
            'y': spaces.Box(low=0, high=768, shape=(1,), dtype=np.float32),
            'text': spaces.Text(max_length=100),
            'url': spaces.Text(max_length=200),
            'direction': spaces.Discrete(2),
            'element_id': spaces.Text(max_length=50),
            'start_x': spaces.Box(low=0, high=1024, shape=(1,), dtype=np.float32),
            'start_y': spaces.Box(low=0, high=768, shape=(1,), dtype=np.float32)
        })

        self.observation_space = spaces.Dict({
            'screenshot': spaces.Box(low=0, high=255, shape=(768, 1024, 3), dtype=np.uint8),
            'dom_info': spaces.Text(max_length=10000),
            'current_url': spaces.Text(max_length=200)
        })

        self.browser = None
        self.page = None
        self.playwright = None
        self.headless = headless
        self.environment_url = environment_url

        self.keyboard_map = {
            'ctrl+a': 'Control+a', 'ctrl+c': 'Control+c', 'ctrl+v': 'Control+v', 'ctrl+x': 'Control+x',
            'delete': 'Delete', 'backspace': 'Backspace', 'return': 'Enter', 'enter': 'Enter',
            'escape': 'Escape', 'tab': 'Tab', 'space': ' '
        }

    async def async_reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Asynchronous reset implementation"""
        super().reset(seed=seed)

        try:
            if not self.browser:
                await self._init_browser()
            else:
                context = self.browser.contexts[0]
                await context.close()
                context = await self.browser.new_context(viewport={"width": 1024, "height": 768})
                self.page = await context.new_page()

            await self.page.goto(self.environment_url, wait_until="networkidle")

            observation = await self._get_observation()
            return observation, {}

        except Exception as e:
            print(f"Error in reset: {e}")
            return self._get_empty_observation(), {}

    async def async_step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Asynchronous step implementation"""
        try:
            action_type = action.get('action_type', 0)

            if action_type == 0:
                x, y = float(action.get('x', 0)), float(action.get('y', 0))
                await self.page.mouse.click(x, y)
                print(f"Clicked at ({x}, {y})")

            elif action_type == 1:
                text = action.get('text', '')
                element_id = action.get('element_id', '')
                if element_id:
                    await self.page.click(f"#{element_id}")
                await self.page.keyboard.type(text)
                print(f"Typed: {text}")

            elif action_type == 2:
                direction = 'down' if action.get('direction', 1) == 1 else 'up'
                pixels = 200 if direction == 'down' else -200
                await self.page.evaluate(f"window.scrollBy(0, {pixels})")
                print(f"Scrolled {direction}")

            elif action_type == 3:
                url = action.get('url', '')
                await self.page.goto(url, wait_until="networkidle")
                print(f"Navigated to: {url}")

            elif action_type == 4:
                key = action.get('text', '')
                if key.lower() in self.keyboard_map:
                    await self.page.keyboard.press(self.keyboard_map[key.lower()])
                else:
                    await self.page.keyboard.press(key)
                print(f"Pressed key: {key}")

            elif action_type == 5:
                start_x, start_y = float(action.get('start_x', 0)), float(action.get('start_y', 0))
                end_x, end_y = float(action.get('x', 0)), float(action.get('y', 0))
                await self.page.mouse.move(start_x, start_y)
                await self.page.mouse.down()
                await self.page.mouse.move(end_x, end_y)
                await self.page.mouse.up()
                print(f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")

            elif action_type == 6:
                x, y = float(action.get('x', 0)), float(action.get('y', 0))
                await self.page.mouse.dblclick(x, y)
                print(f"Double-clicked at ({x}, {y})")

            elif action_type == 7:
                x, y = float(action.get('x', 0)), float(action.get('y', 0))
                await self.page.mouse.click(x, y, button='right')
                print(f"Right-clicked at ({x}, {y})")

            await asyncio.sleep(0.5)

            observation = await self._get_observation()

            reward = await self._calculate_reward(observation)
            terminated = False
            truncated = False
            info = {}

            return observation, reward, terminated, truncated, info

        except Exception as e:
            print(f"Error in step: {e}")
            observation = await self._get_observation()
            return observation, 0.0, False, True, {"error": str(e)}

    async def _init_browser(self):
        """Initialize the browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = await self.browser.new_context(viewport={"width": 1024, "height": 768})
        self.page = await context.new_page()

        self.page.on("pageerror", lambda err: print(f"Page error: {err}"))

    async def _get_observation(self) -> Dict:
        """Get current state observation"""
        try:
            screenshot = await self.page.screenshot()
            screenshot_array = np.array(Image.open(io.BytesIO(screenshot)))

            dom_info = await self.page.evaluate("""
                () => JSON.stringify(Array.from(document.querySelectorAll('*'))
                    .filter(el => el.textContent?.trim())
                    .slice(0, 50)
                    .map(el => ({
                        tag: el.tagName.toLowerCase(),
                        text: el.textContent.trim().substring(0, 100),
                        id: el.id || '',
                        class: el.className || ''
                    })))
            """)

            current_url = self.page.url

            return {
                'screenshot': screenshot_array,
                'dom_info': dom_info,
                'current_url': current_url
            }
        except Exception as e:
            print(f"Error getting observation: {e}")
            return self._get_empty_observation()

    def _get_empty_observation(self) -> Dict:
        """Return empty observation for error cases"""
        return {
            'screenshot': np.zeros((768, 1024, 3), dtype=np.uint8),
            'dom_info': "[]",
            'current_url': ""
        }

    async def async_close(self):
        """Clean up resources"""
        if self.browser:
            await self.browser.close()
            await self.playwright.stop()
            print("Browser closed")

    def reset(self, seed=None, options=None):
        """Synchronous reset for Gym compatibility"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_reset(seed, options))

    def step(self, action):
        """Synchronous step for Gym compatibility"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_step(action))

    def close(self):
        """Synchronous close for Gym compatibility"""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_close())

    async def _calculate_reward(self, observation: Dict) -> float:
        """Calculate reward based on current state (0-1 scale for RL training)"""
        try:
            current_url = observation.get('current_url', '')
            dom_info = observation.get('dom_info', '[]')

            reward = 0.1

            try:
                dom_elements = json.loads(dom_info)
                if len(dom_elements) > 0:
                    reward += 0.1

                text_content = ' '.join([elem.get('text', '').lower() for elem in dom_elements])
                if any(word in text_content for word in ['success', 'complete', 'done']):
                    reward += 0.4
                elif any(word in text_content for word in ['error', 'failed', 'invalid']):
                    reward -= 0.2

            except (json.JSONDecodeError, TypeError):
                pass

            return max(0.0, min(reward, 1.0))

        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0


class WebAgent:
    """Web agent that uses Claude computer use with Gym environment."""

    def __init__(self, api_key: str, environment_url: str = "https://www.saucedemo.com/", model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.judge = Anthropic(api_key=api_key)

        self.env = WebEnvironment(environment_url=environment_url, headless=False)

        self.reward_history = []
        self.task_feedback_history = []
        self.max_feedback_history = 10
        self.messages = []

    async def _convert_claude_action_to_gym_action(self, action_input: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Claude computer action format to Gym action space format"""
        action_type = action_input.get('action', '')
        gym_action = {}

        action_type_map = {
            'click': 0, 'left_click': 0,
            'type': 1,
            'scroll': 2,
            'navigate': 3,
            'key': 4,
            'left_click_drag': 5,
            'double_click': 6,
            'right_click': 7
        }

        if action_type in action_type_map:
            gym_action['action_type'] = action_type_map[action_type]

            if 'x' in action_input and 'y' in action_input:
                gym_action['x'] = float(action_input['x'])
                gym_action['y'] = float(action_input['y'])
            elif 'coordinate' in action_input:
                x, y = action_input['coordinate']
                gym_action['x'] = float(x)
                gym_action['y'] = float(y)

            if action_type == 'left_click_drag' and 'start_coordinate' in action_input:
                start_x, start_y = action_input['start_coordinate']
                gym_action['start_x'] = float(start_x)
                gym_action['start_y'] = float(start_y)

            if action_type in ['type', 'key']:
                gym_action['text'] = action_input.get('text', '')

            if 'element_id' in action_input:
                gym_action['element_id'] = action_input['element_id']

            if action_type == 'scroll':
                direction = action_input.get('direction', 'down').lower()
                gym_action['direction'] = 1 if direction == 'down' else 0

            if action_type == 'navigate':
                gym_action['url'] = action_input.get('url', '')

            return gym_action

        return None

    async def execute_action(self, action_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Claude computer action through Gym"""
        try:
            print(f"Executing action: {action_input}")

            gym_action = await self._convert_claude_action_to_gym_action(action_input)

            if gym_action:
                _, reward, terminated, truncated, info = await self.env.async_step(gym_action)
                return {"success": True, "message": f"Executed {action_input.get('action', '')} action"}
            elif action_input.get('action', '') == 'screenshot':
                return {"success": True, "message": "Screenshot taken"}
            else:
                return {"success": False, "message": f"Unsupported action: {action_input.get('action', '')}"}

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _screenshot_to_base64(self, screenshot_array: np.ndarray) -> str:
        """Convert screenshot array to base64 string"""
        try:
            img = Image.fromarray(screenshot_array)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            print(f"Error converting screenshot to base64: {e}")

    def build_prompt_with_feedback(self, task: str, observation: Dict) -> str:
        """Build prompt with learning feedback"""
        prompt = f"Task: {task}\n\nCurrent URL: {observation.get('current_url', 'localhost:3001/')}\n\nComplete this task."

        if self.task_feedback_history:
            prompt += f"\n\n{'='*50}\nLEARNING FROM PREVIOUS ATTEMPTS:\n{'='*50}"

            for i, history in enumerate(self.task_feedback_history[-2:], 1):
                status = 'SUCCESS' if history['success'] else 'NEEDS IMPROVEMENT'
                prompt += f"\n\nATTEMPT {i}: Score {history['reward']:.2f}/1.0 ({status})"

                if 'IMPROVEMENT SUGGESTIONS:' in history['feedback']:
                    suggestions = history['feedback'].split('IMPROVEMENT SUGGESTIONS:')[-1].strip()
                    prompt += f"\nKey Improvements: {suggestions}"

            prompt += f"\n{'='*50}\nApply these lessons."

        return prompt

    async def evaluate_task(self, task: str) -> Dict[str, Any]:
        """Evaluate task completion"""
        try:
            observation = await self.env._get_observation()
            screenshot_array = observation['screenshot']
            screenshot_b64 = self._screenshot_to_base64(screenshot_array)
            dom_info = observation['dom_info']
            current_url = observation['current_url']

            actions = []
            for msg in self.messages:
                if msg['role'] == 'assistant':
                    for content in msg.get('content', []):
                        if hasattr(content, 'type') and content.type == 'tool_use':
                            actions.append(f"- {content.input.get('action', 'unknown')}")

            eval_prompt = f"""Evaluate web automation task completion.

            TASK: {task}
            URL: {current_url}
            DOM: {dom_info}
            ACTIONS: {chr(10).join(actions) if actions else 'No actions'}

            Return JSON with two separate assessments:
            1. A 0.0-1.0 reward score for RL training (how well was the task executed)
            2. A boolean success flag (was the task actually completed)

            {{
                "task_progress": {{
                    "completed_task": {{"score": 0.0-1.0, "feedback": "detailed explanation"}}
                }},
                "overall_feedback": "comprehensive feedback",
                "improvement_suggestions": "specific suggestions",
                "reward_score": 0.0-1.0,
                "task_completed": true/false
            }}"""

            response = self.judge.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=1500,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": eval_prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}}
                ]}]
            )

            json_match = re.search(r'\{.*\}', response.content[0].text.strip(), re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())

                reward = eval_data.get('reward_score', eval_data.get('total_score', 0.0))
                success = eval_data.get('task_completed', False)

                feedback = f"""EVALUATION: {reward:.2f}/1.0
                TASK COMPLETED: {success}
                FEEDBACK: {eval_data.get('overall_feedback', 'No feedback')}
                IMPROVEMENTS: {eval_data.get('improvement_suggestions', 'No suggestions')}"""

                return {
                    'reward': reward,
                    'success': success,
                    'feedback': feedback,
                    'component_scores': eval_data.get('task_progress', {})
                }
        except Exception as e:
            print(f"Evaluation error: {e}")

        return {'reward': 0.0, 'feedback': "Evaluation failed", 'success': False, 'component_scores': {}}

    async def run_task(self, task: str, max_iterations: int = 50, reset_environment: bool = True):
        """Run web automation task using RL environment"""
        if reset_environment:
            observation, _ = await self.env.async_reset()
            print(f"Environment reset. URL: {observation['current_url']}")
        else:
            observation = await self.env._get_observation()
            print(f"Continuing with current browser state. URL: {observation['current_url']}")

        screenshot_b64 = self._screenshot_to_base64(observation['screenshot'])

        initial_prompt = self.build_prompt_with_feedback(task, observation)

        self.messages = [{"role": "user", "content": [
            {"type": "text", "text": initial_prompt},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}}
        ]}]

        for iteration in range(1, max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{max_iterations} ---")

            try:
                response = self.client.beta.messages.create(
                    model=self.model, max_tokens=4096, messages=self.messages,
                    tools=[{"type": "computer_20250124", "name": "computer",
                           "display_width_px": 1024, "display_height_px": 768, "display_number": 1}],
                    betas=["computer-use-2025-01-24"]
                )

                self.messages.append({"role": "assistant", "content": response.content})

                for block in response.content:
                    if block.type == "text":
                        print(f"Claude: {block.text}")

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = await self.execute_action(block.input)
                        tool_results.append({
                            "type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })

                if not tool_results:
                    break

                self.messages.append({"role": "user", "content": tool_results})

                observation = await self.env._get_observation()
                screenshot_b64 = self._screenshot_to_base64(observation['screenshot'])

                self.messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"Updated screenshot. URL: {observation['current_url']}"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}}
                ]})

            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                break

        evaluation = await self.evaluate_task(task)
        self.reward_history.append(evaluation['reward'])
        self.task_feedback_history.append({
            'task': task, 'reward': evaluation['reward'],
            'success': evaluation['success'],
            'feedback': evaluation['feedback'], 'component_scores': evaluation['component_scores']
        })

        if len(self.task_feedback_history) > self.max_feedback_history:
            self.task_feedback_history = self.task_feedback_history[-self.max_feedback_history:]

        return self.messages, evaluation

    async def close(self):
        """Clean up resources"""
        await self.env.async_close()


async def main():
    """Main function with learning loop"""
    parser = argparse.ArgumentParser(description='Web automation agent with RL environment')
    parser.add_argument('--environment', '-e', type=str, default='https://www.saucedemo.com/', 
                       help='URL of the website/environment to interact with (default: https://www.saucedemo.com/)')
    parser.add_argument('--task', '-t', type=str, default='Sign in, add a product to the cart, and checkout',
                       help='Task description for the agent to complete (default: Sign in, add a product to the cart, and checkout)')
    parser.add_argument('--max-iterations', '-i', type=int, default=50,
                       help='Maximum iterations per attempt (default: 50)')
    parser.add_argument('--attempts', '-a', type=int, default=5,
                       help='Number of attempts (pass@k rate) (default: 5)')
    
    args = parser.parse_args()
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable")
        return

    agent = WebAgent(api_key=api_key, environment_url=args.environment)
    task = args.task

    reset_next = True

    for attempt in range(1, args.attempts + 1):
        print(f"\n{'='*60}\nATTEMPT {attempt}/{args.attempts}: {task}\n{'='*60}")

        try:
            result, evaluation = await agent.run_task(task, max_iterations=args.max_iterations, reset_environment=reset_next)

            print(f"\n=== ATTEMPT {attempt} COMPLETE ===")
            print(f"Score: {evaluation['reward']:.3f} | Success: {evaluation['success']}")

            if len(agent.reward_history) > 1:
                print(f"History: {[f'{r:.2f}' for r in agent.reward_history]}")

            reset_next = not evaluation['success']

            if reset_next:
                print("Task not completed")
            else:
                print("Task completed successfully")

            if evaluation['success']:
                print(f"\nTASK COMPLETED SUCCESSFULLY")
                break

        except Exception as e:
            print(f"Error in attempt {attempt}: {e}")
            import traceback
            traceback.print_exc()
            reset_next = True

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Attempts: {len(agent.reward_history)}")
    print(f"Scores: {[f'{r:.2f}' for r in agent.reward_history]}")
    if agent.reward_history:
        print(f"Best: {max(agent.reward_history):.3f}")
        print(f"Latest: {agent.reward_history[-1]:.3f}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
