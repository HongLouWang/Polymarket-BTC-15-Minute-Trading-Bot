import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_bot():
    """Run the ETH 5-minute bot with auto-restart using this Python environment."""

    bot_script = "bot-eth-5min.py"
    python_cmd = sys.executable
    bot_args = sys.argv[1:] if len(sys.argv) > 1 else []

    print("=" * 80)
    print("ETH 5-MIN TRADING BOT - AUTO-RESTART WRAPPER")
    print("=" * 80)
    print(f"Platform: {sys.platform}")
    print(f"Python: {python_cmd}")
    print(f"Bot script: {bot_script}")
    print(f"Bot arguments: {bot_args}")
    print(f"Virtual env: {sys.prefix}")
    print("=" * 80)
    print()

    if not os.path.exists(bot_script):
        print(f"ERROR: Bot script '{bot_script}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print()
        print("Available .py files:")
        for file in os.listdir("."):
            if file.endswith(".py"):
                print(f"  - {file}")
        print()
        print("Please set bot_script to your bot filename")
        sys.exit(1)

    restart_count = 0

    while True:
        restart_count += 1

        print("=" * 80)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print(f"Starting bot (restart #{restart_count})...")
        print(f"Command: {python_cmd} {bot_script} {' '.join(bot_args)}")
        print("=" * 80)
        print()

        try:
            env = os.environ.copy()
            env["MARKET_INTERVAL_SECONDS"] = "300"
            env["MARKET_SLUG_PREFIX"] = "eth-updown-5m"
            env["MARKET_5M_INTERVAL_SECONDS"] = "300"
            env["MARKET_5M_SLUG_PREFIX"] = "eth-updown-5m"
            env["MARKET_ETH_5M_SLUG_PREFIX"] = "eth-updown-5m"
            env.setdefault("TRADE_JOURNAL_ETH_5M_PATH", "trade_journal_eth_5min.jsonl")
            env.setdefault("PAPER_TRADES_ETH_5M_PATH", "paper_trades_eth_5min.json")
            env.setdefault("ETH_SPOT_PRODUCT_ID", "ETH-USD")
            env.setdefault("ETH_DERIBIT_CURRENCY", "ETH")
            env.setdefault("ETH_REDIS_SIMULATION_KEY", "eth_trading:simulation_mode")

            cmd = [python_cmd, bot_script] + bot_args
            result = subprocess.run(cmd, env=env, check=False)

            exit_code = result.returncode

            print()
            print("=" * 80)
            print(f"Bot stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Exit code: {exit_code}")
            print("=" * 80)

            if exit_code in [0, 143, 15, -15]:
                print("Normal auto-restart - loading fresh filters...")
                wait_time = 2
            else:
                print(f"Error detected (code {exit_code}) - waiting before retry...")
                wait_time = 10

            print(f"Restarting in {wait_time} seconds...")
            print()
            time.sleep(wait_time)

        except KeyboardInterrupt:
            print()
            print("=" * 80)
            print("Keyboard interrupt received - stopping wrapper")
            print("=" * 80)
            break

        except Exception as exc:
            print()
            print("=" * 80)
            print(f"ERROR running bot: {exc}")
            print("=" * 80)
            print("Waiting 10 seconds before retry...")
            print()
            time.sleep(10)


if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        print("\nStopped by user")
        sys.exit(0)
