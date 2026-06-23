#!/usr/bin/env python3
"""
Quick test script to verify robot server connection.
Run this on the POLICY COMPUTER to test connection to robot server.
"""

import argparse
import sys
from pathlib import Path

# remote_environment_adapter lives in the parent xarm_scripts/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remote_environment_adapter import RemoteEnvironmentAdapter


def test_connection(host: str, port: int):
    """Test connection to robot server."""
    print(f"\n{'=' * 60}")
    print("Testing connection to robot server")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"{'=' * 60}\n")

    try:
        # Create adapter
        print("1. Creating remote environment adapter...")
        env = RemoteEnvironmentAdapter(host=host, port=port, timeout=10.0)
        print("   ✓ Connected successfully!")

        # Get metadata
        print("\n2. Getting metadata...")
        metadata = env.get_metadata()
        print(f"   Camera names: {metadata.get('camera_names')}")
        print(f"   Control freq: {metadata.get('control_freq')} Hz")
        print(f"   Obs history: {metadata.get('obs_history_len')}")
        print(f"   Intervention: {metadata.get('enable_intervention')}")

        # Get status
        print("\n3. Getting status...")
        status = env.get_status()
        print(f"   Episode active: {status.get('episode_active')}")
        print(f"   Episode index: {status.get('episode_idx')}")

        # Test reset
        print("\n4. Testing reset...")
        obs, info = env.reset(seed=42)
        print("   ✓ Reset successful!")
        print(f"   Image cameras: {list(obs['images'].keys())}")
        print(f"   State keys: {list(obs['state'].keys())}")

        # Test a single step
        print("\n5. Testing step with zero action...")
        import numpy as np

        action = np.zeros(14, dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        print("   ✓ Step successful!")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}, Truncated: {truncated}")

        # Close
        print("\n6. Closing environment...")
        env.close()
        print("   ✓ Closed successfully!")

        print(f"\n{'=' * 60}")
        print("✓ All tests passed! Robot server is ready.")
        print(f"{'=' * 60}\n")
        return True

    except Exception as e:
        print(f"\n✗ Connection test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Is robot server running?")
        print(f"     Check: curl http://{host}:{port}/api/health")
        print("  2. Is the host/port correct?")
        print("  3. Is firewall blocking the port?")
        print(f"  4. Can you ping the host? ping {host}")
        print(f"{'=' * 60}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test connection to robot environment server")
    parser.add_argument("--host", type=str, default="localhost", help="Robot server hostname or IP")
    parser.add_argument("--port", type=int, default=8080, help="Robot server port")
    args = parser.parse_args()

    success = test_connection(args.host, args.port)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
