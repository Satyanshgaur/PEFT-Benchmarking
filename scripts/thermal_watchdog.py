"""Thermal Watchdog Daemon for PEFT Benchmarking.

Monitors CPU and GPU temperatures every N seconds.
1. Fan Boost Tier (≥ 75°C): Automatically activates max fan cooling performance profiles.
2. Safety Shutdown Tier (≥ 88°C): Automatically terminates training processes if thermals hit maximum safety threshold.
"""

import os
import sys
import time
import glob
import signal
import datetime
import argparse
import subprocess


def get_gpu_temp() -> float:
    """Query current GPU temperature via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True
        )
        return float(out.strip())
    except Exception:
        return 0.0


def get_cpu_temp() -> float:
    """Query maximum CPU core temperature from hwmon thermal sensors."""
    temps = []
    for path in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
        try:
            with open(path, "r") as f:
                val = float(f.read().strip()) / 1000.0
                if 0 < val < 150:
                    temps.append(val)
        except Exception:
            pass
    return max(temps) if temps else 0.0


def set_fan_performance_mode():
    """Trigger system performance profile to force maximum fan airflow."""
    try:
        subprocess.run(["tuned-adm", "profile", "throughput-performance"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def terminate_training_processes(log_file: str, reason: str):
    """Terminate all active Python benchmark training processes."""
    timestamp = datetime.datetime.now().isoformat()
    msg = f"🔥 [{timestamp}] THERMAL SHUTDOWN TRIGGERED: {reason}\n"
    print(f"\n{msg}", flush=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg)

    # Find and kill run_single_benchmark.py and run_full_matrix.py processes
    current_pid = os.getpid()
    try:
        pids_out = subprocess.check_output(
            ["pgrep", "-f", "scripts/run_"],
            text=True
        ).strip().split()

        for p_str in pids_out:
            pid = int(p_str)
            if pid != current_pid:
                print(f"⚠️ Terminating training process PID {pid} due to thermal limit...")
                try:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(1)
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    except subprocess.CalledProcessError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Dynamic thermal watchdog monitor for training execution.")
    parser.add_argument("--fan_boost_temp", type=float, default=75.0, help="Temperature in °C to trigger max fan performance profile (default: 75.0)")
    parser.add_argument("--max_temp", type=float, default=88.0, help="Maximum allowed temperature in °C for safety shutdown (default: 88.0)")
    parser.add_argument("--check_interval", type=float, default=3.0, help="Monitoring check interval in seconds (default: 3.0)")
    parser.add_argument("--log_file", type=str, default="results/thermal_shutdown.log", help="Path to thermal shutdown log file")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    print(f"🛡️ Dynamic Thermal Watchdog Active: Fan Boost at {args.fan_boost_temp}°C | Safety Stop at {args.max_temp}°C")

    fan_boosted = False

    while True:
        gpu_temp = get_gpu_temp()
        cpu_temp = get_cpu_temp()
        max_current = max(gpu_temp, cpu_temp)

        # Trigger max fan profile as thermals approach boost threshold
        if max_current >= args.fan_boost_temp and not fan_boosted:
            set_fan_performance_mode()
            fan_boosted = True
            print(f"🌀 High temperature detected ({max_current:.1f}°C): Max fan performance mode activated!", flush=True)

        # Safety shutdown if thermals reach upper limit
        if gpu_temp >= args.max_temp:
            reason = f"GPU Temperature ({gpu_temp:.1f}°C) exceeded safety limit of {args.max_temp}°C!"
            terminate_training_processes(args.log_file, reason)
            sys.exit(1)

        if cpu_temp >= args.max_temp:
            reason = f"CPU Temperature ({cpu_temp:.1f}°C) exceeded safety limit of {args.max_temp}°C!"
            terminate_training_processes(args.log_file, reason)
            sys.exit(1)

        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
