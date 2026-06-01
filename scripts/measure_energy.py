import sys
sys.stdout.reconfigure(encoding='utf-8')
import subprocess
import threading
import time
import re

try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_PYNVML = True
    print(f"GPU: {pynvml.nvmlDeviceGetName(handle)}")
except Exception as e:
    print(f"pynvml not available: {e}")
    print("Install with: pip install nvidia-ml-py")
    sys.exit(1)

BINARY = "run_cuda.exe"
MODEL  = "llama3_2_1B_q8_group.bin"
SAMPLE_INTERVAL = 0.05  # 50ms

def sample_power(stop_event, samples):
    while not stop_event.is_set():
        try:
            mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            samples.append(mw / 1000.0)
        except Exception:
            pass
        time.sleep(SAMPLE_INTERVAL)

def parse_throughput(text):
    """Extract tok/s values from run_cuda output."""
    # matches lines like "  -> Throughput:       44.00 tok/s"
    values = re.findall(r'Throughput:\s+([\d.]+)\s+tok/s', text)
    return [float(v) for v in values]

def measure_stage(label):
    print(f"\n{'='*50}")
    print(f"Measuring: {label}")
    print(f"{'='*50}")

    samples = []
    stop_event = threading.Event()

    # Warmup: run once without measuring
    # Start power sampling
    t = threading.Thread(target=sample_power, args=(stop_event, samples))
    t.start()

    wall_start = time.time()
    proc = subprocess.Popen([BINARY, MODEL], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    try:
        stdout_bytes, _ = proc.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_bytes, _ = proc.communicate()
    result_text = stdout_bytes.decode('utf-8', errors='replace')
    wall_end = time.time()
    wall_end = time.time()

    stop_event.set()
    t.join()

    elapsed = wall_end - wall_start
    output = result_text

    # Drop first 20% of samples (loading / warmup phase)
    trim = max(1, len(samples) // 5)
    active_samples = samples[trim:]
    avg_power_w = sum(active_samples) / len(active_samples) if active_samples else 0

    throughputs = parse_throughput(output)
    # Separate Stage 5 (first 4) and Stage 6 (last 4) prompts
    stage5_toks = throughputs[:4]
    stage6_toks = throughputs[4:]

    print(f"Wall time     : {elapsed:.1f}s")
    print(f"Avg power     : {avg_power_w:.1f} W  (n={len(active_samples)} samples)")
    print(f"Min/Max power : {min(active_samples):.1f} / {max(active_samples):.1f} W")

    if stage5_toks:
        avg5 = sum(stage5_toks) / len(stage5_toks)
        tokj5 = avg5 / avg_power_w if avg_power_w > 0 else 0
        print(f"\nStage 5 (Flash Attn):")
        print(f"  tok/s  = {avg5:.2f}  (runs: {[f'{v:.2f}' for v in stage5_toks]})")
        print(f"  tok/J  = {tokj5:.3f}  ({avg5:.2f} tok/s / {avg_power_w:.1f} W)")

    if stage6_toks:
        avg6 = sum(stage6_toks) / len(stage6_toks)
        tokj6 = avg6 / avg_power_w if avg_power_w > 0 else 0
        print(f"\nStage 6 (dp4a W8A8):")
        print(f"  tok/s  = {avg6:.2f}  (runs: {[f'{v:.2f}' for v in stage6_toks]})")
        print(f"  tok/J  = {tokj6:.3f}  ({avg6:.2f} tok/s / {avg_power_w:.1f} W)")

    return avg_power_w, stage5_toks, stage6_toks

if __name__ == "__main__":
    print("Energy Measurement — MiniLlama-Infra")
    print(f"Binary: {BINARY} | Model: {MODEL}")

    avg_power, s5, s6 = measure_stage("Stage 5 + Stage 6")

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    if s5 and s6 and avg_power > 0:
        avg5 = sum(s5) / len(s5)
        avg6 = sum(s6) / len(s6)
        tokj5 = avg5 / avg_power
        tokj6 = avg6 / avg_power
        print(f"GPU avg power  : {avg_power:.1f} W")
        print(f"Stage 5        : {avg5:.2f} tok/s  |  {tokj5:.3f} tok/J")
        print(f"Stage 6 dp4a   : {avg6:.2f} tok/s  |  {tokj6:.3f} tok/J")
        print(f"Energy gain    : {tokj6/tokj5:.2f}x (Stage 6 vs Stage 5)")
    print("="*50)
