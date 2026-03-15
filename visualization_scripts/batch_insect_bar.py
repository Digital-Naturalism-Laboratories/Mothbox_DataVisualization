import subprocess
import os
import time

script_path = r"C:\Users\andre\Documents\GitHub\Mothbox_DataVisualization\visualization_scripts\insect_bar.py"

# ── Folders to process ────────────────────────────────────────────────────────
# Each entry is the path to a folder of rembg-processed insect images.
input_paths = [
    r"F:\Deployments\Panama\Hoya_119m_bothDeer_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_163m_unrulyArao_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_168m_doubleParina_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_277m_adeptTurca_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_408m_calmoBarbo_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_310m_flatHapuku_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_505m_prizeCrab_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_606m_grisMejua_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_714m_remoteAhulla_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_812m_liftAlce_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_916m_layerMomoto_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_1004m_accionSauro_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_1300m_alertTollo_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_1110m_cuervoCinife_2025-01-26\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_1416m_fondoGorila_2025-01-27\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_1508m_waveUrta_2025-01-27\2025-01-27\patches\rembg",
    r"F:\Deployments\Panama\Hoya_1534m_wingedHapuku_2025-01-27\2025-01-27\patches\rembg",
]

# ── Shared insect_bar.py settings (passed as CLI args) ────────────────────────
# Remove or comment out any you want to leave at their defaults.
BAR_ARGS = [
    "--width",   "2000",
    "--scale",   "0.2",
    "--padding", "2",
    # "--background", "255,255,255",   # uncomment for white background
    # "--outline",                      # uncomment to draw coloured outlines
    # "--no-sort-by-size",              # uncomment to disable size sorting
    # "--no-sort-large-bottom",         # uncomment to put smallest at bottom
    # "--cluster",                      # uncomment to enable perceptual clustering
    # "--limit", "500",                 # uncomment to cap images per folder
]

# ─────────────────────────────────────────────────────────────────────────────

start_time_overall = time.time()
succeeded = []
failed    = []

for i, input_path in enumerate(input_paths, start=1):
    folder_name = os.path.basename(os.path.normpath(input_path))
    print(f"\n{'='*60}")
    print(f"[{i}/{len(input_paths)}] Processing: {folder_name}")
    print(f"  Path: {input_path}")

    if not os.path.isdir(input_path):
        print(f"  ⚠  Folder not found — skipping.")
        failed.append(input_path)
        continue

    t0 = time.time()
    try:
        subprocess.run(
            ["python", script_path, "--input", input_path] + BAR_ARGS,
            check=True
        )
        elapsed = time.time() - t0
        print(f"  ✓ Done in {elapsed:.1f}s")
        succeeded.append(input_path)
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"  ✗ FAILED after {elapsed:.1f}s (exit code {e.returncode})")
        failed.append(input_path)

# ── Summary ───────────────────────────────────────────────────────────────────
total = time.time() - start_time_overall
hours   = int(total // 3600)
minutes = int((total % 3600) // 60)
seconds = total % 60

print(f"\n{'='*60}")
print(f"Batch complete: {len(succeeded)} succeeded, {len(failed)} failed")
print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:.1f}s")

if failed:
    print("\nFailed folders:")
    for p in failed:
        print(f"  ✗ {p}")
