#!/bin/bash
# Hardware and OS performance optimization script for Intel Core Ultra 5 125H & Intel Arc iGPU
set -e

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./scripts/tune-hardware.sh)"
    exit 1
fi

echo "[*] Setting CPU scaling governor to performance on all cores..."
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$g" 2>/dev/null || true
done

echo "[*] Pinning Intel Arc GPU (gt0) to max 2200 MHz boost clock..."
echo 2200 > /sys/devices/pci0000:00/0000:00:02.0/drm/card1/gt/gt0/rps_min_freq_mhz 2>/dev/null || true
echo 2200 > /sys/devices/pci0000:00/0000:00:02.0/drm/card1/gt/gt0/rps_boost_freq_mhz 2>/dev/null || true
echo 2200 > /sys/devices/pci0000:00/0000:00:02.0/drm/card1/gt_min_freq_mhz 2>/dev/null || true
echo 2200 > /sys/devices/pci0000:00/0000:00:02.0/drm/card1/gt_boost_freq_mhz 2>/dev/null || true

echo "[*] Enabling Transparent Huge Pages (always) for 2MB memory blocks..."
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo always > /sys/kernel/mm/transparent_hugepage/defrag

echo "[*] Tuning swappiness to 10..."
sysctl -w vm.swappiness=10

echo "[+] Hardware and memory optimizations applied successfully."
