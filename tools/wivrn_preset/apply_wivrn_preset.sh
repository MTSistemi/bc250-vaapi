#!/usr/bin/env bash
# bc250-encoding-decoding-fix - https://github.com/simpmix/bc250-encoding-decoding-fix
#
# apply_wivrn_preset.sh - Applies optimized BC-250 VR streaming configuration for WiVRn
#

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}${BOLD}   Apply BC-250 WiVRn VR Streaming Preset           ${NC}"
echo -e "${BLUE}======================================================${NC}"

CODEC="h264"
if [[ "$1" == "--hevc" || "$1" == "--h265" ]]; then
    CODEC="hevc"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$CODEC" == "hevc" ]; then
    SOURCE_CONF="$SCRIPT_DIR/config_hevc.json"
    echo -e "  -> Selecting Codec: ${GREEN}HEVC (H.265) Low-Latency VA-API${NC}"
else
    SOURCE_CONF="$SCRIPT_DIR/config.json"
    echo -e "  -> Selecting Codec: ${GREEN}H.264 Baseline/High VA-API${NC} (use --hevc for HEVC)"
fi

# Configure user WiVRn directory
TARGET_DIR="$HOME/.config/wivrn"
mkdir -p "$TARGET_DIR"

TARGET_CONF="$TARGET_DIR/config.json"

if [ -f "$TARGET_CONF" ]; then
    BACKUP_FILE="${TARGET_CONF}.backup.$(date +%s)"
    echo -e "  -> Backing up existing WiVRn config to: $BACKUP_FILE"
    cp "$TARGET_CONF" "$BACKUP_FILE"
fi

echo -e "  -> Applying BC-250 VR configuration to: $TARGET_CONF"
cp "$SOURCE_CONF" "$TARGET_CONF"

# Setup WiVRn environment configuration for BC-250 (passive OpenMP wait, 4 threads, 4 slices, low QP floor)
ENV_FILE="$HOME/.config/environment.d/98-wivrn-bc250.conf"
mkdir -p "$HOME/.config/environment.d"

cat << 'EOF' > "$ENV_FILE"
LIBVA_DRIVER_NAME=bc250
BC250_FAST_MODE=1
BC250_PIPELINE=1
BC250_MAX_CPU_THREADS=4
BC250_SLICES_PER_FRAME=4
BC250_HEVC_SLICES=4
BC250_QP_MIN=8
OMP_WAIT_POLICY=PASSIVE
GOMP_SPINCOUNT=0
OMP_NUM_THREADS=4
OMP_DYNAMIC=FALSE
EOF

echo -e "  ${GREEN}✓ Configured user session environment in $ENV_FILE${NC}"

# Check systemd user service if present
if systemctl --user is-active --quiet wivrn 2>/dev/null; then
    echo -e "  -> Restarting WiVRn service..."
    systemctl --user restart wivrn
    echo -e "  ${GREEN}✓ WiVRn restarted successfully!${NC}"
fi

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}${BOLD}   WiVRn Preset Successfully Applied!                ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "\nHardware Configuration Summary:"
echo -e "  * Headset Throughput Target: ${GREEN}~200+ Mbits/s${NC} (unlocked via BC250_QP_MIN=8)"
echo -e "  * Target Motion-to-Photon:   ${GREEN}~28-35 ms${NC}"
echo -e "  * Codec Selected:            ${GREEN}${CODEC^^} VA-API${NC}"
echo -e "  * Slices / Concurrency:      ${GREEN}4 slices, 4 OpenMP worker threads${NC}"
echo -e "  * Host CPU Wait Mode:        ${GREEN}PASSIVE (0% idle spin)${NC}"
echo -e "\nTo launch WiVRn manually with this environment:"
echo -e "  ${YELLOW}wivrn-server${NC}"
