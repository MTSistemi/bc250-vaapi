#!/usr/bin/env bash
# bc250-encoding-decoding-fix - https://github.com/simpmix/bc250-encoding-decoding-fix
#
# install_cachyos_arch.sh - Automated Arch Linux & CachyOS Package Installer
#

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}${BOLD}   BC-250 Driver Installer for Arch / CachyOS       ${NC}"
echo -e "${BLUE}======================================================${NC}"

WITH_32BIT=0
for arg in "$@"; do
    case "$arg" in
        --with-32bit|--with-i386)
            WITH_32BIT=1
            ;;
        -h|--help)
            echo "Usage: ./tools/install_cachyos_arch.sh [options]"
            echo "Options:"
            echo "  --with-32bit   Also build and install 32-bit companion driver for Steam Link"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARCH_DIR="$REPO_ROOT/packaging/arch"

if [ ! -f "$ARCH_DIR/PKGBUILD" ]; then
    echo -e "${YELLOW}Error: PKGBUILD not found in $ARCH_DIR${NC}"
    exit 1
fi

echo -e "  -> Installing required build dependencies via pacman..."
sudo pacman -S --needed --noconfirm base-devel git cmake meson ninja libva vulkan-headers glslang x264

echo -e "  -> Building and installing bc250-vaapi package..."
cd "$ARCH_DIR"
makepkg -si --noconfirm

if [ "$WITH_32BIT" -eq 1 ]; then
    echo -e "\n${BLUE}======================================================${NC}"
    echo -e "${BLUE}${BOLD}   Building 32-bit Companion Driver for Steam Link  ${NC}"
    echo -e "${BLUE}======================================================${NC}"
    bash "$REPO_ROOT/tools/build_32bit.sh"
fi

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}${BOLD}   Package Successfully Installed!                   ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "The driver is now registered system-wide with automatic PCI detection."
echo -e "To verify driver functionality:"
echo -e "  ${YELLOW}LIBVA_DRIVER_NAME=bc250 vainfo${NC}"

if [ "$WITH_32BIT" -eq 0 ]; then
    echo -e "\n${YELLOW}${BOLD}Note for Steam Link users (32-bit driver):${NC}"
    echo -e "Steam Link on Linux runs as a 32-bit client and requires the 32-bit companion driver."
    echo -e "To build and install the 32-bit driver for Steam Link:"
    echo -e "  ${BOLD}./tools/build_32bit.sh${NC}"
fi
