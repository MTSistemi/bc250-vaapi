#!/usr/bin/env bash
# bc250-encoding-decoding-fix - https://github.com/simpmix/bc250-encoding-decoding-fix
#
# apply_gpu_ringfence.sh - Ring-fence GPU Compute Units for the Sunshine
#                          encoder, so a game cannot take the whole GPU.
#
# The honest short version of what this can and cannot do is in
# docs/streaming-ringfence.md. The one-paragraph version:
#
#   There is no "reserve N CUs for this process" API on this stack.
#   VK_AMD_shader_core_policy, which is the vendor extension that would
#   provide it, is not exposed by RADV at all. What Mesa does provide -
#   AMD_CU_MASK, since 22.0, for both radeonsi and RADV - restricts a
#   *process* to a mask of CUs within each shader array, and it is read
#   when the Mesa screen is created. So the fence is built here, in the
#   environment of the two processes, before either starts. The granularity
#   is per-shader-array, which is the part that surprises people: on a
#   40-CU part that is "one CU in each of 8 arrays", not "2 of the 40".
#
# Usage:
#   apply_gpu_ringfence.sh [--cus-per-array N] [--no-priority] [--dry-run]
#   apply_gpu_ringfence.sh --remove
#
#   --cus-per-array N   How many CUs per shader array to leave to the
#                       encoder. Default 2, which is the finest split Mesa
#                       will accept on gfx10 (it rejects a mask that does not
#                       have both a CU in {0,1,3,4} and one in {2,3}).
#   --no-priority       Skip the queue-priority half. Without this, the
#                       script also gives Sunshine CAP_SYS_NICE and sets
#                       BC250_QUEUE_PRIORITY=high, which is the only way the
#                       existing VK_EXT_global_priority knob can get above
#                       the driver default - it is refused outright to an
#                       unprivileged process.
#   --dry-run           Print everything, change nothing.
#   --remove            Undo everything this script did.

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

DROPIN_DIR="$HOME/.config/systemd/user/sunshine.service.d"
DROPIN="$DROPIN_DIR/bc250-gpu-ringfence.conf"
GAME_WRAPPER_DIR="$HOME/.local/bin"
GAME_WRAPPER="$GAME_WRAPPER_DIR/bc250-cu-masked-game"

CUS_PER_ARRAY=2
DO_PRIORITY=1
DRY_RUN=0
MODE="apply"

while [ $# -gt 0 ]; do
    case "$1" in
        --cus-per-array) CUS_PER_ARRAY="${2:-}"; shift 2 ;;
        --cus-per-array=*) CUS_PER_ARRAY="${1#*=}"; shift ;;
        --no-priority) DO_PRIORITY=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --remove) MODE="remove"; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${NC}" >&2; exit 1 ;;
    esac
done

if ! [[ "$CUS_PER_ARRAY" =~ ^[0-9]+$ ]] || [ "$CUS_PER_ARRAY" -lt 1 ]; then
    echo -e "${RED}--cus-per-array needs a positive integer${NC}" >&2
    exit 1
fi

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}${BOLD}   BC-250 GPU ring-fence for Sunshine                 ${NC}"
echo -e "${BLUE}======================================================${NC}"

# --------------------------------------------------------------------------
# Remove
# --------------------------------------------------------------------------
if [ "$MODE" == "remove" ]; then
    if [ "$DRY_RUN" -eq 0 ]; then
        rm -f "$DROPIN"
        rmdir "$DROPIN_DIR" 2>/dev/null || true
        rm -f "$GAME_WRAPPER"
        # Leave the file capability alone: setcap on /usr/bin/sunshine is a
        # system-wide change and only useful with the drop-in present, but
        # removing it is a privilege decision that belongs to the admin, not
        # to a script that was asked to undo its own edits. Printed instead.
    fi
    echo -e "  -> Removed: $DROPIN"
    echo -e "  -> Removed: $GAME_WRAPPER"
    if command -v setcap >/dev/null 2>&1 && [ -e /usr/bin/sunshine ]; then
        if getcap /usr/bin/sunshine 2>/dev/null | grep -q sys_nice; then
            echo -e "  ${YELLOW}Note: cap_sys_nice is still set on /usr/bin/sunshine. To remove it:${NC}"
            echo -e "        sudo setcap -r cap_sys_nice /usr/bin/sunshine"
        fi
    fi
    if command -v systemctl >/dev/null 2>&1; then
        systemctl --user daemon-reload 2>/dev/null || true
        systemctl --user restart sunshine 2>/dev/null || true
    fi
    echo -e "${GREEN}  ✓ Ring-fence removed.${NC}"
    exit 0
fi

# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------
# Asked of the driver, not guessed: run any VA-API/Vulkan client with
# BC250_CU_REPORT=1 and it prints the numbers, because this driver can query
# them from the same physical device it encodes on. vkcube/vulkaninfo are the
# fallbacks, and if neither works the script stops rather than inventing a
# mask - a wrong AMD_CU_MASK is rejected by Mesa with a message on the game's
# stderr and silently means "all CUs", which is the one outcome that would
# leave the user believing they had fenced something they had not.
read_topology() {
    if command -v vkcube >/dev/null 2>&1; then
        BC250_CU_REPORT=1 vkcube --c 2>&1 | grep -E 'CU topology|ring-fence: game' && return 0
    fi
    if command -v vainfo >/dev/null 2>&1; then
        BC250_CU_REPORT=1 vainfo 2>&1 | grep -E 'CU topology|ring-fence: game' && return 0
    fi
    if command -v vulkaninfo >/dev/null 2>&1; then
        RADV_DEBUG=info vulkaninfo 2>&1 | grep -E 'CU topology|ring-fence: game' && return 0
    fi
    return 1
}

TOPO="$(read_topology 2>/dev/null || true)"
if [ -z "$TOPO" ]; then
    echo -e "${RED}  Could not read the CU topology.${NC}"
    echo -e "  Start a VA-API client with BC250_CU_REPORT=1 and read the line it prints,"
    echo -e "  or run:  RADV_DEBUG=info vulkaninfo 2>&1 | grep -iE 'cu|shader_engine'"
    echo -e "  Nothing has been changed."
    exit 1
fi
echo -e "  -> From the driver:"
echo -e "$TOPO" | sed 's/^/     /'

SE_LINE="$(echo "$TOPO" | grep -m1 'CU topology' || true)"
if ! echo "$SE_LINE" | grep -qE '= *[0-9]+ CUs'; then
    echo -e "${RED}  Topology line not in the expected form: $SE_LINE${NC}"
    echo -e "  Nothing has been changed."
    exit 1
fi
# "  N SE x M SA x K CU/SA = T CUs"
CU_PER_SA="$(echo "$SE_LINE" | sed -E 's/.*x ([0-9]+) CU\/SA.*/\1/')"
TOTAL_CUS="$(echo "$SE_LINE" | sed -E 's/.*= *([0-9]+) CUs.*/\1/')"
if ! [[ "$CU_PER_SA" =~ ^[0-9]+$ ]] || ! [[ "$TOTAL_CUS" =~ ^[0-9]+$ ]] ||
   [ "$CU_PER_SA" -lt 1 ] || [ "$TOTAL_CUS" -lt 1 ]; then
    echo -e "${RED}  Could not parse the topology out of: $SE_LINE${NC}"
    echo -e "  Nothing has been changed."
    exit 1
fi
ARRAYS="$(( TOTAL_CUS / CU_PER_SA ))"

# The masks themselves, in Mesa's own syntax: a list of CUs to ENABLE, within
# one shader array, applied identically to every shader array. See
# set_custom_cu_en_mask() in Mesa's src/amd/common/ac_gpu_info.c.
#
# ⚠️ The split is FOUND BY SEARCH, not computed as "the top K". Mesa requires
# CU2 and CU3 in every mask (PS late-alloc) and at least one of CU0/CU1/CU3/CU4
# (SPI late-alloc), which means a contiguous range is never legal on either
# side: CU2/CU3 can only be in one half of a contiguous split. It also means
# neither half can be smaller than 2 CUs. A mask that breaks any of these is
# rejected with a message and then ignored - the process gets all CUs - so this
# searches for a split Mesa will actually accept, and validates both halves
# before writing anything.
#
# This is the same algorithm as bc250_pick_cu_split() in the driver, which is
# what prints the masks this script consumes.
cu_mask_legal() {   # $@ = the CU indices in the mask
    local m=0 full ge i
    for i in "$@"; do
        [ "$i" -ge "$CU_PER_SA" ] && return 1
        m=$(( m | (1 << i) ))
    done
    [ "$m" -eq 0 ] && return 1
    full=$(( (1 << CU_PER_SA) - 1 ))
    [ $(( m & full )) -eq 0 ] && return 1
    ge=$(( full & ~0x3 ))              # gfx10: everything except CU2/CU3
    [ $(( m & ge )) -eq 0 ] && return 1
    [ $(( full & m & ~ge )) -eq 0 ] && return 1
    return 0
}

# Renders a CU set (array CU_LIST) as Mesa's ID-list syntax:
# comma-separated items, each "N" or "N-M".
format_cu_set() {
    local -a s=("$@")
    local out="" i=0 j n
    while [ "$i" -lt "${#s[@]}" ]; do
        j="$i"
        while [ $(( j + 1 )) -lt "${#s[@]}" && [ "${s[j+1]}" -eq $(( s[j] + 1 )) ]; do
            j=$(( j + 1 ))
        done
        if [ "$j" -eq "$i" ]; then
            out="${out:+$out,}${s[i]}"
        else
            out="${out:+$out,}${s[i]}-${s[j]}"
        fi
        i=$(( j + 1 ))
    done
    echo "$out"
}

# Smallest legal disjoint split giving the encoder $1 CUs per shader array.
# On success sets the global arrays ENC_SET / GAME_SET. Same algorithm, and the
# same answer, as bc250_pick_cu_split() in the driver.
find_split() {
    local want="$1"
    local k start i j p moved
    local -a chosen game in_enc

    ENC_SET=()
    GAME_SET=()
    [ "$CU_PER_SA" -lt 4 ] && return 1

    for (( k = want; k + 2 <= CU_PER_SA; k++ )); do
        for (( start = 0; start + k <= CU_PER_SA; start++ )); do
            chosen=()
            for (( i = 0; i < k; i++ )); do chosen+=( $(( start + i )) ); done

            while :; do
                # Odometer over the k chosen indices: advance the rightmost
                # one that can still move, then cascade the ones after it.
                p=$(( k - 1 ))
                moved=0
                while [ "$p" -ge 0 ]; do
                    if [ "${chosen[p]}" -lt $(( CU_PER_SA - (k - p) )) ]; then
                        chosen[p]=$(( chosen[p] + 1 ))
                        for (( j = p + 1; j < k; j++ )); do
                            chosen[j]=$(( chosen[j-1] + 1 ))
                        done
                        moved=1
                        break
                    fi
                    p=$(( p - 1 ))
                done
                [ "$moved" -eq 0 ] && break

                game=()
                for (( i = 0; i < CU_PER_SA; i++ )); do
                    in_enc=0
                    for (( j = 0; j < k; j++ )); do
                        [ "${chosen[j]}" -eq "$i" ] && in_enc=1
                    done
                    [ "$in_enc" -eq 0 ] && game+=( "$i" )
                done

                if cu_mask_legal "${chosen[@]}" && cu_mask_legal "${game[@]}"; then
                    ENC_SET=( "${chosen[@]}" )
                    GAME_SET=( "${game[@]}" )
                    return 0
                fi
            done
        done
    done
    return 1
}

# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
if [ "$CUS_PER_ARRAY" -ge "$CU_PER_SA" ]; then
    echo -e "${RED}  --cus-per-array $CUS_PER_ARRAY leaves nothing for the game (this part has $CU_PER_SA CUs per shader array).${NC}"
    echo -e "  Nothing has been changed."
    exit 1
fi
if ! find_split "$CUS_PER_ARRAY"; then
    echo -e "${RED}  No legal AMD_CU_MASK split exists for a ${CU_PER_SA} CU/SA part.${NC}"
    echo -e "  Mesa requires CU2/CU3 in *every* mask, so each half needs at least 2 CUs"
    echo -e "  and a contiguous range is never legal. Ring-fencing is not available here;"
    echo -e "  see docs/streaming-ringfence.md for what to do instead. Nothing changed."
    exit 1
fi

ENC_MASK="$(format_cu_set "${ENC_SET[@]}")"
GAME_MASK="$(format_cu_set "${GAME_SET[@]}")"
ENC_N=${#ENC_SET[@]}
GAME_N=${#GAME_SET[@]}

echo
echo -e "  ${BOLD}Planned fence${NC}"
echo -e "     encoder : AMD_CU_MASK=${GREEN}${ENC_MASK}${NC}  ($ENC_N of $CU_PER_SA CU/SA, $(( ENC_N * ARRAYS )) of $TOTAL_CUS CUs)"
echo -e "     game    : AMD_CU_MASK=${GREEN}${GAME_MASK}${NC}  ($GAME_N of $CU_PER_SA CU/SA, $(( GAME_N * ARRAYS )) of $TOTAL_CUS CUs)"
echo -e "     ${GAME_N}CU per shader array is the finest split Mesa will accept on this part:"
echo -e "     CU2/CU3 are required in every mask, so neither half can be a contiguous range."
if [ "$ENC_N" -ne "$CUS_PER_ARRAY" ]; then
    echo -e "     ${YELLOW}Note:${NC} asked for $CUS_PER_ARRAY, got $ENC_N - the search takes the nearest"
    echo -e "           legal size at or above what was asked for."
fi

# --------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------
echo
if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${YELLOW}  --dry-run: nothing was changed.${NC}"
    echo -e "  ${BOLD}Would write${NC} $DROPIN:"
    echo "      [Service]"
    echo "      Environment=AMD_CU_MASK=${ENC_MASK}"
    if [ "$DO_PRIORITY" -eq 1 ]; then
        echo "      Environment=BC250_QUEUE_PRIORITY=high"
        echo "      AmbientCapabilities=CAP_SYS_NICE"
    fi
    echo "      [Service]"
    echo -e "  ${BOLD}Would write${NC} $GAME_WRAPPER (for the game's launch command):"
    echo "      #!/usr/bin/env bash"
    echo "      export AMD_CU_MASK=${GAME_MASK}"
    echo "      exec \"\$@\""
    exit 0
fi

mkdir -p "$DROPIN_DIR"
{
    echo "# Written by apply_gpu_ringfence.sh - BC-250 encoder GPU ring-fence."
    echo "# Topology: $SE_LINE"
    echo "# Mesa reads AMD_CU_MASK when it creates a screen, so this only takes"
    echo "# effect for a Sunshine that starts after this file exists."
    echo "[Service]"
    echo "Environment=AMD_CU_MASK=${ENC_MASK}"
    if [ "$DO_PRIORITY" -eq 1 ]; then
        echo "# BC250_QUEUE_PRIORITY needs CAP_SYS_NICE: VK_EXT_global_priority is"
        echo "# refused to an unprivileged process (VK_ERROR_NOT_PERMITTED_KHR)."
        echo "Environment=BC250_QUEUE_PRIORITY=high"
        echo "AmbientCapabilities=CAP_SYS_NICE"
    fi
} > "$DROPIN"
echo -e "  -> Wrote $DROPIN"
echo -e "     Sunshine will be limited to AMD_CU_MASK=${ENC_MASK} ($CUS_PER_ARRAY CU of $CU_PER_SA per shader array)."

mkdir -p "$GAME_WRAPPER_DIR"
{
    echo "#!/usr/bin/env bash"
    echo "# Written by apply_gpu_ringfence.sh - BC-250 encoder GPU ring-fence."
    echo "# Keep the top $CUS_PER_ARRAY CU of each of the $ARRAYS shader arrays for the"
    echo "# encoder: the game gets the bottom $(( GAME_HI + 1 )) per array."
    echo "export AMD_CU_MASK=${GAME_MASK}"
    echo 'exec "$@"'
} > "$GAME_WRAPPER"
chmod 0755 "$GAME_WRAPPER"
echo -e "  -> Wrote $GAME_WRAPPER"
echo -e "     Launch the game as:  $GAME_WRAPPER <game command>   (Steam: set it as a launch option)"

if [ "$DO_PRIORITY" -eq 1 ]; then
    if command -v setcap >/dev/null 2>&1 && [ -e /usr/bin/sunshine ]; then
        echo -e "  -> Granting cap_sys_nice to /usr/bin/sunshine (needed for the queue priority)"
        sudo setcap cap_sys_nice+ep /usr/bin/sunshine || \
            echo -e "  ${YELLOW}     setcap failed - the priority half will not work, the CU half still will.${NC}"
    else
        echo -e "  ${YELLOW}  /usr/bin/sunshine not found or setcap missing: skipping the priority half.${NC}"
    fi
fi

if command -v systemctl >/dev/null 2>&1; then
    systemctl --user daemon-reload || true
    if systemctl --user is-active --quiet sunshine 2>/dev/null; then
        echo -e "  -> Restarting Sunshine"
        systemctl --user restart sunshine
    else
        echo -e "  ${YELLOW}  Sunshine is not running as a user service; start it to pick this up.${NC}"
    fi
fi

echo
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}${BOLD}   Ring-fence applied                                 ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo
echo "  What this does:"
echo "    * the game can no longer dispatch on the top $CUS_PER_ARRAY CU of any shader array,"
echo "      so the encoder has them to itself; a saturated game stops being able to"
echo "      take every CU on the part."
echo "  What it does not do:"
echo "    * it does not reserve those CUs against anything else that is unmasked -"
echo "      the compositor, gamescope and Sunshine's own capture path are not"
echo "      restricted by this, only the game and Sunshine."
echo "    * it does not create GPU time. The encoder's own cost is ~2.3 ms of shader"
echo "      execution per 1080p frame; under a saturating game most of its frame time"
echo "      is spent waiting to be scheduled (see docs/DEVLOG.md). Masking the game"
echo "      narrows that wait; it cannot remove it."
echo "    * it does not change the encoder's CPU-side cost, which is where a live"
echo "      stream spends most of a contended frame."
echo
echo "  Verify: BC250_GOVERNOR_STATS=1 BC250_CU_REPORT=1 sunshine   (or in its journal)"
