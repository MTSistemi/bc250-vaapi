#!/bin/bash
# More than one slice per picture, compared with the reference decoder
# byte for byte.
#
#     test_hevc_slices.sh [path/hevcps]
#
# ⚠️ Needs kvazaar, and for the same reason the tile suite does: x265
# writes one slice per picture and nothing else here writes more. Every
# other HEVC suite is therefore blind to the whole question, which is how
# a decoder that erased three quarters of every multi-slice picture
# passed three hundred tests.
#
# Two shapes, and they fail differently:
#
#   --slices tiles : one INDEPENDENT slice per tile. Each carries its own
#                    header and starts its own contexts.
#   --slices wpp   : one DEPENDENT segment per coding tree row. These
#                    carry no header past the address and continue the
#                    arithmetic decoder of the segment before them.
#
# Ten bits need a kvazaar built for them (KVAZAAR10), see
# test_hevc_tiles.sh; without one that section is skipped.
set -u
BIN="${1:-/tmp/hevcps}"
KVZ=kvazaar
PIX=yuv420p
DEPTH_ARGS=""

if ! command -v kvazaar >/dev/null 2>&1; then
    echo "kvazaar is not installed - slices cannot be tested, skipping"
    echo "  (apt install kvazaar)"
    exit 0
fi

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

passed=0
failed=0

check() {
    local name="$1" wh="$2" count="$3"
    shift 3

    rm -f "$T/s.265"
    ffmpeg -v error -y -f lavfi -i "testsrc2=size=$wh:rate=25" \
           -frames:v "$count" -pix_fmt "$PIX" -f rawvideo "$T/in.yuv" 2>/dev/null
    # shellcheck disable=SC2086
    "$KVZ" -i "$T/in.yuv" --input-res "$wh" $DEPTH_ARGS "$@" \
           -o "$T/s.265" >/dev/null 2>&1
    if [ ! -s "$T/s.265" ]; then
        printf '  %-46s kvazaar produced no stream\n' "$name"
        failed=$((failed + 1)); return
    fi

    ffmpeg -v error -y -i "$T/s.265" -f rawvideo -pix_fmt "$PIX" \
           "$T/ref.yuv" 2>/dev/null
    "$BIN" -q "$T/s.265" "$T/ours.yuv" >/dev/null 2>&1
    if [ ! -s "$T/ours.yuv" ]; then
        printf '  %-46s nothing came out\n' "$name"
        failed=$((failed + 1)); return
    fi
    if ! cmp -s "$T/ref.yuv" "$T/ours.yuv"; then
        printf '  %-46s DIFFERENT: %s\n' "$name" \
               "$(python3 - "$T/ref.yuv" "$T/ours.yuv" <<'PY'
import sys
a = open(sys.argv[1], 'rb').read(); b = open(sys.argv[2], 'rb').read()
if len(a) != len(b):
    print("lengths %d against %d" % (len(a), len(b)))
else:
    d = [i for i in range(len(a)) if a[i] != b[i]]
    print("%d byte, first at %d, maximum %d"
          % (len(d), d[0], max(abs(a[i]-b[i]) for i in d)))
PY
)"
        failed=$((failed + 1)); return
    fi
    printf '  %-46s identical\n' "$name"
    passed=$((passed + 1))
}

echo "independent slices, one per tile"
check "2x2, four slices" 320x240 6 --tiles 2x2 --slices tiles
check "1x4, four slices" 320x240 6 --tiles 1x4 --slices tiles
check "4x1, four slices" 320x240 6 --tiles 4x1 --slices tiles
check "3x3, nine slices" 320x240 6 --tiles 3x3 --slices tiles
check "5x4, one unit each" 320x240 6 --tiles 5x4 --slices tiles
check "with B pictures" 320x240 8 --tiles 2x2 --slices tiles --gop 8
check "no deblocking" 320x240 6 --tiles 2x2 --slices tiles --no-deblock
check "qp 40" 320x240 6 --tiles 2x2 --slices tiles --qp 40
check "640x480" 640x480 6 --tiles 2x2 --slices tiles
check "1280x720" 1280x720 4 --tiles 3x2 --slices tiles

echo
echo "dependent segments, one per row"
check "four rows" 320x240 6 --wpp --slices wpp
check "with B pictures" 320x240 8 --wpp --slices wpp --gop 8
check "no deblocking" 320x240 6 --wpp --slices wpp --no-deblock
check "no sao" 320x240 6 --wpp --slices wpp --no-sao
check "qp 10" 320x240 6 --wpp --slices wpp --qp 10
check "qp 40" 320x240 6 --wpp --slices wpp --qp 40
check "640x480, eight rows" 640x480 6 --wpp --slices wpp
check "1280x720, twelve rows" 1280x720 4 --wpp --slices wpp
check "176x144, three rows" 176x144 6 --wpp --slices wpp

echo
echo "ten bits"
if [ -n "${KVAZAAR10:-}" ] && [ -x "$KVAZAAR10" ]; then
    KVZ="$KVAZAAR10"
    PIX=yuv420p10le
    DEPTH_ARGS="--input-bitdepth 10"
    check "2x2, four slices" 320x240 6 --tiles 2x2 --slices tiles
    check "3x3, nine slices" 320x240 6 --tiles 3x3 --slices tiles
    check "slices, with B pictures" 320x240 8 --tiles 2x2 --slices tiles --gop 8
    check "segments, four rows" 320x240 6 --wpp --slices wpp
    check "segments, with B pictures" 320x240 8 --wpp --slices wpp --gop 8
    check "segments, qp 10" 320x240 6 --wpp --slices wpp --qp 10
    check "segments, 1280x720" 1280x720 4 --wpp --slices wpp
else
    echo "  no ten-bit kvazaar (set KVAZAAR10) - skipped"
fi

echo
printf 'identical %d, differing %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
