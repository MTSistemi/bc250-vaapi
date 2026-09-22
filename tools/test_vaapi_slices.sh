#!/bin/bash
# Multi-slice HEVC through the driver, on the board.
#
#     test_vaapi_slices.sh [driver directory]
#
# The harness suite proves the decoder handles several slices per
# picture. This proves the road to it, which for slices is its own thing:
# the driver is handed one slice at a time and has always been, so
# everything that belongs to the picture rather than to the slice has to
# survive between those calls.
#
# ⚠️ Needs kvazaar, like its harness twin, and for the same reason: x265
# writes one slice per picture. Skips rather than fails when missing.
set -u
DRI="${1:-/tmp/dri}"

if ! command -v kvazaar >/dev/null 2>&1; then
    echo "kvazaar is not installed - slices cannot be tested, skipping"
    echo "  (apt install kvazaar)"
    exit 0
fi

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

export LIBVA_DRIVERS_PATH="$DRI"
export LIBVA_DRIVER_NAME=bc250

passed=0
failed=0

check() {
    local name="$1" wh="$2" count="$3"
    shift 3

    ffmpeg -v error -y -f lavfi -i "testsrc2=size=$wh:rate=25" \
           -frames:v "$count" -pix_fmt yuv420p -f rawvideo "$T/in.yuv" 2>/dev/null
    kvazaar -i "$T/in.yuv" --input-res "$wh" "$@" -o "$T/s.265" >/dev/null 2>&1
    if [ ! -s "$T/s.265" ]; then
        printf '  %-40s kvazaar produced no stream\n' "$name"
        failed=$((failed + 1)); return
    fi

    ffmpeg -v error -y -i "$T/s.265" -f rawvideo -pix_fmt yuv420p \
           "$T/sw.yuv" 2>/dev/null
    ffmpeg -v error -y -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 \
           -hwaccel_output_format nv12 -i "$T/s.265" \
           -f rawvideo -pix_fmt yuv420p "$T/hw.yuv" 2>"$T/err"

    if [ ! -s "$T/hw.yuv" ]; then
        printf '  %-40s nothing came out: %s\n' "$name" \
               "$(grep -v bc250-gpu "$T/err" | tail -1)"
        failed=$((failed + 1)); return
    fi
    if ! cmp -s "$T/sw.yuv" "$T/hw.yuv"; then
        printf '  %-40s DIFFERENT: %s\n' "$name" \
               "$(python3 - "$T/sw.yuv" "$T/hw.yuv" <<'PY'
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
    printf '  %-40s identical\n' "$name"
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
printf 'identical %d, differing %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
