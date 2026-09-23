#!/bin/bash
# Tiled HEVC through the driver, on the board.
#
#     test_vaapi_tiles.sh [driver directory]
#
# The harness suite proves the decoder walks tiles correctly. This proves
# the road to it, which for tiles is its own thing: the layout does not
# arrive as a parameter set to be parsed, it arrives already taken apart
# in VAPictureParameterBufferHEVC, and the column widths have to be
# carried across by hand.
#
# ⚠️ Needs kvazaar, like its harness twin, and for the same reason: x265
# cannot write tiles. Skips rather than fails when it is missing. Ten bits
# need a kvazaar built for them, KVAZAAR10, as in test_hevc_tiles.sh.
#
# ⚠️ The pictures are taken as VA surfaces and downloaded explicitly. With
# an ordinary output format ffmpeg falls back to its own software decoder
# whenever the driver declines, and the suite then compares ffmpeg with
# itself and passes.
set -u
DRI="${1:-/tmp/dri}"
KVZ=kvazaar
PIX=yuv420p
HW=nv12
DEPTH_ARGS=""

if ! command -v kvazaar >/dev/null 2>&1; then
    echo "kvazaar is not installed - tiles cannot be tested, skipping"
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

    rm -f "$T/s.265" "$T/hw.yuv"
    ffmpeg -v error -y -f lavfi -i "testsrc2=size=$wh:rate=25" \
           -frames:v "$count" -pix_fmt "$PIX" -f rawvideo "$T/in.yuv" 2>/dev/null
    # shellcheck disable=SC2086
    "$KVZ" -i "$T/in.yuv" --input-res "$wh" $DEPTH_ARGS "$@" \
           -o "$T/s.265" >/dev/null 2>&1
    if [ ! -s "$T/s.265" ]; then
        printf '  %-40s kvazaar produced no stream\n' "$name"
        failed=$((failed + 1)); return
    fi

    ffmpeg -v error -y -i "$T/s.265" -f rawvideo -pix_fmt "$PIX" \
           "$T/sw.yuv" 2>/dev/null
    ffmpeg -v error -y -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 \
           -hwaccel_output_format vaapi -i "$T/s.265" \
           -vf "hwdownload,format=$HW" \
           -f rawvideo -pix_fmt "$PIX" "$T/hw.yuv" 2>"$T/err"

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

echo "the shapes"
check "2x2" 320x240 6 --tiles 2x2
check "1x4, four rows" 320x240 6 --tiles 1x4
check "4x1, four columns" 320x240 6 --tiles 4x1
check "3x3" 320x240 6 --tiles 3x3
check "5x4, one unit each" 320x240 6 --tiles 5x4

echo
echo "columns where the encoder puts them"
check "split at 64" 320x240 6 --tiles-width-split 64
check "split at 64 and 192" 320x240 6 --tiles-width-split 64,192

echo
echo "content, rate and size"
check "qp 10" 320x240 6 --tiles 2x2 --qp 10
check "qp 40" 320x240 6 --tiles 2x2 --qp 40
check "a GOP with B pictures" 320x240 8 --tiles 2x2 --gop 8
check "no deblocking" 320x240 6 --tiles 2x2 --no-deblock
check "640x480" 640x480 6 --tiles 2x2
check "1280x720" 1280x720 4 --tiles 3x2

echo
echo "ten bits, on P010 surfaces"
if [ -n "${KVAZAAR10:-}" ] && [ -x "$KVAZAAR10" ]; then
    KVZ="$KVAZAAR10"
    PIX=yuv420p10le
    HW=p010
    DEPTH_ARGS="--input-bitdepth 10"
    check "2x2" 320x240 6 --tiles 2x2
    check "3x3" 320x240 6 --tiles 3x3
    check "split at 64 and 192" 320x240 6 --tiles-width-split 64,192
    check "a GOP with B pictures" 320x240 8 --tiles 2x2 --gop 8
    check "1280x720" 1280x720 4 --tiles 3x2
else
    echo "  no ten-bit kvazaar (set KVAZAAR10) - skipped"
fi

echo
printf 'identical %d, differing %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
