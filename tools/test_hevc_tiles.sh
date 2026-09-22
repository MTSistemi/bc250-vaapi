#!/bin/bash
# Tiled HEVC, compared with the reference decoder byte for byte.
#
#     test_hevc_tiles.sh [path/hevcps]
#
# ⚠️ This suite needs kvazaar, and it is the only one that does. x265
# cannot write tiles at all - it is wavefront-only - so every other HEVC
# suite here is blind to them by construction, which is how the decoder
# came to walk the picture in raster order for as long as it did.
#
# Debian has kvazaar; if it is missing this skips rather than fails, and
# says so, because a suite that cannot run is not a suite that passed.
#
# ⚠️ Not covered: ten bits. The packaged kvazaar is built for eight, and
# nothing else here writes tiles. The tile code is about scan order and
# availability, both counted in units and coordinates and neither of them
# aware of sample depth, so there is reason to expect it works - but
# expecting is not testing, and this is the gap.
set -u
BIN="${1:-/tmp/hevcps}"

if ! command -v kvazaar >/dev/null 2>&1; then
    echo "kvazaar is not installed - tiles cannot be tested, skipping"
    echo "  (apt install kvazaar)"
    exit 0
fi

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

passed=0
failed=0

# name, WxH, frames, everything else to hand kvazaar
check() {
    local name="$1" wh="$2" count="$3"
    shift 3

    ffmpeg -v error -y -f lavfi -i "testsrc2=size=$wh:rate=25" \
           -frames:v "$count" -pix_fmt yuv420p -f rawvideo "$T/in.yuv" 2>/dev/null
    kvazaar -i "$T/in.yuv" --input-res "$wh" "$@" -o "$T/s.265" >/dev/null 2>&1
    if [ ! -s "$T/s.265" ]; then
        printf '  %-44s kvazaar produced no stream\n' "$name"
        failed=$((failed + 1)); return
    fi

    ffmpeg -v error -y -i "$T/s.265" -f rawvideo -pix_fmt yuv420p \
           "$T/ref.yuv" 2>/dev/null
    "$BIN" -q "$T/s.265" "$T/ours.yuv" >/dev/null 2>&1
    if [ ! -s "$T/ours.yuv" ]; then
        printf '  %-44s nothing came out\n' "$name"
        failed=$((failed + 1)); return
    fi
    if ! cmp -s "$T/ref.yuv" "$T/ours.yuv"; then
        printf '  %-44s DIFFERENT: %s\n' "$name" \
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
    printf '  %-44s identical\n' "$name"
    passed=$((passed + 1))
}

echo "the shapes"
check "2x2" 320x240 4 --tiles 2x2
check "1x4, four rows" 320x240 4 --tiles 1x4
check "4x1, four columns" 320x240 4 --tiles 4x1
check "3x3" 320x240 4 --tiles 3x3
check "2x3" 320x240 4 --tiles 2x3
check "5x4, one unit each" 320x240 4 --tiles 5x4

echo
echo "columns where the encoder puts them"
check "split at 64" 320x240 4 --tiles-width-split 64
check "split at 128" 320x240 4 --tiles-width-split 128
check "split at 64 and 192" 320x240 4 --tiles-width-split 64,192

echo
echo "content and rate"
check "qp 10" 320x240 4 --tiles 2x2 --qp 10
check "qp 40" 320x240 4 --tiles 2x2 --qp 40
check "a GOP with B pictures" 320x240 8 --tiles 2x2 --gop 8
check "no deblocking" 320x240 4 --tiles 2x2 --no-deblock
check "twelve pictures" 320x240 12 --tiles 2x2

echo
echo "sizes"
check "176x144" 176x144 4 --tiles 2x2
check "640x480" 640x480 4 --tiles 2x2
check "1280x720" 1280x720 4 --tiles 3x2
check "58x50, to be cropped" 58x50 4 --tiles 1x1

echo
printf 'identical %d, differing %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
