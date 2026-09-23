#!/bin/bash
# Decode H.265 through the driver, on the board, and compare with the
# software decoder byte for byte.
#
#     test_vaapi_hevc.sh [directory del driver]
#
# The harness suites prove the decoder. This proves the road to it: the
# translation from VA's structures into the decoder's own, the surfaces,
# and the picture on its way back out as NV12. A decoder that is right and
# a driver that hands it the wrong parameters look the same from outside.
set -u
DRI="${1:-/tmp/dri}"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

export LIBVA_DRIVERS_PATH="$DRI"
export LIBVA_DRIVER_NAME=bc250

passed=0
failed=0

check() {
    local name="$1" source="$2" count="$3" par="$4" w="$5" h="$6"

    ffmpeg -v error -y -f lavfi -i "$source" -frames:v "$count" \
           -c:v libx265 -x265-params "log-level=none:$par" \
           -pix_fmt yuv420p -f hevc "$T/s.265" 2>/dev/null
    if [ ! -s "$T/s.265" ]; then
        printf '  %-40s ffmpeg produced no stream\n' "$name"
        failed=$((failed + 1)); return
    fi

    ffmpeg -v error -y -i "$T/s.265" -f rawvideo -pix_fmt yuv420p \
           "$T/sw.yuv" 2>/dev/null
    # ⚠️ As VA surfaces, downloaded explicitly: with an ordinary output
    # format ffmpeg falls back to its software decoder whenever the driver
    # declines, and this would compare ffmpeg with itself.
    ffmpeg -v error -y -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 \
           -hwaccel_output_format vaapi -i "$T/s.265" \
           -vf 'hwdownload,format=nv12' \
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
    print("%d byte, maximum %d" % (len(d), max((abs(a[i]-b[i]) for i in d), default=0)))
PY
)"
        failed=$((failed + 1)); return
    fi
    printf '  %-40s identical\n' "$name"
    passed=$((passed + 1))
}

S1="testsrc2=size=176x144:rate=25"
S2="testsrc2=size=320x240:rate=25"

echo "a single picture"
check "intra, qp 28" "$S1" 1 "qp=28" 176 144
check "intra, qp 12" "$S1" 1 "qp=12" 176 144
check "intra, qp 44" "$S1" 1 "qp=44" 176 144

echo
echo "sequences"
check "P only" "$S1" 8 "bframes=0:qp=28" 176 144
check "with B" "$S1" 12 "bframes=3:qp=28" 176 144
check "B pyramid" "$S1" 16 "bframes=3:b-pyramid=1:qp=28" 176 144
check "four references" "$S1" 12 "ref=4:qp=28" 176 144
check "weights" "$S2" 12 "weightp=1:weightb=1:qp=28" 320 240
check "crf" "$S2" 12 "crf=28" 320 240
check "wpp on" "$S1" 12 "wpp=1:qp=28" 176 144
check "wpp off" "$S1" 12 "wpp=0:qp=28" 176 144
check "no deblocking" "$S1" 8 "deblock=false:qp=28" 176 144
check "no sao" "$S1" 8 "sao=0:qp=28" 176 144
check "CTU 16" "$S1" 8 "ctu=16:qp=28" 176 144
check "CTU 32" "$S1" 8 "ctu=32:qp=28" 176 144

echo
echo "sizes"
check "320x240" "$S2" 8 "qp=28" 320 240
check "640x480" "testsrc2=size=640x480:rate=25" 6 "qp=28" 640 480
check "1280x720" "testsrc2=size=1280x720:rate=25" 4 "qp=30" 1280 720

echo
printf 'identical %d, differing %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
