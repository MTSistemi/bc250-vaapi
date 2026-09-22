#!/bin/bash
# Main 10, compared with the reference decoder sample by sample.
#
#     test_hevc_10bit.sh [path/hevcps]
#
# The other pixel suites prove eight bits. This one proves the other
# depth the decoder is built for, over the same ground: intra, inter, B
# pictures, both loop filters, both wavefront settings.
#
# Everything the decoder does above eight bits is a shift or a limit that
# moves with the depth, and every one of them is invisible at eight - a
# byte-sized memset that fills the right value, a quantisation offset
# that happens to be zero. So the comparison here is against ffmpeg on
# yuv420p10le, and it has to be exact.
#
# ⚠️ Samples, not bytes. `cmp` on a ten-bit file reports "byte 1" and
# tells you nothing about how far off you are, so the difference is
# measured on sixteen-bit words below.
set -u
BIN="${1:-/tmp/hevcps}"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

passed=0
failed=0

check() {
    local name="$1" source="$2" count="$3" par="$4"
    local faults="" wpp

    for wpp in 0 1; do
        ffmpeg -v error -y -f lavfi -i "$source" -frames:v "$count" \
               -c:v libx265 -x265-params "log-level=none:wpp=$wpp:$par" \
               -pix_fmt yuv420p10le -f hevc "$T/s.265" 2>/dev/null
        if [ ! -s "$T/s.265" ]; then
            faults="$faults wpp=$wpp:no-stream"; continue
        fi
        ffmpeg -v error -y -i "$T/s.265" -f rawvideo -pix_fmt yuv420p10le \
               "$T/ref.yuv" 2>/dev/null
        "$BIN" -q "$T/s.265" "$T/ours.yuv" >/dev/null 2>&1
        if [ ! -s "$T/ours.yuv" ]; then
            faults="$faults wpp=$wpp:nothing-out"; continue
        fi
        if ! cmp -s "$T/ref.yuv" "$T/ours.yuv"; then
            faults="$faults wpp=$wpp:$(python3 - "$T/ref.yuv" "$T/ours.yuv" <<'PY'
import array
import sys
a = array.array("H"); b = array.array("H")
a.frombytes(open(sys.argv[1], 'rb').read())
b.frombytes(open(sys.argv[2], 'rb').read())
if len(a) != len(b):
    print("lengths-%d-against-%d" % (len(a), len(b)))
else:
    d = [i for i in range(len(a)) if a[i] != b[i]]
    print("%d-samples-max-%d" % (len(d),
                                 max((abs(a[i]-b[i]) for i in d), default=0)))
PY
)"
        fi
    done

    if [ -n "$faults" ]; then
        printf '  %-42s DIFFERENT:%s\n' "$name" "$faults"
        failed=$((failed + 1)); return
    fi
    printf '  %-42s identical\n' "$name"
    passed=$((passed + 1))
}

S1="testsrc2=size=176x144:rate=25"
S2="testsrc2=size=320x240:rate=25"
M="mandelbrot=size=320x240"

echo "intra only"
check "one picture" "$S1" 1 "qp=28"
check "all intra" "$S1" 8 "keyint=1:qp=28"
check "low qp" "$S1" 4 "keyint=1:qp=10"
check "high qp" "$S1" 4 "keyint=1:qp=44"
check "lossless" "$S1" 4 "keyint=1:lossless=1"
check "strong smoothing off" "$S1" 4 "keyint=1:strong-intra-smoothing=0"
check "transform skip" "$S1" 4 "keyint=1:tskip=1:qp=28"

echo
echo "P only"
check "4 pictures" "$S1" 4 "bframes=0:qp=28"
check "12 pictures" "$S1" 12 "bframes=0:qp=28"
check "four references" "$S1" 12 "bframes=0:ref=4:qp=28"
check "no tmvp" "$S1" 12 "bframes=0:temporal-mvp=0:qp=28"
check "with tmvp" "$S1" 12 "bframes=0:temporal-mvp=1:qp=28"
check "weighted P" "$S2" 12 "bframes=0:weightp=1:qp=28"

echo
echo "with B"
check "2 B" "$S1" 12 "bframes=2:qp=28"
check "3 B, pyramid" "$S1" 16 "bframes=3:b-pyramid=1:qp=28"
check "8 B" "$S1" 16 "bframes=8:qp=28"
check "weighted B" "$S2" 16 "bframes=3:weightb=1:qp=28"
check "long GOP" "$S1" 24 "keyint=24:bframes=3:qp=28"

echo
echo "the loop filters"
check "no deblocking" "$S1" 12 "deblock=false:qp=28"
check "deblocking offsets +6/-6" "$S1" 12 "qp=28:deblock=6,-6"
check "no sao" "$S1" 12 "sao=0:qp=28"
check "sao at high qp" "$S1" 12 "qp=44"
check "sao before deblocking" "$S1" 12 "qp=34:sao-non-deblock=1"
check "sao with CTU 16" "$S1" 12 "qp=34:ctu=16"

echo
echo "block sizes"
check "CTU 16" "$S1" 12 "ctu=16:qp=28"
check "CTU 32" "$S1" 12 "ctu=32:qp=28"
check "TU 4" "$S1" 12 "max-tu-size=4:qp=28"
check "AMP on" "$S1" 12 "amp=1:rect=1:qp=28"

echo
echo "rate control and sizes"
check "crf" "$S2" 16 "crf=28"
check "fixed bitrate" "$S2" 16 "bitrate=400"
check "640x480" "testsrc2=size=640x480:rate=25" 8 "qp=28"
check "58x50, to be cropped" "testsrc2=size=58x50:rate=25" 12 "qp=28"
check "mandelbrot" "$M" 12 "crf=20"
check "flat grey" "color=c=gray:size=176x144:rate=25" 8 "qp=28"

echo
printf 'identical %d, differing %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
