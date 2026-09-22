#!/bin/bash
# Decode whole sequences and compare them with the reference decoder,
# byte for byte, every picture.
#
#     test_hevc_inter_pixel.sh [path/hevcps]
#
# The intra suite proves one picture. This proves the rest of them: the
# motion vectors a picture inherits from its neighbours, the samples they
# fetch from pictures already decoded, and - just as easy to get wrong -
# the order the pictures come out in, which for B frames is not the order
# they went in.
#
# Nothing is switched off in the encoder. Both loop filters are on, and so
# is everything x265 does by default.
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
               -pix_fmt yuv420p -f hevc "$T/s.265" 2>/dev/null
        if [ ! -s "$T/s.265" ]; then
            faults="$faults wpp=$wpp:no-stream"; continue
        fi
        ffmpeg -v error -y -i "$T/s.265" -f rawvideo -pix_fmt yuv420p \
               "$T/ref.yuv" 2>/dev/null
        "$BIN" -q "$T/s.265" "$T/ours.yuv" >/dev/null 2>&1
        if [ ! -s "$T/ours.yuv" ]; then
            faults="$faults wpp=$wpp:nothing-out"; continue
        fi
        if ! cmp -s "$T/ref.yuv" "$T/ours.yuv"; then
            faults="$faults wpp=$wpp:$(python3 - "$T/ref.yuv" "$T/ours.yuv" <<'PY'
import sys
a = open(sys.argv[1], 'rb').read()
b = open(sys.argv[2], 'rb').read()
if len(a) != len(b):
    print("lengths-%d-against-%d" % (len(a), len(b)))
else:
    d = [i for i in range(len(a)) if a[i] != b[i]]
    print("%d-byte-max-%d" % (len(d), max((abs(a[i]-b[i]) for i in d), default=0)))
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
M="mandelbrot=size=176x144"

echo "P only"
check "4 pictures" "$S1" 4 "bframes=0:qp=28"
check "12 pictures" "$S1" 12 "bframes=0:qp=28"
check "one reference" "$S1" 12 "bframes=0:ref=1:qp=28"
check "four references" "$S1" 12 "bframes=0:ref=4:qp=28"
check "no tmvp" "$S1" 12 "bframes=0:temporal-mvp=0:qp=28"
check "with tmvp" "$S1" 12 "bframes=0:temporal-mvp=1:qp=28"
check "merge at 1" "$S1" 12 "bframes=0:max-merge=1:qp=28"
check "merge a 5" "$S1" 12 "bframes=0:max-merge=5:qp=28"
check "weights on P" "$S2" 12 "bframes=0:weightp=1:qp=28"
check "no weights" "$S2" 12 "bframes=0:weightp=0:qp=28"

echo
echo "with B"
check "2 B" "$S1" 12 "bframes=2:qp=28"
check "3 B, pyramid" "$S1" 16 "bframes=3:b-pyramid=1:qp=28"
check "3 B, no pyramid" "$S1" 16 "bframes=3:b-pyramid=0:qp=28"
check "8 B" "$S1" 16 "bframes=8:qp=28"
check "weights on B" "$S2" 16 "bframes=3:weightb=1:qp=28"
check "GOP of 8" "$S1" 24 "keyint=8:bframes=3:qp=28"
check "long GOP" "$S1" 24 "keyint=24:bframes=3:qp=28"

echo
echo "partitions and blocks"
check "AMP on" "$S1" 12 "amp=1:rect=1:qp=28"
check "no rectangular partitions" "$S1" 12 "rect=0:amp=0:qp=28"
check "CTU 16" "$S1" 12 "ctu=16:qp=28"
check "CTU 32" "$S1" 12 "ctu=32:qp=28"
check "min CU 16" "$S1" 12 "min-cu-size=16:qp=28"
check "TU inter deep" "$S1" 12 "tu-inter-depth=3:qp=28"

echo
echo "stream control and tools"
check "crf" "$S2" 16 "crf=28"
check "crf, group 16" "$S2" 16 "crf=28:qg-size=16"
check "bitrate fixed" "$S2" 16 "bitrate=400"
check "no deblocking" "$S1" 12 "deblock=false:qp=28"
check "no sao" "$S1" 12 "sao=0:qp=28"
check "lossless" "$S1" 6 "lossless=1"
check "low qp" "$S1" 12 "qp=10"
check "high qp" "$S1" 12 "qp=44"
check "transform skip" "$S1" 12 "tskip=1:qp=28"
check "sign hide off" "$S1" 12 "signhide=0:qp=28"

echo
echo "sizes and contents"
check "320x240" "$S2" 12 "qp=28"
check "640x480" "testsrc2=size=640x480:rate=25" 8 "qp=28"
check "58x50, to be cropped" "testsrc2=size=58x50:rate=25" 12 "qp=28"
check "mandelbrot" "$M" 12 "qp=28"
check "still picture" "color=c=gray:size=176x144:rate=25" 12 "qp=28"

echo
printf 'identical %d, differing %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
