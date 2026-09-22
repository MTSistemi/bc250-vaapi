#!/bin/bash
# Do P and B slices land where they should.
set -u
BIN="${1:-/tmp/hevcps}"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

passed=0
failed=0

check() {
    local name="$1" count="$2" source="$3" par="$4"
    local faults="" wpp out
    for wpp in 0 1; do
        ffmpeg -v error -y -f lavfi -i "$source" -frames:v "$count" \
               -c:v libx265 -x265-params "log-level=none:wpp=$wpp:$par" \
               -pix_fmt yuv420p -f hevc "$T/s.265" 2>/dev/null
        if [ ! -s "$T/s.265" ]; then
            faults="$faults wpp=$wpp:no-stream"; continue
        fi
        out=$("$BIN" -q "$T/s.265" 2>&1)
        if ! echo "$out" | grep -q '0 skipped, 0 lost'; then
            faults="$faults wpp=$wpp:[$(echo "$out" | tail -1)]"
        fi
    done
    if [ -n "$faults" ]; then
        printf '  %-40s NO:%s\n' "$name" "$faults"
        failed=$((failed + 1)); return
    fi
    printf '  %-40s walked\n' "$name"
    passed=$((passed + 1))
}

S1="testsrc2=size=176x144:rate=25"
S2="testsrc2=size=320x240:rate=25"

echo "temporal structure"
check "P only, 8 pictures" 8 "$S1" "bframes=0:qp=28"
check "2 B" 8 "$S1" "bframes=2:qp=28"
check "4 B, pyramid" 16 "$S1" "bframes=4:b-pyramid=1:qp=28"
check "8 B" 16 "$S1" "bframes=8:qp=28"
check "long GOP" 24 "$S1" "keyint=240:qp=28"
check "every picture intra" 8 "$S1" "keyint=1:qp=28"

echo
echo "partitions"
check "AMP on" 8 "$S1" "amp=1:rect=1:qp=28"
check "AMP off" 8 "$S1" "amp=0:rect=1:qp=28"
check "no rectangular partitions" 8 "$S1" "rect=0:amp=0:qp=28"
check "CTU 16" 8 "$S1" "ctu=16:rect=1:amp=1:qp=28"
check "CTU 32" 8 "$S1" "ctu=32:rect=1:amp=1:qp=28"
check "min CU 8" 8 "$S1" "min-cu-size=8:rect=1:amp=1:qp=28"
check "min CU 32" 8 "$S1" "min-cu-size=32:qp=28"

echo
echo "motion e references"
check "4 references" 12 "$S2" "ref=4:qp=28"
check "1 reference" 12 "$S2" "ref=1:qp=28"
check "no tmvp" 8 "$S1" "temporal-mvp=0:qp=28"
check "with tmvp" 8 "$S1" "temporal-mvp=1:qp=28"
check "merge a 2" 8 "$S1" "max-merge=2:qp=28"
check "merge a 5" 8 "$S1" "max-merge=5:qp=28"
check "search wide" 8 "$S2" "me=star:merange=57:qp=28"
check "weights on P" 12 "$S2" "weightp=1:qp=28"
check "weights on P and B" 12 "$S2" "weightp=1:weightb=1:qp=28"

echo
echo "tools"
check "transform skip" 8 "$S1" "tskip=1:qp=28"
check "sign hide off" 8 "$S1" "signhide=0:qp=28"
check "crf" 12 "$S2" "crf=28"
check "crf, group 16" 12 "$S2" "crf=28:qg-size=16"
check "bitrate fixed" 12 "$S2" "bitrate=300"
check "lossless" 6 "$S1" "lossless=1"
check "no deblocking" 8 "$S1" "deblock=false:qp=28"
check "no sao" 8 "$S1" "sao=0:qp=28"
check "low qp" 8 "$S1" "qp=8"
check "high qp" 8 "$S1" "qp=45"
check "640x480" 8 "testsrc2=size=640x480:rate=25" "qp=28"
check "58x50, to be cropped" 8 "testsrc2=size=58x50:rate=25" "qp=28"
check "mandelbrot in motion" 8 "mandelbrot=size=320x240" "qp=28"

echo
printf 'walked %d, failed %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
