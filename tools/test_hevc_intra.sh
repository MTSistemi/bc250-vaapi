#!/bin/bash
# Walk the coding tree of intra slices, and check the slice lands.
#
#     test_hevc_intra.sh [path/hevcps]
#
# Nothing is reconstructed: no prediction, no transform, no samples. What
# is checked is that every bin of the syntax was read against the right
# context, and CABAC makes that checkable without any pixels at all.
#
# ⚠️ A slice read correctly ends exactly where it should - the
# end_of_slice_segment_flag after the last coding tree unit comes back one,
# with the arithmetic decoder at the end of the NAL. A slice read wrongly
# almost never does: it runs out of data, or finishes early with bytes to
# spare, or claims the picture ended in the middle of it. So "did it land"
# is a hard test to pass by accident, and it is available long before
# anything can be compared sample by sample.
#
# ⚠️ Every case runs twice, with wavefront parallelism off and on, because
# it changes the slice data itself: a bit and a byte alignment at the end of
# every coding tree row, and the arithmetic decoder restarted there from a
# snapshot of the row above. A decoder that only ever saw wpp=0 would read
# most real streams wrongly, since x265 turns it on by default.
set -u
BIN="${1:-/tmp/hevcps}"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

passed=0
failed=0

check() {
    local name="$1"; shift
    local source="$1"; shift
    local params="$1"; shift

    local faults="" size="" wpp out lost
    for wpp in 0 1; do
        ffmpeg -v error -y -f lavfi -i "$source" -frames:v 1 \
               -c:v libx265 -x265-params "log-level=none:wpp=$wpp:$params" \
               -pix_fmt yuv420p -f hevc "$T/s.265" 2>/dev/null
        if [ ! -s "$T/s.265" ]; then
            faults="$faults wpp=$wpp:no-stream"
            continue
        fi
        size=$(wc -c < "$T/s.265")
        out=$("$BIN" -q "$T/s.265" 2>&1)
        lost=$(echo "$out" | sed -n 's/.*, \([0-9]*\) lost.*/\1/p')
        if [ "$lost" != "0" ]; then
            faults="$faults wpp=$wpp:$(echo "$out" | grep -m1 '\^' \
                    | sed 's/.*\^ //' | tr ' ' '-')"
        fi
    done
    if [ -n "$faults" ]; then
        printf '  %-44s DOES NOT LAND:%s\n' "$name" "$faults"
        failed=$((failed + 1)); return
    fi
    printf '  %-44s walked  (%s byte)\n' "$name" "$size"
    passed=$((passed + 1))
}

S1="testsrc2=size=176x144:rate=25"
S2="testsrc2=size=320x240:rate=25"

echo "the QP scale"
for qp in 2 8 16 22 28 34 40 51; do
    check "qp $qp" "$S1" "qp=$qp"
done

echo
echo "block sizes"
for ctu in 16 32 64; do
    check "CTU $ctu" "$S1" "ctu=$ctu:qp=28"
done
check "max TU 16" "$S1" "max-tu-size=16:qp=28"
check "TU maximum 8" "$S1" "max-tu-size=8:qp=28"
check "intra TU depth 1" "$S1" "tu-intra-depth=1:qp=28"
check "intra TU depth 3" "$S1" "tu-intra-depth=3:qp=28"
check "CTU 16 e TU 8" "$S1" "ctu=16:max-tu-size=8:qp=28"

echo
echo "tools che change la syntax"
check "no SAO" "$S1" "sao=0:qp=28"
check "SAO on" "$S1" "sao=1:qp=28"
check "sign hide off" "$S1" "signhide=0:qp=28"
check "transform skip" "$S1" "tskip=1:qp=28"
check "strong smoothing off" "$S1" "strong-intra-smoothing=0:qp=28"
check "rd maximum" "$S1" "rd=6:qp=28"
check "cu-lossless" "$S1" "cu-lossless=1:qp=28"

echo
echo "qp che varies inside l'picture"
check "strong aq" "$S2" "aq-mode=2:aq-strength=1.5:qp=28"
check "crf instead of qp" "$S2" "crf=28"
check "low crf" "$S2" "crf=14"

echo
echo "sizes"
check "320x240" "$S2" "qp=28"
check "640x480" "testsrc2=size=640x480:rate=25" "qp=28"
check "58x50, to be cropped" "testsrc2=size=58x50:rate=25" "qp=28"
check "1920x1080" "testsrc2=size=1920x1080:rate=25" "qp=30"
check "picture tiny" "testsrc2=size=32x32:rate=25" "qp=28"

echo
echo "other content"
check "mandelbrot" "mandelbrot=size=320x240" "qp=24"
check "noise" "testsrc2=size=176x144:rate=25" "qp=4"

echo
printf 'walked %d, failed %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
