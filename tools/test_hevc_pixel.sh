#!/bin/bash
# Decode intra pictures and compare them with the reference decoder, byte
# for byte.
#
#     test_hevc_pixel.sh [path/hevcps]
#
# H.265 decoding is exact arithmetic, like H.264's: a conformant decoder
# produces the same samples as every other one, to the bit. So the only
# useful pass mark is "identical", and any difference at all is a bug
# however small it looks.
#
# Nothing is turned off in the streams any more: both loop filters are
# in, so the encoder is left to use whatever it wants. A comparison run
# against a stream with the hard parts switched off says very little, and
# the temptation to leave such a switch in place after the feature lands
# is the reason to write this down.
#
# ⚠️ To turn deblocking off in x265 the switch is deblock=false. deblock=0
# sets the filter's beta and tC offsets to zero and leaves it running,
# which is a different thing entirely: it cost half an hour of chasing
# differences that all sat on eight-sample boundaries and looked like a
# prediction bug. x264 has the same trap with -x264-params deblock=0.
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

    local faults="" size="" wpp
    for wpp in 0 1; do
        ffmpeg -v error -y -f lavfi -i "$source" -frames:v 1 -c:v libx265 \
               -x265-params "log-level=none:wpp=$wpp:$params" \
               -pix_fmt yuv420p -f hevc "$T/s.265" 2>/dev/null
        if [ ! -s "$T/s.265" ]; then
            faults="$faults wpp=$wpp:no-stream"
            continue
        fi
        size=$(wc -c < "$T/s.265")
        ffmpeg -v error -y -i "$T/s.265" -f rawvideo -pix_fmt yuv420p \
               "$T/ref.yuv" 2>/dev/null
        "$BIN" -q "$T/s.265" "$T/ours.yuv" >/dev/null 2>&1
        if [ ! -s "$T/ours.yuv" ]; then
            faults="$faults wpp=$wpp:nothing-out"
            continue
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
        printf '  %-44s DIFFERENT:%s\n' "$name" "$faults"
        failed=$((failed + 1)); return
    fi
    printf '  %-44s identical  (%s byte)\n' "$name" "$size"
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
check "TU maximum 4" "$S1" "max-tu-size=4:qp=28"
check "intra TU depth 1" "$S1" "tu-intra-depth=1:qp=28"
check "intra TU depth 4" "$S1" "tu-intra-depth=4:qp=28"
check "CTU 16, TU 8" "$S1" "ctu=16:max-tu-size=8:qp=28"

echo
echo "tools"
check "sign hide off" "$S1" "signhide=0:qp=28"
check "transform skip" "$S1" "tskip=1:qp=28"
check "strong smoothing off" "$S1" "strong-intra-smoothing=0:qp=28"
check "strong smoothing on" "$S1" "strong-intra-smoothing=1:qp=28"
check "maximum rd" "$S1" "rd=6:qp=28"
check "strong aq" "$S2" "aq-mode=2:aq-strength=1.5:qp=28"
check "rdoq off" "$S1" "rdoq-level=0:qp=28"
check "strong psy-rd" "$S1" "psy-rd=4.0:qp=28"
check "lossless" "$S1" "lossless=1"

echo
echo "the deblocking filter"
check "filter off" "$S1" "qp=28:deblock=false"
check "offsets +3/+3" "$S1" "qp=28:deblock=3,3"
check "offsets -3/-3" "$S1" "qp=28:deblock=-3,-3"
check "offsets +6/-6" "$S1" "qp=28:deblock=6,-6"
check "offsets -6/+6" "$S1" "qp=28:deblock=-6,6"
check "filter at high qp" "$S1" "qp=48"
check "filter at low qp" "$S1" "qp=6"
check "filter with CTU 16" "$S1" "qp=34:ctu=16"
check "filter with TU 4" "$S1" "qp=34:max-tu-size=4"

echo
echo "the sample adaptive offset"
check "sao off" "$S1" "qp=28:sao=0"
check "sao at qp 34" "$S1" "qp=34"
check "sao at qp 44" "$S1" "qp=44"
check "sao before deblocking" "$S1" "qp=34:sao-non-deblock=1"
check "sao with CTU 16" "$S1" "qp=34:ctu=16"
check "sao with CTU 32" "$S1" "qp=34:ctu=32"
check "sao, offset limit" "$S1" "qp=44:sao-lookahead-depth=0"
check "sao on mandelbrot" "mandelbrot=size=320x240" "qp=32"

echo
# ⚠️ Rate control is where the quantisation parameter stops standing still.
# A decoder can be byte-exact on every constant-QP stream in the world and
# still be wrong here, because only here does the parameter get predicted
# from the neighbours rather than read from the slice header.
echo "stream control, one QP per group"
check "crf" "$S2" "crf=28"
check "low crf" "$S2" "crf=12"
check "crf, group 32" "$S2" "crf=28:qg-size=32"
check "crf, group 16" "$S2" "crf=28:qg-size=16"
check "crf, CTU 32" "$S2" "crf=28:ctu=32"
check "crf, CTU 16" "$S2" "crf=28:ctu=16"
check "crf 1920x1080" "testsrc2=size=1920x1080:rate=25" "crf=30"
check "crf mandelbrot" "mandelbrot=size=320x240" "crf=20"
check "bitrate fixed" "$S2" "bitrate=300"
check "aq 3 with crf" "$S2" "crf=28:aq-mode=3"

echo
echo "sizes"
check "320x240" "$S2" "qp=28"
check "640x480" "testsrc2=size=640x480:rate=25" "qp=28"
check "58x50, to be cropped" "testsrc2=size=58x50:rate=25" "qp=28"
check "1920x1080" "testsrc2=size=1920x1080:rate=25" "qp=30"
check "32x32" "testsrc2=size=32x32:rate=25" "qp=28"

echo
echo "other content"
check "mandelbrot" "mandelbrot=size=320x240" "qp=24"
check "almost noise" "$S1" "qp=4"
check "flat grey" "color=c=gray:size=176x144" "qp=28"

echo
printf 'identical %d, differing %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
