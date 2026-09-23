#!/bin/bash
# Decode through the driver, on the board, and compare against the software
# decoder byte for byte.
#
#     prova_vaapi.sh
#
# prova_conformita.sh exercises the decoder through the standalone harness,
# which feeds it directly. This one goes the whole way round: ffmpeg parses
# the headers, fills in VAPictureParameterBufferH264 and the slice
# parameters, and hands them to the driver through libva. So it tests the
# translation layer - reference lists mapped onto frame store slots, the
# weights, the quantisation matrices, slice_data_bit_offset - which the
# harness cannot reach.
#
# ⚠️ -hwaccel_output_format vaapi, then hwdownload. Asking for nv12 directly
# makes ffmpeg insert a scaler it cannot configure, and the run dies with
# "Error reinitializing filters" long before the driver is involved.
set -u
export LIBVA_DRIVER_NAME=bc250
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

passed=0
failed=0

check() {
    local name="$1"; shift
    local source="$1"; shift

    ffmpeg -v error -y -f lavfi -i "$source" "$@" \
           -pix_fmt yuv420p -f h264 "$T/s.264" 2>/dev/null
    if [ ! -s "$T/s.264" ]; then
        printf '  %-44s ffmpeg produced no stream\n' "$name"
        failed=$((failed + 1)); return
    fi

    ffmpeg -v error -y -i "$T/s.264" -f rawvideo -pix_fmt yuv420p \
           "$T/sw.yuv" 2>/dev/null

    local err
    err=$(ffmpeg -v error -y -hwaccel vaapi -hwaccel_output_format vaapi \
                 -i "$T/s.264" -vf 'hwdownload,format=nv12' \
                 -pix_fmt yuv420p -f rawvideo "$T/hw.yuv" 2>&1 \
          | grep -v '^\[bc250-gpu\]' | head -2)

    if [ ! -s "$T/hw.yuv" ]; then
        printf '  %-44s NOTHING CAME OUT: %s\n' "$name" "$err"
        failed=$((failed + 1)); return
    fi
    if cmp -s "$T/sw.yuv" "$T/hw.yuv"; then
        printf '  %-44s identical  (%s byte)\n' "$name" "$(wc -c < "$T/s.264")"
        passed=$((passed + 1))
    else
        local d
        d=$(python3 - "$T/sw.yuv" "$T/hw.yuv" <<'PY'
import sys
a = open(sys.argv[1], 'rb').read()
b = open(sys.argv[2], 'rb').read()
if len(a) != len(b):
    print("lengths differ: %d against %d" % (len(a), len(b)))
else:
    d = [i for i in range(len(a)) if a[i] != b[i]]
    print("%d byte differing su %d, error max %d"
          % (len(d), len(a), max((abs(a[i]-b[i]) for i in d), default=0)))
PY
)
        printf '  %-44s DIFFERENT: %s\n' "$name" "$d"
        failed=$((failed + 1))
    fi
}

S1="testsrc2=size=176x144:rate=25"
S2="testsrc2=size=320x240:rate=25"

echo "pictures intra"
for qp in 1 18 26 40 51; do
    check "intra qp $qp" "$S1" -frames:v 1 -c:v libx264 -profile:v main -qp $qp
done
check "intra 8x8, high" "$S1" -frames:v 1 -c:v libx264 -profile:v high -qp 20
check "intra CAVLC" "$S1" -frames:v 1 -c:v libx264 -profile:v baseline -qp 26
check "intra, 4 slice" "$S1" -frames:v 1 -c:v libx264 -profile:v main -qp 26 \
    -x264opts slices=4
check "intra, no deblocking" "$S1" -frames:v 1 -c:v libx264 -profile:v main \
    -qp 26 -x264opts no-deblock

echo
echo "sequences P"
check "10 pictures P" "$S1" -frames:v 10 -c:v libx264 -profile:v main -qp 26 -bf 0 -g 30
check "10 pictures P, 3 references" "$S1" -frames:v 10 -c:v libx264 \
    -profile:v main -qp 26 -bf 0 -refs 3 -g 30
check "12 pictures P, high 8x8" "$S1" -frames:v 12 -c:v libx264 \
    -profile:v high -qp 24 -bf 0 -g 6
check "12 pictures P CAVLC" "$S1" -frames:v 12 -c:v libx264 -profile:v high \
    -qp 24 -bf 0 -g 6 -x264opts cabac=0

echo
echo "sequences B"
check "12 pictures B" "$S1" -frames:v 12 -c:v libx264 -profile:v main -qp 26 \
    -bf 2 -g 6 -x264opts b-pyramid=none
check "12 pictures B, pyramid" "$S1" -frames:v 12 -c:v libx264 \
    -profile:v main -qp 26 -bf 2 -g 30
check "12 pictures B, direct temporal" "$S1" -frames:v 12 -c:v libx264 \
    -profile:v main -qp 26 -bf 2 -g 6 -x264opts direct=temporal
check "12 pictures B, weights explicit" "$S1" -frames:v 12 -c:v libx264 \
    -profile:v main -qp 26 -bf 2 -g 4 -x264opts weightp=2:weightb=1
check "12 pictures B CAVLC" "$S1" -frames:v 12 -c:v libx264 -profile:v high \
    -qp 24 -bf 2 -g 6 -x264opts cabac=0:b-pyramid=none

echo
echo "quantisation matrices"
check "intra, matrices JVT" "$S1" -frames:v 1 -c:v libx264 -profile:v high \
    -qp 22 -x264opts cqm=jvt
check "12 pictures, matrices JVT" "$S1" -frames:v 12 -c:v libx264 \
    -profile:v high -qp 24 -bf 2 -g 6 -x264opts cqm=jvt:b-pyramid=none

echo
echo "sizes and loads"
check "320x240, 20 pictures" "$S2" -frames:v 20 -c:v libx264 -profile:v high \
    -preset slow -crf 24 -g 10
check "640x480, 20 pictures" "testsrc2=size=640x480:rate=25" -frames:v 20 \
    -c:v libx264 -profile:v high -preset medium -crf 25 -g 10
check "58x50, edges to be cropped" "testsrc2=size=58x50:rate=25" -frames:v 12 \
    -c:v libx264 -profile:v high -qp 26 -bf 2 -g 6

echo
printf 'passed %d, failed %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
