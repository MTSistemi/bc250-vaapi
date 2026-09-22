#!/bin/bash
# Post-processing through the driver, on the board.
#
#     test_vaapi_vpp.sh [driver directory]
#
# ⚠️ This is the only suite here that does not compare byte for byte, and
# it cannot be. Two scalers that both do the right thing still disagree
# in the last bit, because "the right thing" is a filter shape and there
# is more than one defensible one. So the scaling cases are measured in
# dB against ffmpeg's own scalers, with a floor well under what was
# measured, and a case that drops below it means something broke rather
# than something drifted.
#
# The real correctness anchor is the 1:1 case, which IS byte for byte. At
# a ratio of one the footprint loop runs once with a zero offset and the
# sample lands exactly on a source pixel, so a scaler that gets the plane
# addressing, the chroma mapping or the half-sample offset wrong cannot
# produce the input back unchanged. Both depths are checked that way.
#
# ⚠️ Every case is fed from ONE generated file, deliberately. Feeding the
# two paths from two separate lavfi runs compares two different pictures:
# some generators are seeded per run, and this suite briefly "measured"
# 11 dB that way before anyone noticed it was comparing noise to noise.
#
# ⚠️ Not covered: source cropping. VAProcPipelineParameterBuffer's
# surface_region is implemented and validated in the driver, but none of
# ffmpeg's VAAPI filters ever sets it, so there is no way to reach it
# from here. It needs a program written against libva directly.
set -u
DRI="${1:-/tmp/dri}"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

export LIBVA_DRIVERS_PATH="$DRI"
export LIBVA_DRIVER_NAME=bc250
DEV=/dev/dri/renderD128

passed=0
failed=0

# The floors. Measured on a BC-250 against a hard synthetic pattern:
# 2:1 down 70.9 dB, 3:1 and 4:1 down about 40, every enlargement 68 dB or
# better. Set low enough that filter-shape noise cannot trip them.
FLOOR_DOWN=35
FLOOR_UP=60

psnr_of() {
    ffmpeg -v info -s "$3" -pix_fmt "$1" -i "$T/ours.yuv" \
           -s "$3" -pix_fmt "$1" -i "$T/ref.yuv" \
           -lavfi psnr -f null - 2>&1 | grep -o 'average:[0-9.]*' \
           | head -1 | cut -d: -f2
}

# name, pixel format, source file, WxH, ffmpeg scaler to compare with, floor
check() {
    local name="$1" pf="$2" base="$3" wh="$4" flag="$5" floor="$6"
    local w="${wh%x*}" h="${wh#*x}"

    ffmpeg -v error -y -s 640x480 -pix_fmt "$pf" -i "$base" \
           -vf "scale=$w:$h:flags=$flag,format=$pf" \
           -f rawvideo "$T/ref.yuv" 2>/dev/null
    ffmpeg -v error -y -vaapi_device "$DEV" -s 640x480 -pix_fmt "$pf" -i "$base" \
           -vf "hwupload,scale_vaapi=$w:$h,hwdownload,format=$pf" \
           -f rawvideo "$T/ours.yuv" 2>"$T/err"

    if [ ! -s "$T/ours.yuv" ]; then
        printf '  %-40s nothing came out: %s\n' "$name" \
               "$(grep -v bc250-gpu "$T/err" | tail -1)"
        failed=$((failed + 1)); return
    fi

    local db
    db=$(psnr_of "$pf" "$base" "$wh")
    if [ -z "$db" ]; then
        printf '  %-40s no measurement\n' "$name"
        failed=$((failed + 1)); return
    fi
    if awk -v a="$db" -v b="$floor" 'BEGIN{exit !(a < b)}'; then
        printf '  %-40s TOO FAR: %s dB, floor %s\n' "$name" "$db" "$floor"
        failed=$((failed + 1)); return
    fi
    printf '  %-40s %s dB\n' "$name" "$db"
    passed=$((passed + 1))
}

# name, pixel format, source file: the same size in and out, byte for byte
check_exact() {
    local name="$1" pf="$2" base="$3"

    ffmpeg -v error -y -vaapi_device "$DEV" -s 640x480 -pix_fmt "$pf" -i "$base" \
           -vf "hwupload,scale_vaapi=640:480,hwdownload,format=$pf" \
           -f rawvideo "$T/ours.yuv" 2>"$T/err"

    if [ ! -s "$T/ours.yuv" ]; then
        printf '  %-40s nothing came out: %s\n' "$name" \
               "$(grep -v bc250-gpu "$T/err" | tail -1)"
        failed=$((failed + 1)); return
    fi
    if ! cmp -s "$base" "$T/ours.yuv"; then
        printf '  %-40s NOT IDENTICAL\n' "$name"
        failed=$((failed + 1)); return
    fi
    printf '  %-40s identical\n' "$name"
    passed=$((passed + 1))
}

ffmpeg -v error -y -f lavfi -i 'testsrc2=size=640x480:rate=1' -frames:v 1 \
       -pix_fmt nv12 -f rawvideo "$T/src8.yuv" 2>/dev/null
ffmpeg -v error -y -f lavfi -i 'testsrc2=size=640x480:rate=1' -frames:v 1 \
       -pix_fmt p010le -f rawvideo "$T/src10.yuv" 2>/dev/null
if [ ! -s "$T/src8.yuv" ] || [ ! -s "$T/src10.yuv" ]; then
    echo "ffmpeg produced no source"
    exit 1
fi

echo "the same size in and out, which has to be exact"
check_exact "NV12 640x480 -> 640x480" nv12 "$T/src8.yuv"
check_exact "P010 640x480 -> 640x480" p010le "$T/src10.yuv"

echo
echo "shrinking, against the area filter"
check "NV12 -> 320x240" nv12 "$T/src8.yuv" 320x240 area "$FLOOR_DOWN"
check "NV12 -> 214x160" nv12 "$T/src8.yuv" 214x160 area "$FLOOR_DOWN"
check "NV12 -> 160x120" nv12 "$T/src8.yuv" 160x120 area "$FLOOR_DOWN"
check "P010 -> 320x240" p010le "$T/src10.yuv" 320x240 area "$FLOOR_DOWN"
check "P010 -> 160x120" p010le "$T/src10.yuv" 160x120 area "$FLOOR_DOWN"

echo
echo "enlarging, against the bilinear filter"
check "NV12 -> 800x600" nv12 "$T/src8.yuv" 800x600 bilinear "$FLOOR_UP"
check "NV12 -> 960x720" nv12 "$T/src8.yuv" 960x720 bilinear "$FLOOR_UP"
check "NV12 -> 1280x960" nv12 "$T/src8.yuv" 1280x960 bilinear "$FLOOR_UP"
check "P010 -> 1280x960" p010le "$T/src10.yuv" 1280x960 bilinear "$FLOOR_UP"

echo
printf 'passed %d, failed %d\n' "$passed" "$failed"
[ "$failed" -eq 0 ]
