#!/bin/bash
# The official JCT-VC conformance bitstreams, through the driver.
#
#     test_vaapi_conformance.sh [directory of .zip] [driver directory]
#
# test_hevc_conformance.sh runs them through the decoder directly. This
# runs them the way an application does - ffmpeg, VA-API, our driver -
# and compares with ffmpeg's own software decoder. It covers the part the
# harness cannot: that what VA hands over is translated into what the
# decoder expects, for every tool the vectors exercise. VA-API passes
# the parameter sets already taken apart, the quantisation matrices
# already resolved and the reference lists already built, so every one
# of those is a translation that can be wrong on its own.
#
# ⚠️ The pictures are taken as VA surfaces and downloaded explicitly
# (hwaccel_output_format vaapi, then hwdownload). With an ordinary output
# format ffmpeg falls back to its software decoder whenever the driver
# declines a profile, says nothing about it, and the comparison then
# passes against itself.
set -u
ZIPS="${1:-$HOME/jctvc}"
DRI="${2:-/tmp/dri}"
DEV="${DEV:-/dev/dri/renderD128}"

if [ ! -d "$ZIPS" ]; then
    echo "no bitstreams in $ZIPS - see tools/fetch_conformance.sh"
    exit 0
fi

export LIBVA_DRIVERS_PATH="$DRI"
export LIBVA_DRIVER_NAME=bc250

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# Flat, the way `unzip -j` would, but through python: this runs on the
# board, and a board image has no reason to carry unzip.
extract() {
    python3 - "$1" "$2" <<'PY'
import os, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    for info in z.infolist():
        if info.is_dir():
            continue
        with open(os.path.join(sys.argv[2], os.path.basename(info.filename)),
                  'wb') as f:
            f.write(z.read(info))
PY
}

# ⚠️ When ffmpeg's software decoder disagrees, the archive's own digest
# decides - an exact match of the whole output only, which cannot be a
# false positive. It is taken over the cropped picture, so the hardware
# path is decoded once more with cropping on. SAODBLK_A and SAODBLK_B
# need this: ffmpeg 9.0.2 gets them wrong, and the driver matches the
# published digest.
published_md5_matches() {
    local sum
    sum=$(ffmpeg -v error -hwaccel vaapi -hwaccel_device "$DEV" \
                 -hwaccel_output_format vaapi -i "$1" -noautoscale \
                 -vf "hwdownload,format=$3" -f rawvideo -pix_fmt "$2" - \
                 2>/dev/null | md5sum | cut -d' ' -f1)
    grep -rqsi --include='*md5*' "$sum" "$4"
}

ok=0
refused=0
wrong=0
noref=0

for z in "$ZIPS"/*.zip; do
    [ -e "$z" ] || continue
    name=$(basename "$z" .zip)

    rm -rf "$T/x"; mkdir -p "$T/x"
    extract "$z" "$T/x" 2>/dev/null
    stream=$(find "$T/x" -maxdepth 1 -type f \( -name '*.bit' -o -name '*.bin' \) \
             | head -1)
    if [ -z "$stream" ]; then
        printf '  %-34s no bitstream in the archive\n' "$name"
        noref=$((noref + 1)); continue
    fi

    fmt=$(ffprobe -v error -select_streams v -show_entries stream=pix_fmt \
          -of csv=p=0 "$stream" 2>/dev/null | head -1)
    case "$fmt" in
        yuv420p) hw=nv12 ;;
        yuv420p10le) hw=p010 ;;
        *) printf '  %-34s %s is not a format the driver offers\n' \
                  "$name" "${fmt:-unknown}"
           noref=$((noref + 1)); continue ;;
    esac

    # ⚠️ The whole decoded picture on both sides, conformance window
    # included (-apply_cropping 0). ffmpeg cannot crop a VA surface on the
    # left or at the top, so with cropping on, a stream like CONFWIN_A
    # comes back 414x238 from the hardware path and 412x236 from the
    # software one - a difference in ffmpeg, not in the driver. Comparing
    # the full picture is also the stricter test.
    if ! ffmpeg -v error -y -apply_cropping 0 -i "$stream" -noautoscale \
                -f rawvideo -pix_fmt "$fmt" "$T/sw.yuv" 2>/dev/null \
       || [ ! -s "$T/sw.yuv" ]; then
        printf '  %-34s ffmpeg will not decode it either\n' "$name"
        noref=$((noref + 1)); continue
    fi

    rm -f "$T/hw.yuv"
    ffmpeg -v error -y -hwaccel vaapi -hwaccel_device "$DEV" \
           -hwaccel_output_format vaapi -apply_cropping 0 -i "$stream" \
           -noautoscale \
           -vf "hwdownload,format=$hw" -f rawvideo -pix_fmt "$fmt" \
           "$T/hw.yuv" 2>"$T/err"
    if [ ! -s "$T/hw.yuv" ]; then
        why=$(grep -m1 -oiE 'unsupported[^.]*|not supported[^.]*|failed[^.]*' \
              "$T/err")
        printf '  %-34s REFUSED: %s\n' "$name" "${why:-nothing came out}"
        refused=$((refused + 1)); continue
    fi

    if cmp -s "$T/sw.yuv" "$T/hw.yuv"; then
        printf '  %-34s identical\n' "$name"
        ok=$((ok + 1))
    elif published_md5_matches "$stream" "$fmt" "$hw" "$T/x"; then
        printf '  %-34s identical to the published digest; ffmpeg is not\n' \
               "$name"
        ok=$((ok + 1))
    else
        detail=$(python3 - "$T/sw.yuv" "$T/hw.yuv" <<'PY'
import sys
a = open(sys.argv[1], 'rb').read(); b = open(sys.argv[2], 'rb').read()
if len(a) != len(b):
    print("lengths %d against %d" % (len(a), len(b)))
else:
    d = [i for i in range(len(a)) if a[i] != b[i]]
    print("%d byte of %d, first at %d" % (len(d), len(a), d[0]))
PY
)
        why=$(grep -m1 -oiE 'unsupported[^.]*|failed[^.]*' "$T/err")
        printf '  %-34s DIFFERENT: %s%s\n' "$name" "$detail" "${why:+ ($why)}"
        wrong=$((wrong + 1))
    fi
done

echo
printf 'identical %d, refused %d, differing %d, out of scope %d\n' \
       "$ok" "$refused" "$wrong" "$noref"
[ "$wrong" -eq 0 ]
