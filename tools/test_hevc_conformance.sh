#!/bin/bash
# The official JCT-VC conformance bitstreams.
#
#     test_hevc_conformance.sh [directory of .zip] [path/hevcps]
#
# Everything else here compares against the reference decoder on streams
# x265 and kvazaar were asked to make. That is a real test and it is not
# this one: an encoder only emits what it chooses to emit, so a whole
# corner of the standard can stay untouched for three hundred passing
# tests. These bitstreams exist to hit those corners on purpose - each
# one is named after the tool it stresses.
#
# Get them with tools/fetch_conformance.sh, or by hand from
# https://www.itu.int/wftp3/av-arch/jctvc-site/bitstream_exchange/draft_conformance/HEVC_v1/
#
# ⚠️ The comparison is against ffmpeg and not against the .md5 shipped in
# the archive. Those digests are taken over the decoded picture, and the
# archives disagree with each other about whether that means before or
# after the conformance window is applied - so a mismatch there says
# nothing useful on its own. ffmpeg passes this suite; if we match
# ffmpeg, we decode the stream.
#
# ⚠️ A refusal is not the same as a wrong answer and is not reported as
# one. This decoder declines several tools outright, and the point of
# running this is to get the list of which - so the summary groups the
# failures by the reason the decoder gave.
set -u
ZIPS="${1:-$HOME/jctvc}"
BIN="${2:-/tmp/hevcps}"

if [ ! -d "$ZIPS" ]; then
    echo "no bitstreams in $ZIPS - see tools/fetch_conformance.sh"
    exit 0
fi

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# ⚠️ The archive's own digest, as a fallback and only as a fallback. A
# mismatch there proves nothing (see above), but an exact match of the
# whole output can only mean the same bytes - and it is what lets a
# stream ffmpeg itself cannot take be checked at all. VPSSPSPPS_A sends a
# PPS before the SPS it names, which is legal, and ffmpeg rejects the PPS.
published_md5_matches() {
    local sum
    sum=$(md5sum "$2" | cut -d' ' -f1)
    grep -rqsi --include='*md5*' "$sum" "$1"
}

ok=0
refused=0
wrong=0
noref=0
: > "$T/reasons"

for z in "$ZIPS"/*.zip; do
    [ -e "$z" ] || continue
    name=$(basename "$z" .zip)

    rm -rf "$T/x"; mkdir -p "$T/x"
    unzip -o -q -j "$z" -d "$T/x" 2>/dev/null

    stream=$(find "$T/x" -maxdepth 1 -type f \( -name '*.bit' -o -name '*.bin' \) \
             | head -1)
    if [ -z "$stream" ]; then
        printf '  %-34s no bitstream in the archive\n' "$name"
        noref=$((noref + 1)); continue
    fi

    # ffmpeg first: if the reference decoder will not take it either, the
    # stream is outside what this test can say anything about.
    if ! ffmpeg -v error -y -i "$stream" -f rawvideo "$T/ref.yuv" 2>"$T/fferr" \
       || [ ! -s "$T/ref.yuv" ]; then
        printf '  %-34s ffmpeg will not decode it either\n' "$name"
        noref=$((noref + 1)); continue
    fi

    "$BIN" "$stream" "$T/ours.yuv" > "$T/log" 2>&1
    if [ ! -s "$T/ours.yuv" ]; then
        why=$(grep -oE '\^ .*' "$T/log" | sed 's/^\^ //' | sort -u | head -1)
        [ -n "$why" ] || why="nothing came out"
        printf '  %-34s REFUSED: %s\n' "$name" "$why"
        echo "$why" >> "$T/reasons"
        refused=$((refused + 1)); continue
    fi

    if cmp -s "$T/ref.yuv" "$T/ours.yuv"; then
        printf '  %-34s identical\n' "$name"
        ok=$((ok + 1))
    elif published_md5_matches "$T/x" "$T/ours.yuv"; then
        printf '  %-34s identical to the published digest; ffmpeg is not\n' \
               "$name"
        ok=$((ok + 1))
    else
        detail=$(python3 - "$T/ref.yuv" "$T/ours.yuv" <<'PY'
import sys
a = open(sys.argv[1], 'rb').read(); b = open(sys.argv[2], 'rb').read()
if len(a) != len(b):
    print("lengths %d against %d" % (len(a), len(b)))
else:
    d = [i for i in range(len(a)) if a[i] != b[i]]
    print("%d byte of %d, first at %d" % (len(d), len(a), d[0]))
PY
)
        why=$(grep -oE '\^ .*' "$T/log" | sed 's/^\^ //' | sort -u | head -1)
        printf '  %-34s DIFFERENT: %s%s\n' "$name" "$detail" \
               "${why:+ ($why)}"
        echo "${why:-samples differ}" >> "$T/reasons"
        wrong=$((wrong + 1))
    fi
done

echo
printf 'identical %d, refused %d, differing %d, out of scope %d\n' \
       "$ok" "$refused" "$wrong" "$noref"

if [ -s "$T/reasons" ]; then
    echo
    echo "why, most common first:"
    sort "$T/reasons" | uniq -c | sort -rn | sed 's/^/  /'
fi

[ "$wrong" -eq 0 ] && [ "$refused" -eq 0 ]
