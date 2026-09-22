#!/bin/bash
# Decode a spread of streams and compare against the reference decoder, byte
# for byte.
#
#     prova_conformita.sh [path/h264dec]
#
# H.264 decoding is exact arithmetic: a conformant decoder produces the same
# samples as every other conformant decoder, to the bit. So the only useful
# pass mark is "identical", and any difference at all is a bug, however
# small it looks.
#
# ⚠️ -x264opts no-deblock, not -x264-params deblock=0: the second is accepted
# and silently ignored, which produced two byte-identical streams and a very
# confusing half hour.
set -u
DEC="${1:-/tmp/h264dec}"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

passed=0
failed=0
skipped=0

check() {
    local name="$1"; shift
    local source="$1"; shift
    local dim="$1"; shift

    ffmpeg -v error -y -f lavfi -i "$source" "$@" \
           -pix_fmt yuv420p -f h264 "$T/s.264" 2>/dev/null
    if [ ! -s "$T/s.264" ]; then
        printf '  %-44s ffmpeg produced no stream\n' "$name"
        skipped=$((skipped + 1)); return
    fi

    ffmpeg -v error -y -i "$T/s.264" -f rawvideo -pix_fmt yuv420p "$T/ref.yuv" 2>/dev/null
    local out err
    out=$("$DEC" "$T/s.264" "$T/ours.yuv" 2>&1)
    local rc=$?

    if [ $rc -eq 3 ]; then
        printf '  %-44s outside coverage: %s\n' "$name" "$(echo "$out" | head -1)"
        skipped=$((skipped + 1)); return
    fi
    if [ $rc -ne 0 ]; then
        printf '  %-44s REFUSED (%d) %s\n' "$name" "$rc" "$(echo "$out" | head -1)"
        failed=$((failed + 1)); return
    fi

    if cmp -s "$T/ref.yuv" "$T/ours.yuv"; then
        printf '  %-44s identical  (%s, %s byte)\n' "$name" "$dim" "$(wc -c < "$T/s.264")"
        passed=$((passed + 1))
    else
        err=$(python3 - "$T/ref.yuv" "$T/ours.yuv" <<'PY'
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
        printf '  %-44s DIFFERENT: %s\n' "$name" "$err"
        failed=$((failed + 1))
    fi
}

SRC1="testsrc2=size=176x144:rate=25"
SRC2="testsrc2=size=320x240:rate=25"
SRC3="testsrc2=size=64x64:rate=25"

echo "un picture intra, several QP e profiles"
# qp 0 is absent on purpose: libx264 encodes it losslessly, which is
# High 4:4:4 Predictive, a profile this decoder refuses up front.
for qp in 1 10 18 26 34 44 51; do
    check "intra qp $qp, main" "$SRC1" 176x144 \
        -frames:v 1 -c:v libx264 -profile:v main -qp $qp
done

echo
echo "transform 8x8 (profile high)"
for qp in 12 22 32; do
    check "intra 8x8 qp $qp" "$SRC1" 176x144 \
        -frames:v 1 -c:v libx264 -profile:v high -qp $qp
done

echo
echo "no deblocking filter"
check "intra no-deblock" "$SRC1" 176x144 \
    -frames:v 1 -c:v libx264 -profile:v main -qp 26 -x264opts no-deblock

echo
echo "sizes that are not multiples of 16"
check "58x50, edges to be cropped" "testsrc2=size=58x50:rate=25" 58x50 \
    -frames:v 1 -c:v libx264 -profile:v main -qp 24
check "320x240" "$SRC2" 320x240 \
    -frames:v 1 -c:v libx264 -profile:v main -qp 24

echo
echo "piu' slice per picture"
check "4 slice" "$SRC1" 176x144 \
    -frames:v 1 -c:v libx264 -profile:v main -qp 26 -x264opts slices=4

echo
echo "sequences with P pictures"
for n in 2 5 15; do
    check "$n pictures, no B" "$SRC1" 176x144         -frames:v $n -c:v libx264 -profile:v main -qp 26 -bf 0 -g 30
done
check "10 pictures, 3 references" "$SRC1" 176x144     -frames:v 10 -c:v libx264 -profile:v main -qp 26 -bf 0 -refs 3 -g 30
check "10 pictures, high 8x8" "$SRC1" 176x144     -frames:v 10 -c:v libx264 -profile:v high -qp 26 -bf 0 -g 30
check "10 pictures, 4 slice" "$SRC1" 176x144     -frames:v 10 -c:v libx264 -profile:v main -qp 26 -bf 0 -g 30 -x264opts slices=4
check "20 pictures, crf e preset slow" "$SRC1" 176x144     -frames:v 20 -c:v libx264 -profile:v high -preset slow -crf 26 -bf 0 -g 8
check "12 pictures 320x240, crf" "$SRC2" 320x240     -frames:v 12 -c:v libx264 -profile:v high -crf 24 -bf 0 -g 6

echo
echo "pictures B"
check "9 pictures, 2 B, no pyramid" "$SRC1" 176x144     -frames:v 9 -c:v libx264 -profile:v main -qp 26 -bf 2 -g 30 -x264opts b-pyramid=none
check "16 pictures, 3 B, no pyramid" "$SRC1" 176x144     -frames:v 16 -c:v libx264 -profile:v main -qp 24 -bf 3 -g 8 -x264opts b-pyramid=none
check "12 pictures B, high 8x8" "$SRC1" 176x144     -frames:v 12 -c:v libx264 -profile:v high -qp 26 -bf 2 -g 6 -x264opts b-pyramid=none
check "12 pictures B, 320x240" "$SRC2" 320x240     -frames:v 12 -c:v libx264 -profile:v high -qp 24 -bf 2 -g 6 -x264opts b-pyramid=none
check "10 pictures, pyramid B" "$SRC1" 176x144     -frames:v 10 -c:v libx264 -profile:v main -qp 26 -bf 2 -g 30
check "20 pictures, every x264 default" "$SRC1" 176x144     -frames:v 20 -c:v libx264 -profile:v high -preset slow -crf 25 -g 10

echo
echo "CAVLC"
NC="-x264opts cabac=0"
check "intra CAVLC" "$SRC1" 176x144 \
    -frames:v 1 -c:v libx264 -profile:v baseline -qp 26
check "10 pictures CAVLC" "$SRC1" 176x144 \
    -frames:v 10 -c:v libx264 -profile:v baseline -qp 26 -g 30

# Low QP means big coefficients, and big coefficients are the only way to
# reach the level escape codes: level_prefix 15 and up, where the suffix
# length stops following suffixLength and starts following the prefix.
for qp in 1 8 16 34 51; do
    check "intra CAVLC qp $qp" "$SRC1" 176x144 \
        -frames:v 1 -c:v libx264 -profile:v main -qp $qp $NC
done

check "intra 8x8 CAVLC" "$SRC1" 176x144 \
    -frames:v 1 -c:v libx264 -profile:v high -qp 18 $NC
check "58x50 CAVLC" "testsrc2=size=58x50:rate=25" 58x50 \
    -frames:v 1 -c:v libx264 -profile:v main -qp 24 $NC
check "4 slice CAVLC" "$SRC1" 176x144 \
    -frames:v 1 -c:v libx264 -profile:v main -qp 26 -x264opts cabac=0:slices=4
check "intra CAVLC no-deblock" "$SRC1" 176x144 \
    -frames:v 1 -c:v libx264 -profile:v main -qp 26 -x264opts cabac=0:no-deblock

check "10 pictures CAVLC, 3 references" "$SRC1" 176x144 \
    -frames:v 10 -c:v libx264 -profile:v main -qp 26 -bf 0 -refs 3 -g 30 $NC
check "12 pictures CAVLC, high 8x8" "$SRC1" 176x144 \
    -frames:v 12 -c:v libx264 -profile:v high -qp 24 -bf 0 -g 6 $NC
check "20 pictures CAVLC, preset slow" "$SRC1" 176x144 \
    -frames:v 20 -c:v libx264 -profile:v high -preset slow -crf 26 -bf 0 -g 8 $NC
check "12 pictures CAVLC 320x240" "$SRC2" 320x240 \
    -frames:v 12 -c:v libx264 -profile:v high -crf 24 -bf 0 -g 6 $NC

check "9 pictures CAVLC, 2 B" "$SRC1" 176x144 \
    -frames:v 9 -c:v libx264 -profile:v main -qp 26 -bf 2 -g 30 \
    -x264opts cabac=0:b-pyramid=none
check "12 pictures CAVLC B, high 8x8" "$SRC1" 176x144 \
    -frames:v 12 -c:v libx264 -profile:v high -qp 26 -bf 2 -g 6 \
    -x264opts cabac=0:b-pyramid=none
check "10 pictures CAVLC, pyramid B" "$SRC1" 176x144 \
    -frames:v 10 -c:v libx264 -profile:v main -qp 26 -bf 2 -g 30 $NC
check "12 pictures CAVLC B, 320x240" "$SRC2" 320x240 \
    -frames:v 12 -c:v libx264 -profile:v high -qp 24 -bf 2 -g 6 \
    -x264opts cabac=0:b-pyramid=none
check "8 pictures CAVLC, weights explicit" "$SRC1" 176x144 \
    -frames:v 8 -c:v libx264 -profile:v main -qp 26 -bf 2 -g 4 \
    -x264opts cabac=0:weightp=2:weightb=1:b-pyramid=none

echo
echo "direct temporal"
TD="-x264opts direct=temporal"
check "9 pictures, direct temporal" "$SRC1" 176x144 \
    -frames:v 9 -c:v libx264 -profile:v main -qp 26 -bf 2 -g 30 \
    -x264opts direct=temporal:b-pyramid=none
check "16 pictures, 3 B, direct temporal" "$SRC1" 176x144 \
    -frames:v 16 -c:v libx264 -profile:v main -qp 24 -bf 3 -g 8 \
    -x264opts direct=temporal:b-pyramid=none
check "12 pictures temporal, high 8x8" "$SRC1" 176x144 \
    -frames:v 12 -c:v libx264 -profile:v high -qp 26 -bf 2 -g 6 \
    -x264opts direct=temporal:b-pyramid=none
check "10 pictures temporal, pyramid B" "$SRC1" 176x144 \
    -frames:v 10 -c:v libx264 -profile:v main -qp 26 -bf 2 -g 30 $TD
check "20 pictures temporal, pyramid e 3 rif" "$SRC1" 176x144 \
    -frames:v 20 -c:v libx264 -profile:v high -qp 24 -bf 3 -refs 3 -g 10 $TD
check "12 pictures temporal, 320x240" "$SRC2" 320x240 \
    -frames:v 12 -c:v libx264 -profile:v high -crf 24 -bf 2 -g 6 $TD
check "12 pictures temporal, weights implicit" "$SRC1" 176x144 \
    -frames:v 12 -c:v libx264 -profile:v main -qp 26 -bf 2 -g 6 \
    -x264opts direct=temporal:weightb=1
check "12 pictures temporal CAVLC" "$SRC1" 176x144 \
    -frames:v 12 -c:v libx264 -profile:v main -qp 26 -bf 2 -g 6 \
    -x264opts direct=temporal:cabac=0
check "16 pictures temporal CAVLC, pyramid" "$SRC1" 176x144 \
    -frames:v 16 -c:v libx264 -profile:v high -qp 24 -bf 3 -g 8 \
    -x264opts direct=temporal:cabac=0
check "20 pictures temporal, preset slow" "$SRC1" 176x144 \
    -frames:v 20 -c:v libx264 -profile:v high -preset slow -crf 25 -g 10 $TD
check "12 pictures temporal, 4 slice" "$SRC1" 176x144 \
    -frames:v 12 -c:v libx264 -profile:v main -qp 26 -bf 2 -g 6 \
    -x264opts direct=temporal:slices=4
check "12 pictures temporal, 58x50" "testsrc2=size=58x50:rate=25" 58x50 \
    -frames:v 12 -c:v libx264 -profile:v high -qp 26 -bf 2 -g 6 $TD

echo
echo "quantisation matrices"

# A distinct value in every position: a list kept in the wrong order then
# cannot come out right by accident. The 4x4 lists run 8..38 and the 8x8
# ones 8..71, both in the raster order x264's file format expects.
cqm_file() {
    local f="$1"
    { for name in INTRA4X4_LUMA INTRA4X4_CHROMAU INTRA4X4_CHROMAV \
                  INTER4X4_LUMA INTER4X4_CHROMAU INTER4X4_CHROMAV; do
          echo "$name"
          for r in 0 1 2 3; do
              for c in 0 1 2 3; do printf ' %d' $((8 + 2 * (r * 4 + c))); done
              echo
          done
      done
      for name in INTRA8X8_LUMA INTER8X8_LUMA; do
          echo "$name"
          for r in 0 1 2 3 4 5 6 7; do
              for c in 0 1 2 3 4 5 6 7; do printf ' %d' $((8 + r * 8 + c)); done
              echo
          done
      done
    } > "$f"
}
cqm_file "$T/cqm.txt"

check "intra, matrices JVT" "$SRC1" 176x144 \
    -frames:v 1 -c:v libx264 -profile:v high -qp 22 -x264opts cqm=jvt
check "12 pictures, matrices JVT" "$SRC1" 176x144 \
    -frames:v 12 -c:v libx264 -profile:v high -qp 24 -bf 2 -g 6 \
    -x264opts cqm=jvt:b-pyramid=none
check "12 pictures, matrices JVT CAVLC" "$SRC1" 176x144 \
    -frames:v 12 -c:v libx264 -profile:v high -qp 24 -bf 2 -g 6 \
    -x264opts cqm=jvt:cabac=0:b-pyramid=none
check "intra, matrices su size" "$SRC1" 176x144 \
    -frames:v 1 -c:v libx264 -profile:v high -qp 22 \
    -x264opts "cqmfile=$T/cqm.txt"
check "12 pictures, matrices su size" "$SRC1" 176x144 \
    -frames:v 12 -c:v libx264 -profile:v high -qp 24 -bf 2 -g 6 \
    -x264opts "cqmfile=$T/cqm.txt:b-pyramid=none"
check "12 pictures su size, CAVLC" "$SRC1" 176x144 \
    -frames:v 12 -c:v libx264 -profile:v high -qp 24 -bf 2 -g 6 \
    -x264opts "cqmfile=$T/cqm.txt:cabac=0:b-pyramid=none"
check "20 pictures su size, preset slow" "$SRC2" 320x240 \
    -frames:v 20 -c:v libx264 -profile:v high -preset slow -crf 24 -g 10 \
    -x264opts "cqmfile=$T/cqm.txt"

echo
printf 'passed %d, failed %d, skipped %d
' "$passed" "$failed" "$skipped"
[ "$failed" -eq 0 ]
