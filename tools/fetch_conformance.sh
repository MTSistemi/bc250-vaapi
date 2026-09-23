#!/bin/bash
# Fetch the official JCT-VC conformance bitstreams for HEVC version 1.
#
#     fetch_conformance.sh [directory]
#
# 147 archives, a few hundred megabytes in total, from the ITU's own
# archive. They are the Main, Main 10 and Main Still Picture set - the
# profiles this decoder claims - and each one is named after the tool it
# is built to stress.
#
# Skips anything already there, so it can be re-run.
set -u
OUT="${1:-$HOME/jctvc}"
BASE=https://www.itu.int/wftp3/av-arch/jctvc-site/bitstream_exchange/draft_conformance/HEVC_v1

mkdir -p "$OUT" || exit 1
cd "$OUT" || exit 1

echo "asking the archive what is in it"
list=$(curl -sS -m 120 "$BASE/" | grep -oE '[A-Za-z0-9_]+\.zip' | sort -u)
if [ -z "$list" ]; then
    echo "the listing came back empty - the archive may have moved"
    exit 1
fi

total=$(echo "$list" | wc -l)
have=0
got=0
echo "$total archives"

for f in $list; do
    if [ -s "$f" ]; then
        have=$((have + 1)); continue
    fi
    if curl -sS -m 300 -O "$BASE/$f"; then
        got=$((got + 1))
    else
        echo "  could not fetch $f"
    fi
done

echo "already had $have, fetched $got, in $OUT"
echo "now: tools/test_hevc_conformance.sh $OUT"
