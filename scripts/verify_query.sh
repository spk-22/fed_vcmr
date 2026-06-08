#!/usr/bin/env bash

# Manual visual verification: query → midpoint frame → JPG on PC.
# Usage: ./verify_query.sh "person sitting on stairs"
set -e
QUERY="$1"
if [ -z "$QUERY" ]; then echo "Usage: $0 \"<query text>\""; exit 1; fi

OUT_DIR="C:/prism/pc_verification/verification"
TMP_DIR="C:/prism/pc_verification/tmp"
mkdir -p "$OUT_DIR" "$TMP_DIR"

SAFE=$(echo "$QUERY" | tr -c 'A-Za-z0-9' '_' | cut -c1-40)

# 1. Clear logcat, fire query, capture logs
adb logcat -c
(adb logcat -s VCMR_RESULT:I > "$TMP_DIR/$SAFE.log" 2>&1 &)
LPID=$!
sleep 1
adb shell "am broadcast -a com.fedvcmr.SEARCH --es query '$QUERY' -p com.fedvcmr" > /dev/null
sleep 6
kill $LPID 2>/dev/null || true

# 2. Parse top-3 hits: videoId, tStart, tEnd
python C:/prism/scripts/_extract_verification.py "$QUERY" "$TMP_DIR/$SAFE.log" "$OUT_DIR"
