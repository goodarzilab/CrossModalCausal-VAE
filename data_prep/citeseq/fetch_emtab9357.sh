#!/usr/bin/env bash
# Parallel, resumable fetch of E-MTAB-9357 processed data.
# Derives the file list from the EBI-provided FTP script, filters by modality,
# and pulls over HTTPS with bounded parallelism.
#
# Usage:
#   ./fetch_emtab9357.sh E-MTAB-9357-unix-ftp.txt [outdir] [jobs]
#   MODALITIES='gex|pro|cd4_tcr|cd8_tcr|bcr' ./fetch_emtab9357.sh ...   # override
#
# Re-running is safe: completed files are skipped, partials resume.

set -euo pipefail

SCRIPT="$(readlink -f "${1:?usage: $0 <E-MTAB-9357-unix-ftp.txt> [outdir] [jobs]}")"
OUTDIR="${2:-./emtab9357}"
JOBS="${3:-8}"
MODALITIES="${MODALITIES:-gex|pro}"

BASE="https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/357/E-MTAB-9357/Files"

mkdir -p "$OUTDIR"/{data,logs}
cd "$OUTDIR"

# ---- 1. build the filtered URL list -----------------------------------------
sed -n 's/^mget "\(.*\)"$/\1/p' "$SCRIPT" > all_names.txt
grep -E "_(${MODALITIES})_library_" all_names.txt > want_names.txt

n_all=$(wc -l < all_names.txt)
n_want=$(wc -l < want_names.txt)
echo "[*] ${n_all} files listed, ${n_want} match modalities: ${MODALITIES}"

sed "s|^|${BASE}/|" want_names.txt > urls.txt

# ---- 1b. preflight: confirm the HTTPS mirror serves this tree ----------------
probe=$(head -1 urls.txt)
if ! curl -fsI --connect-timeout 20 "$probe" >/dev/null 2>&1; then
  echo "[!] HTTPS mirror unreachable for: $probe"
  echo "[!] falling back to ftp://"
  BASE="ftp://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/357/E-MTAB-9357/Files"
  sed "s|^|${BASE}/|" want_names.txt > urls.txt
  curl -fsI --connect-timeout 20 "$(head -1 urls.txt)" >/dev/null 2>&1 \
    || { echo "[!] FTP unreachable too; check network/proxy from this node"; exit 1; }
fi
echo "[*] mirror OK: ${BASE}"

# ---- 1c. MAGE-TAB metadata (SDRF maps files -> donor / severity / timepoint) -
for m in E-MTAB-9357.sdrf.txt E-MTAB-9357.idf.txt; do
  [[ -s "meta/$m" ]] && continue
  mkdir -p meta
  curl -fsSL --retry 3 -o "meta/$m" "${BASE}/$m" 2>/dev/null \
    || curl -fsSL --retry 3 -o "meta/$m" "${BASE%/Files}/$m" 2>/dev/null \
    || echo "[!] could not fetch $m automatically - grab it from the study page"
done
[[ -s meta/E-MTAB-9357.sdrf.txt ]] && echo "[*] SDRF: $(wc -l < meta/E-MTAB-9357.sdrf.txt) rows"

# ---- 2. download ------------------------------------------------------------
if command -v aria2c >/dev/null 2>&1; then
  echo "[*] aria2c, ${JOBS} concurrent"
  aria2c -i urls.txt -d data \
         -j "$JOBS" -x 2 -s 1 \
         --continue=true --auto-file-renaming=false \
         --max-tries=5 --retry-wait=10 --timeout=60 \
         --console-log-level=warn --summary-interval=30 \
         --log=logs/aria2.log
else
  echo "[*] aria2c not found; curl via xargs -P ${JOBS}"
  xargs -a urls.txt -P "$JOBS" -I{} \
    curl -fsSL --retry 5 --retry-delay 10 --connect-timeout 30 \
         -C - -O --output-dir data {} \
    2> logs/curl.err || true
fi

# ---- 3. verify --------------------------------------------------------------
echo "[*] verifying"
: > logs/missing.txt
: > logs/corrupt.txt

while read -r f; do
  if [[ ! -s "data/$f" ]]; then
    echo "$f" >> logs/missing.txt
  elif ! gzip -t "data/$f" 2>/dev/null; then
    echo "$f" >> logs/corrupt.txt
  fi
done < want_names.txt

nm=$(wc -l < logs/missing.txt)
nc=$(wc -l < logs/corrupt.txt)
echo "[*] ok: $(( n_want - nm - nc ))   missing: ${nm}   corrupt: ${nc}"

if (( nm + nc > 0 )); then
  cat logs/missing.txt logs/corrupt.txt | sed "s|^|${BASE}/|" > logs/retry_urls.txt
  echo "[!] delete corrupt: xargs -a logs/corrupt.txt -I{} rm -f data/{}"
  echo "[!] then rerun this script (it resumes) or:"
  echo "    aria2c -i logs/retry_urls.txt -d data -j 4 --continue=true"
  exit 1
fi

echo "[*] done -> $(pwd)/data"
