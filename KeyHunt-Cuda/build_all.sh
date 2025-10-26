#!/usr/bin/env bash
#
# build_all.sh
# Example script to build multiple CCAP versions,
# package them, generate checksums, and produce
# a text file with the download information
# in Markdown format.
#

# Version / release tag to embed in download links
VERSION="v1.0.2"

# Base download URL for your GitHub releases
RELEASE_URL="https://github.com/jrharp/challengepool.net/releases/download/${VERSION}"

# Define the CCAP versions and their corresponding GPU descriptions
declare -A GPU_MAP=(
  [86]="RTX 30 Series"
  [89]="RTX 40 Series"
  [90]="H100/H200"
  [120]="RTX 50 Series"
)

# This text file will collect all the info
OUTPUT_FILE="${VERSION}_ccap_build_info.txt"

# Clear or create the output file
echo "Download the required version for your NVidia GPU." > "${OUTPUT_FILE}"
echo "" >> "${OUTPUT_FILE}"

# Loop over each CCAP version, build, tar, and compute checksums

for ccap in 86 89 90 120; do

  echo "=========================================="
  echo "Building CCAP=${ccap}..."
  echo "=========================================="

  # Clean and build
  make clean CCAP="${ccap}"
  make gpu=1 CCAP="${ccap}" all

  # Move to build/ (adjust if your artifacts appear elsewhere)
  cd build/ || {
    echo "ERROR: build/ directory not found!"
    exit 1
  }

  # Create the archive (e.g., ccap_61.tar.gz from ccap_61/ directory)
  TAR_NAME="ccap_${ccap}.tar.gz"
  echo "Packaging ${TAR_NAME}..."
  tar -czvf "${TAR_NAME}" "ccap_${ccap}/"

  # Compute and capture SHA256 sum
  SUM=$(sha256sum "${TAR_NAME}" | awk '{print $1}')

  # Append info to our text file in the requested Markdown format:
  # - [`ccap_61.tar.gz`](URL) - GPU - `CHECKSUM`
  echo "- [\`${TAR_NAME}\`](${RELEASE_URL}/${TAR_NAME}) - ${GPU_MAP[$ccap]} - \`${SUM}\`" >> "../${OUTPUT_FILE}"

  # Return to previous directory (where the Makefile lives)
  cd - >/dev/null 2>&1
done

echo "=========================================="
echo "Build and packaging complete."
echo "See '${OUTPUT_FILE}' for download checksums."
echo "=========================================="
