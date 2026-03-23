#!/usr/bin/env bash

set -euo pipefail

dry_run="${1:-0}"
job_ids="$(squeue -u "$USER" --noheader -o '%i %j' | awk '$2 ~ /^ca_/ {print $1}')"

if [[ -z "${job_ids}" ]]; then
  echo "No SLURM jobs with names starting with ca_ found for user ${USER}."
  exit 0
fi

if [[ "${dry_run}" == "1" ]]; then
  echo "Dry run: would cancel SLURM jobs:"
  echo "${job_ids}"
  exit 0
fi

echo "Cancelling SLURM jobs:"
echo "${job_ids}"
scancel ${job_ids}
