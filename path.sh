#! /usr/bin/env bash

if [ "${SLURM_JOB_ACCOUNT}" = "vector" ]; then
    source "$HOME/.bashrc"
    micromamba activate faetar-dev-kit
fi
