#!/usr/bin/env bash

# Shared leaf helpers only. Project scripts retain their own cases, metrics,
# paths, output schemas, and evidence protocols.

gpu_die() {
    local prefix="$1"
    shift
    printf '%s: %s\n' "$prefix" "$*" >&2
    exit 1
}

gpu_require_command() {
    local prefix="$1"
    local command_name="$2"
    command -v "$command_name" >/dev/null 2>&1 || \
        gpu_die "$prefix" "找不到命令: $command_name"
}

gpu_positive_integer() {
    [[ $1 =~ ^[1-9][0-9]*$ ]]
}

gpu_nonnegative_integer() {
    [[ $1 =~ ^[0-9]+$ ]]
}

gpu_has_token() {
    local text="$1"
    local token="$2"
    local word
    local had_noglob=0
    [[ $- == *f* ]] && had_noglob=1
    set -o noglob
    for word in $text; do
        if [[ $word == "$token" ]]; then
            ((had_noglob == 1)) || set +o noglob
            return 0
        fi
    done
    ((had_noglob == 1)) || set +o noglob
    return 1
}
