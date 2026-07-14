#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$script_dir/common.sh"

gpu_positive_integer 42
gpu_nonnegative_integer 0
! gpu_positive_integer 0
! gpu_nonnegative_integer -1
gpu_has_token 'kernel=tiled path=tiled status=PASS' 'path=tiled'
! gpu_has_token 'kernel=tiled-async path=tiled status=PASS' 'kernel=tiled'
! gpu_has_token 'value=axb status=PASS' 'value=a.b'
glob_test_dir="$(mktemp -d)"
trap 'rm -rf "$glob_test_dir"' EXIT
touch "$glob_test_dir/kernel=naive"
pushd "$glob_test_dir" >/dev/null
gpu_has_token 'kernel=* status=PASS' 'kernel=*'
! gpu_has_token 'kernel=* status=PASS' 'kernel=naive'
popd >/dev/null
set -f
gpu_has_token 'kernel=* status=PASS' 'kernel=*'
[[ $- == *f* ]]
! gpu_has_token 'kernel=* status=PASS' 'kernel=naive'
[[ $- == *f* ]]
set +f
gpu_require_command common-test sh

printf '[common_test] status=PASS\n'
