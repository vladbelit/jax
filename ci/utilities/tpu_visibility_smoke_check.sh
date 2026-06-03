#!/bin/bash
# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

function run_tpu_visibility_smoke_check() {
  local visibility_mode="${1:?}"
  local worker_count="${2:?}"
  local python_bin="${3:?}"

  if [[ "$visibility_mode" != "devices" ]]; then
    return 0
  fi

  echo "Checking per-worker TPU_VISIBLE_DEVICES visibility..."
  local i
  for ((i = 0; i < worker_count; i++)); do
    echo "=== TPU visibility smoke check: TPU_VISIBLE_DEVICES=$i ==="
    if ! env -u TPU_VISIBLE_CHIPS \
      TPU_VISIBLE_DEVICES="$i" \
      TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
      TPU_PROCESS_BOUNDS=1,1,1 \
      ALLOW_MULTIPLE_LIBTPU_LOAD=true \
      JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}" \
      "$python_bin" - <<'PY'; then
import os

import jax


print('TPU_VISIBLE_DEVICES:', os.environ.get('TPU_VISIBLE_DEVICES'))
print('TPU_VISIBLE_CHIPS:', os.environ.get('TPU_VISIBLE_CHIPS'))
print(
    'TPU_CHIPS_PER_PROCESS_BOUNDS:',
    os.environ.get('TPU_CHIPS_PER_PROCESS_BOUNDS'),
)
print('TPU_PROCESS_BOUNDS:', os.environ.get('TPU_PROCESS_BOUNDS'))
print('default backend:', jax.default_backend())
print('devices:', jax.devices())
print('local device count:', jax.local_device_count())
PY
      echo "TPU visibility smoke check failed for TPU_VISIBLE_DEVICES=$i"
      return 1
    fi
  done
}
