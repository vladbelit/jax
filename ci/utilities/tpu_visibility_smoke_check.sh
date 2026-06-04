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
      TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1,1 \
      TPU_PROCESS_BOUNDS=1,1,1,1 \
      ALLOW_MULTIPLE_LIBTPU_LOAD=true \
      JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}" \
      "$python_bin" - <<'PY'; then
import os
import pprint

import numpy as np
import jax


_DEVICE_ATTRS = (
    'id',
    'process_index',
    'platform',
    'device_kind',
    'coords',
    'core_on_chip',
    'slice_index',
)


def _safe_value(value):
  if callable(value):
    try:
      return value()
    except TypeError:
      return '<callable>'
  return value


def _device_info(device):
  info = {'repr': repr(device)}
  for attr in _DEVICE_ATTRS:
    if hasattr(device, attr):
      info[attr] = _safe_value(getattr(device, attr))
  return info


def _array_device(array):
  device = getattr(array, 'device', None)
  if device is None:
    return None
  return _safe_value(device)


def _print_device_list(label, devices):
  print(f'{label}:')
  pprint.pp([_device_info(device) for device in devices], sort_dicts=True)


print('TPU_VISIBLE_DEVICES:', os.environ.get('TPU_VISIBLE_DEVICES'))
print('TPU_VISIBLE_CHIPS:', os.environ.get('TPU_VISIBLE_CHIPS'))
print(
    'TPU_CHIPS_PER_PROCESS_BOUNDS:',
    os.environ.get('TPU_CHIPS_PER_PROCESS_BOUNDS'),
)
print('TPU_PROCESS_BOUNDS:', os.environ.get('TPU_PROCESS_BOUNDS'))
print('default backend:', jax.default_backend())
print('process count:', jax.process_count())
print('process index:', jax.process_index())
print('device count:', jax.device_count())
devices = jax.devices()
local_devices = jax.local_devices()
print('len(jax.devices()):', len(devices))
print('len(jax.local_devices()):', len(local_devices))
_print_device_list('jax.devices()', devices)
_print_device_list('jax.local_devices()', local_devices)
addressable_devices = getattr(jax, 'addressable_devices', None)
if addressable_devices is None:
  print('jax.addressable_devices(): unavailable')
else:
  _print_device_list('jax.addressable_devices()', addressable_devices())
local_device_count = jax.local_device_count()
print('local device count:', local_device_count)
if local_device_count != 1:
  raise SystemExit(
      f'Expected exactly one local TPU device; got {local_device_count}'
  )
if len(local_devices) != 1:
  raise SystemExit(
      f'Expected exactly one local TPU device object; got {len(local_devices)}'
  )
probe = jax.device_put(np.arange(4, dtype=np.int32), local_devices[0])
probe.block_until_ready()
print('device_put local array device:', _device_info(_array_device(probe)))
print('device_put local array value:', np.asarray(probe).tolist())
PY
      echo "TPU visibility smoke check failed for TPU_VISIBLE_DEVICES=$i"
      return 1
    fi
  done
}
