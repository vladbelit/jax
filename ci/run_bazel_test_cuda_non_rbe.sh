#!/bin/bash
# Copyright 2024 The JAX Authors.
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
# Run Bazel GPU tests without RBE. This runs two commands: single accelerator
# tests with one GPU a piece, multiaccelerator tests with all GPUS.
# If $JAXCI_BUILD_JAXLIB=false, the job requires that jaxlib, jax-cuda-plugin,
# and jax-cuda-pjrt wheels are stored inside the ../dist folder
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Source default JAXCI environment variables.
source ci/envs/default.env

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

start_log_section() {
  local section_name="$1"
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "::group::$section_name"
  fi
  printf "\n========== %s ==========\n" "$section_name"
}

end_log_section() {
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "::endgroup::"
  fi
}

print_test_configuration() {
  start_log_section "Bazel CUDA Non-RBE test configuration"
  echo "CUDA version: $JAXCI_CUDA_VERSION"
  echo "Bazel config: $TEST_CONFIG"
  echo "Build jaxlib: $JAXCI_BUILD_JAXLIB"
  echo "Build jax: $JAXCI_BUILD_JAX"
  echo "Hermetic Python: $JAXCI_HERMETIC_PYTHON_VERSION"
  echo "Freethreaded Python: $FREETHREADED_FLAG_VALUE"
  echo "GPU count: $gpu_count"
  echo "Memory per GPU (GiB): $memory_per_gpu_gb"
  echo "Max tests per GPU: $max_tests_per_gpu"
  echo "Single-accelerator local test jobs: $num_test_jobs"
  echo "Multi-accelerator local test jobs: $multi_accelerator_num_test_jobs"
  end_log_section
}

# Run Bazel GPU tests (single accelerator and multiaccelerator tests) directly
# on the VM without RBE.
nvidia-smi

# Set up test environment variables.
# Set the number of test jobs to min(num_cpu_cores, gpu_count * max_tests_per_gpu, total_ram_gb / 6)
# We calculate max_tests_per_gpu as memory_per_gpu_gb / 2gb
# Calculate gpu_count * max_tests_per_gpu
export gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export memory_per_gpu_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=0)
export memory_per_gpu_gb=$((memory_per_gpu_gb / 1024))
# Allow 2 GB of GPU RAM per test
export max_tests_per_gpu=$((memory_per_gpu_gb / 2))
export num_test_jobs=$((gpu_count * max_tests_per_gpu))

# Calculate num_cpu_cores
export num_cpu_cores=$(nproc)

# Calculate total_ram_gb / 6
export total_ram_gb=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
export host_memory_limit=$((total_ram_gb / 6))

if [[ $num_cpu_cores -lt $num_test_jobs ]]; then
  num_test_jobs=$num_cpu_cores
fi

if [[ $host_memory_limit -lt $num_test_jobs ]]; then
  num_test_jobs=$host_memory_limit
fi
multi_accelerator_num_test_jobs=8
# End of test environment variables setup.

if [[ "$JAXCI_HERMETIC_PYTHON_VERSION" == *"-nogil" ]]; then
  JAXCI_HERMETIC_PYTHON_VERSION=${JAXCI_HERMETIC_PYTHON_VERSION%-nogil}-ft
  FREETHREADED_FLAG_VALUE="yes"
else
  FREETHREADED_FLAG_VALUE="no"
fi

OVERRIDE_XLA_REPO=""
if [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
  OVERRIDE_XLA_REPO="--override_repository=xla=${JAXCI_XLA_GIT_DIR}"
fi

# Get the CUDA major version only
cuda_major_version="${JAXCI_CUDA_VERSION%%.*}"

if [[ "$JAXCI_BUILD_ARTIFACT_WITH_RBE" == "true" ]]; then
  TEST_CONFIG="rbe_linux_x86_64_cuda$cuda_major_version"
  TEST_STRATEGY="--strategy=TestRunner=local"
  CACHE_OPTION=""
else
  TEST_CONFIG="ci_linux_x86_64_cuda$cuda_major_version"
  CACHE_OPTION="--config=ci_rbe_cache"
  TEST_STRATEGY=""
fi

common_bazel_test_args=(
  test
  "--config=$TEST_CONFIG"
  "--repo_env=HERMETIC_PYTHON_VERSION=$JAXCI_HERMETIC_PYTHON_VERSION"
  "--@rules_python//python/config_settings:py_freethreaded=$FREETHREADED_FLAG_VALUE"
  "--repo_env=HERMETIC_CUDA_UMD_VERSION=13.0.2"
  "--//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB"
  "--//jax:build_jax=$JAXCI_BUILD_JAX"
  "--test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform"
  "--test_env=TF_CPP_MIN_LOG_LEVEL=0"
  "--test_env=JAX_SKIP_SLOW_TESTS=true"
  "--action_env=JAX_ENABLE_X64=$JAXCI_ENABLE_X64"
  "--action_env=NCCL_DEBUG=WARN"
  --color=yes
  --config=cuda_libraries_from_stubs
  --config=hermetic_cuda_umd
)
if [[ -n "$CACHE_OPTION" ]]; then
  common_bazel_test_args+=("$CACHE_OPTION")
fi
if [[ -n "$OVERRIDE_XLA_REPO" ]]; then
  common_bazel_test_args+=("$OVERRIDE_XLA_REPO")
fi
if [[ -n "$TEST_STRATEGY" ]]; then
  common_bazel_test_args+=("$TEST_STRATEGY")
fi

single_accelerator_test_args=(
  "--run_under=$(pwd)/build/parallel_accelerator_execute.sh"
  --test_output=errors
  "--test_env=JAX_ACCELERATOR_COUNT=$gpu_count"
  "--test_env=JAX_TESTS_PER_ACCELERATOR=$max_tests_per_gpu"
  "--local_test_jobs=$num_test_jobs"
  --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow
  --test_tag_filters=-multiaccelerator
)
single_accelerator_test_targets=(
  //tests:gpu_tests
  //tests:backend_independent_tests
  //tests/pallas:gpu_tests
  //tests/pallas:backend_independent_tests
)

multi_accelerator_test_args=(
  --test_output=errors
  "--local_test_jobs=$multi_accelerator_num_test_jobs"
  --test_tag_filters=multiaccelerator
)
multi_accelerator_test_targets=(
  //tests:gpu_tests
  //tests/pallas:gpu_tests
  //tests/multiprocess:gpu_tests
)

print_test_configuration

if [[ "$JAXCI_BUILD_JAXLIB" == "false" || "$JAXCI_BUILD_JAX" == "false" ]]; then
  # Do not proceed to the full-scale testing without first verifying the local
  # wheel resolution works properly.
  bash ci/run_local_wheel_smoke_test.sh "${common_bazel_test_args[@]}"
fi

# Don't abort the script if one command fails to ensure we run both test
# commands below.
set +e

# Runs single accelerator tests with one GPU apiece.
# It appears --run_under needs an absolute path.
# The product of the `JAX_ACCELERATOR_COUNT`` and `JAX_TESTS_PER_ACCELERATOR`
# should match the VM's CPU core count (set in `--local_test_jobs`).
start_log_section "Bazel CUDA Non-RBE single accelerator tests"
bazel "${common_bazel_test_args[@]}" \
  "${single_accelerator_test_args[@]}" \
  -- \
  "${single_accelerator_test_targets[@]}"
first_bazel_cmd_retval=$?
end_log_section

# Runs multiaccelerator tests with all GPUs directly on the VM without RBE...
start_log_section "Bazel CUDA Non-RBE multi-accelerator tests"
bazel "${common_bazel_test_args[@]}" \
  "${multi_accelerator_test_args[@]}" \
  -- \
  "${multi_accelerator_test_targets[@]}"
second_bazel_cmd_retval=$?
end_log_section

ci/utilities/collect_bazel_test_xmls.sh test-artifacts

# Exit with failure if either command fails.
if [[ $first_bazel_cmd_retval -ne 0 ]]; then
  exit $first_bazel_cmd_retval
fi

if [[ $second_bazel_cmd_retval -ne 0 ]]; then
  exit $second_bazel_cmd_retval
fi

exit 0
