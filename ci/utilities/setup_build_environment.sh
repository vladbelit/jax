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
# Set up the build environment for JAX CI jobs. This script depends on the
# "JAXCI_" environment variables set or sourced in the build script.

# Preemptively mark the JAX git directory as safe. This is necessary for JAX CI
# jobs running on Linux runners in GitHub Actions. Without this, git complains
# that the directory has dubious ownership and refuses to run any commands.
# Avoid running on Windows runners as git runs into issues with not being able
# to lock the config file. Other git commands seem to work on the Windows
# runners so we can skip this step for Windows.
# TODO(b/375073267): Remove this once we understand why git repositories are
# being marked as unsafe inside the self-hosted runners.
if [[ ! $(uname -s) =~ "MSYS_NT" ]]; then
  git config --global --add safe.directory $JAXCI_JAX_GIT_DIR
fi

function resolved_xla_git_url() {
  if [[ -n "$JAXCI_XLA_GIT_URL" ]]; then
    echo "$JAXCI_XLA_GIT_URL"
  else
    echo "https://github.com/openxla/xla.git"
  fi
}

function clone_main_xla() {
  local xla_git_url
  xla_git_url="$(resolved_xla_git_url)"
  echo "Cloning XLA from ${xla_git_url} to $(pwd)/xla"
  git clone --depth=1 "${xla_git_url}" $(pwd)/xla
  cd $(pwd)/xla
  echo "XLA commit: $(git log -1 --format=%H)"
  cd ..
  export JAXCI_XLA_GIT_DIR=$(pwd)/xla
}

function ensure_local_xla_checkout() {
  if [[ -z "$JAXCI_XLA_GIT_DIR" ]]; then
    if [[ ! -d $(pwd)/xla ]]; then
      clone_main_xla
    else
      echo "Using existing local XLA folder at $(pwd)/xla."
      export JAXCI_XLA_GIT_DIR=$(pwd)/xla
    fi
  fi
}

function configure_xla_remote() {
  local xla_git_url
  xla_git_url="$(resolved_xla_git_url)"

  pushd "$JAXCI_XLA_GIT_DIR"

  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "${xla_git_url}"
  else
    git remote add origin "${xla_git_url}"
  fi

  popd
}

if [[ -n "$JAXCI_XLA_REF" && -n "$JAXCI_XLA_COMMIT" ]]; then
  echo "Only one of JAXCI_XLA_REF or JAXCI_XLA_COMMIT may be set."
  exit 1
fi

# Clone XLA at HEAD if required.
if [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
  ensure_local_xla_checkout
  configure_xla_remote
fi

if [[ ! -z "$JAXCI_XLA_REF" ]]; then
  ensure_local_xla_checkout
  configure_xla_remote

  pushd "$JAXCI_XLA_GIT_DIR"

  git fetch --depth=1 origin "$JAXCI_XLA_REF"
  echo "JAXCI_XLA_REF is set. Checking out XLA at $JAXCI_XLA_REF from $(resolved_xla_git_url)"
  git checkout FETCH_HEAD

  popd
fi

# If a XLA commit is provided, check out XLA at that commit.
if [[ ! -z "$JAXCI_XLA_COMMIT" ]]; then
  ensure_local_xla_checkout
  configure_xla_remote

  pushd "$JAXCI_XLA_GIT_DIR"

  git fetch --depth=1 origin "$JAXCI_XLA_COMMIT"
  echo "JAXCI_XLA_COMMIT is set. Checking out XLA at $JAXCI_XLA_COMMIT from $(resolved_xla_git_url)"
  git checkout "$JAXCI_XLA_COMMIT"

  popd
fi

if [[ ! -z ${JAXCI_XLA_GIT_DIR} ]]; then
  echo "INFO: Overriding XLA to be read from $JAXCI_XLA_GIT_DIR instead of the"
  echo "pinned version in the WORKSPACE."
  echo "XLA remote: $(resolved_xla_git_url)"
  if [[ ! -z "$JAXCI_XLA_REF" ]]; then
    echo "XLA ref override: $JAXCI_XLA_REF"
  fi
  if [[ ! -z "$JAXCI_XLA_COMMIT" ]]; then
    echo "XLA commit override: $JAXCI_XLA_COMMIT"
  fi
  echo "If you would like to revert this behavior, unset JAXCI_CLONE_MAIN_XLA,"
  echo "JAXCI_XLA_REF, and JAXCI_XLA_COMMIT in your environment. Note that the"
  echo "Bazel RBE test commands override the XLA repository and thus require a"
  echo "local copy of XLA to run."
fi

# On Windows, convert MSYS Linux-like paths to Windows paths.
if [[ $(uname -s) =~ "MSYS_NT" ]]; then
  echo 'Converting MSYS Linux-like paths to Windows paths (for Bazel, Python, etc.)'
  # Convert all "JAXCI.*DIR" variables
  source <(python3 ./ci/utilities/convert_msys_paths_to_win_paths.py --convert $(env | grep "JAXCI.*DIR" | awk -F= '{print $1}'))
fi

function retry {
  local cmd="$1"
  local max_attempts=3
  local attempt=1
  local delay=10

  while [[ $attempt -le $max_attempts ]] ; do
    if eval "$cmd"; then
      return 0
    fi
    echo "Attempt $attempt failed. Retrying in $delay seconds..."
    sleep $delay # Prevent overloading

    attempt=$((attempt + 1))
  done
  echo "$cmd failed after $max_attempts attempts."
  exit 1
}

# Retry "bazel --version" 3 times to avoid flakiness when downloading bazel.
retry "bazel --version"

# Create the output directory if it doesn't exist.
mkdir -p "$JAXCI_OUTPUT_DIR"
