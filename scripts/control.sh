#!/usr/bin/env bash
# More safety, by turning some bugs into errors.
# Without `errexit` you don’t need ! and can replace
# PIPESTATUS with a simple $?, but I don’t do that.
set -o errexit -o pipefail -o noclobber -o nounset

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && (pwd -W 2> /dev/null || pwd))
cd "$SCRIPT_DIR/.."


POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -q7|--test-qwen7b)
      TEST_QWEN7B=YES
      shift # past argument
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [[ -v TEST_QWEN7B ]]; then
  echo "test"
else
    echo -n "\
Please specify the right commandline option:
-e/--environment {environment name} : specify environment(dev/staging/prod)
"
fi

