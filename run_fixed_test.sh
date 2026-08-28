#!/bin/bash

# Run only after all configured fixed-test days are available. fixed_test.sh
# reuses the completed checkpoint and Saliency assets and does not retrain.
set -euo pipefail
export run_fixed_test=1
exec bash fixed_test.sh
