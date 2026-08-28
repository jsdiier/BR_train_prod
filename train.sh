#!/bin/bash

# Keep direct/manual invocations on the same closed workflow as the Luban job.
set -euo pipefail
exec bash fixed_test.sh
