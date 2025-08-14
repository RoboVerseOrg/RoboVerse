#!/bin/bash

for path in "$@"; do
    target_path=${path/roboverse_data_dev/roboverse_data}

    target_dir=$(dirname "$target_path")
    if [ ! -d "$target_dir" ]; then
        mkdir -p "$target_dir"
    fi

    command="cp -r \"$path\" \"$target_path\""
    echo "Executing: $command"
    eval "$command"
done
