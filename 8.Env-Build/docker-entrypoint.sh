#!/bin/bash
set -e


if [ $# -eq 0 ]
  then
    jupyter lab --ip=0.0.0.0 --NotebookApp.token='CXY2019!!!!' --allow-root --no-browser --notebook-dir=/code &> /dev/null &
    code-server-3.1.1-linux-x86_64/code-server --allow-http --password 'CXY2019!!!!' --data-dir /data /code
  else
    exec "$@"
fi
