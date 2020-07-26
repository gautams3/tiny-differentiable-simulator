#!/usr/bin/env bash

function linklatest {
  echo
  echo
  echo "Pointing log/latest to $logfile"
  rm log/latest
  ln -s "$logfile" log/latest
}
trap linklatest EXIT

binary=$(basename $1)
logfile="${binary}_$(date +%Y_%m_%d_%I_%M_%p).log"
echo "Logging to $logfile"
"$@" 2>&1 | tee ./log/$logfile
