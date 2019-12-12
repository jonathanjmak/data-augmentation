#!/bin/bash
rm -rf run_logs_zip
cp -r run_logs run_logs_zip
rm -f run_logs_zip/*/*.path
zip -r run_logs.zip run_logs_zip
