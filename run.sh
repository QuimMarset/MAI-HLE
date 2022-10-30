#! /bin/bash

BASEDIR=experiments/experiment_19

perl scorer.pl $BASEDIR/test_predictions.txt data/raw_data/TEST_FILE_KEY.txt > $BASEDIR/metrics.txt