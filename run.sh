#! /bin/bash

PREDICTIONSDIR=best_predictions
SAVEDIR=metrics
GROUNDTRUTH=data/raw_data/TEST_FILE_KEY.txt


perl scorer.pl $PREDICTIONSDIR/best_predictions_cnn.txt $GROUNDTRUTH > $SAVEDIR/metrics_cnn.txt


perl scorer.pl $PREDICTIONSDIR/best_predictions_attention_lstm.txt $GROUNDTRUTH > $SAVEDIR/metrics_atention_lstm.txt


perl scorer.pl $PREDICTIONSDIR/best_predictions_entity_attention.txt $GROUNDTRUTH > $SAVEDIR/metrics_entity_attention.txt


perl scorer.pl $PREDICTIONSDIR/best_predictions_RBERT.txt $GROUNDTRUTH > $SAVEDIR/metrics_rbert.txt


perl scorer.pl $PREDICTIONSDIR/best_predictions_attention_lstm_bert.txt $GROUNDTRUTH > $SAVEDIR/metrics_attention_lstm_bert.txt


perl scorer.pl $PREDICTIONSDIR/best_predictions_entity_attention_bert.txt $GROUNDTRUTH > $SAVEDIR/metrics_entity_attention_bert.txt