#! /bin/bash

if [ $# -ne 2 ]; then
	echo "USAGE $0 decodings datas"
	exit 1
fi

DECODINGS=$1
DATA=$2

if [ ! -d "$DECODINGS" ]; then
	echo "decodings is not a directory"
	exit 1
elif [ ! -d "$DATA" ]; then
	echo "datas is not a directory"
	exit 1
fi

PER="$(mktemp -t per.XXX)"
CER="$(mktemp -t cer.XXX)"
WER="$(mktemp -t wer.XXX)"
TABLE="$(mktemp -t table.XXX)"

./evaluate_asr.sh -e $DECODINGS -d $DATA -p test -r per |
	grep hyp |
	head -n -1 |
	cut -d' ' -f2,3 > $PER

./evaluate_asr.sh -e $DECODINGS -d $DATA -p test -r cer |
	grep hyp |
	head -n -1 |
	cut -d' ' -f3 > $CER

./evaluate_asr.sh -e $DECODINGS -d $DATA -p test -r wer |
	grep hyp |
	head -n -1 |
	cut -d' ' -f3 > $WER

paste $PER $CER $WER |
	sort -n --key=2 |
	sed "s|'$DECODINGS/\(.*\)/decode/test_greedy.trn_per':|\1|g" |
	sed -r "s/\s+/ /g" > $TABLE

column -t -s' ' -N "$(tput bold)RUN,PER,CER,WER$(tput sgr0)" $TABLE

