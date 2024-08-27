#! /bin/bash

EXP=$1
DATA=$2

./evaluate_asr.sh -e $EXP -d $DATA -p test -r per |
	grep hyp |
	head -n -1 |
	cut -d' ' -f2,3 > /tmp/per.out

./evaluate_asr.sh -e $EXP -d $DATA -p test -r cer |
	grep hyp |
	head -n -1 |
	cut -d' ' -f3 > /tmp/cer.out

paste /tmp/per.out /tmp/cer.out |
	sort -n --key=2 |
	sed "s|'exp/\(.*\)/decode/test_greedy.trn_per':|\1|g" |
	sed -r "s/\s+/ /g" > /tmp/table.out

column -t -s' ' -N "$(tput bold)RUN,PER,CER$(tput sgr0)" /tmp/table.out

