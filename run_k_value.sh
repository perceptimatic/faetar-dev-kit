# specify what folders to work with

# copy lstm trn file to folder

data=data/boothroyd
model="$1"

for x in test train dev; do
    faetar_dev_kit/boothroyd/section_data.sh \
    "$data"/"$x"_"$model"/perplexity_5gram.arpa exp/mms_lsah/decode/"$x"_greedy.trn 3
    
    echo "hp/zp $x"
    python faetar_dev_kit/boothroyd/get_k.py \
    "$data"/"$x"_"$model"/1_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/1_hyp_perplexity_5gram.arpa \
    "$data"/"$x"_"$model"/3_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/3_hyp_perplexity_5gram.arpa

    echo "lp/zp $x"
    python faetar_dev_kit/boothroyd/get_k.py \
    "$data"/"$x"_"$model"/2_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/2_hyp_perplexity_5gram.arpa \
    "$data"/"$x"_"$model"/3_ref_perplexity_5gram.arpa "$data"/"$x"_"$model"/3_hyp_perplexity_5gram.arpa

done

