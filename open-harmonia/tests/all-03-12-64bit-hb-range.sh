dir="0312-64bit-hb-range"
mkdir $dir 

./link-to-64.sh 

date 
echo cpu-range-hb 
./cpu-range-hb.sh > ./$dir/result.cpu-range-hb 

echo range-hb-notransfer
./range-hb-notransfer.sh > ./$dir/result.range-hb-notransfer


