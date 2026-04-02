#! bin/bash
# This file is used to split the crace standard outputs into
# training and testing phases with names 'crace.train' and 
# 'crace.test', respectively.

DIR=$(dirname $0)
# DIR=$(cd "$(dirname "$0")";pwd)
outfile=$DIR/crace-*.stdout
initest=`sed -n '/Initializing tester/=' $outfile` # the begining of test start
ends=`sed -n '/Shutdown complete/=' $outfile` # the end of training and test
endline=0
count=0
FLAG=0

if ! [[ ${initest} =~ ^[-+0-9.e]+$ ]]; then
    cp $outfile $DIR/crace.train
else

    for n in $ends; do
        let startline=${endline}+1
        if [ $count == 0 ]; then                # generate the first file
            let endline=$n+1
            if [[ ${initest} < $n ]] ; then   # onlytest
                sed -n "${startline},${endline}p" $outfile > $DIR/crace.test
                FLAG=1
            else                                # first training
                let endline=${initest}-1
                sed -n "${startline},${endline}p" $outfile > $DIR/crace.train
            fi
            count+=1
        else                                    # generate the second file
            let endline=$n+1
            sed -n "${startline},${endline}p" $outfile > $DIR/crace.test
            FLAG=1
        fi
    done

fi

# if [[ $FLAG == 1 ]] ; then
#     rm $outfile
# fi

if [ -s $DIR/crace-*.stderr ]; then
    mv $DIR/crace-*.stderr $DIR/crace.stderr
fi