
BDIR=$(pwd)

# compile BART with AFL
# make allclean
# CC=afl-gcc make


cd tests
mkdir fuzz
cd fuzz

for dir in cfl coo ra stl stream mcfl ; do

	mkdir in_$dir
	mkdir out_$dir
done

DUR=300
export AFL_SKIP_CPUFREQ=1
export AFL_AUTORESUME=1


# stream
$BDIR/bart ones -- 1 1 - > in_stream/s.hdr
$BDIR/bart --stream-bin-out ones -- 1 1 - > in_stream/b.hdr
afl-fuzz -V $DUR -i in_stream/ -o out_stream/ -m1500 -t100 -- $BDIR/bart flip -- 0 - o
rm o.{cfl,hdr}


# multicfl
$BDIR/bart ones 1 1 o
$BDIR/bart zeros 1 1 z
$BDIR/bart multicfl o z in_mcfl/m
rm o.{cfl,hdr} z.{cfl,hdr}
cp in_mcfl/m.cfl .
afl-fuzz -V $DUR -i in_mcfl/ -o out_mcfl/ -m1500 -t100 -f m.hdr -- $BDIR/bart multicfl -s m o
rm o.{cfl,hdr}


# coo
$BDIR/bart ones 1 1 in_coo/c.coo
afl-fuzz -V $DUR -i in_coo/ -o out_coo/ -m1500 -t100 -f c.coo -- $BDIR/bart flip -- 0 c.coo o.coo
rm o.coo


# i.{hdr,cfl}
$BDIR/bart ones 1 1 in_cfl/i
cp in_cfl/i.cfl .
afl-fuzz -V $DUR -i in_cfl/ -o out_cfl/ -m1500 -t100 -f i.hdr -- $BDIR/bart flip -- 0 i o
rm o.{cfl,hdr}


# ra
$BDIR/bart ones 1 1 in_ra/r.ra
afl-fuzz -V $DUR -i in_ra/ -o out_ra/ -m1500 -t100 -f r.ra -- $BDIR/bart flip -- 0 r.ra o.ra
rm o.ra


# stl
$BDIR/bart stl --model TET in_stl/s.stl
$BDIR/bart stl --binary --model TET in_stl/b.stl
afl-fuzz -V $DUR -i in_stl/ -o out_stl/ -m1500 -t100 -f s.stl -- $BDIR/bart stl --input s.stl o
rm o.{cfl,hdr}



