
$(TESTS_OUT)/tetrahedron.ra: stl
	$(TOOLDIR)/stl --model=TET $@

$(TESTS_OUT)/hexahedron.ra: stl
	$(TOOLDIR)/stl --model=HEX $@

$(TESTS_OUT)/icosahedron.ra: stl
	$(TOOLDIR)/stl --model=ICO $@

$(TESTS_OUT)/ascii_enc.stl:
	echo "solid name" > $@		;\
	echo "facet normal 0 0 1" >> $@	;\
	echo "outer loop" >> $@		;\
	echo "vertex 1 0 0" >> $@	;\
	echo "vertex 0 0 0" >> $@	;\
	echo "vertex 0 1 0" >> $@	;\
	echo "endloop" >> $@		;\
	echo "endfacet" >> $@		;\
	echo "endsolid name" >> $@

tests/test-stl-rw: nrmse stl $(TESTS_OUT)/tetrahedron.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/stl --input $(TESTS_OUT)/tetrahedron.ra tet.stl	;\
	$(TOOLDIR)/stl --input tet.stl tet.ra        			;\
	$(TOOLDIR)/nrmse -t 0 $(TESTS_OUT)/tetrahedron.ra tet.ra 	;\
	rm *.ra *.stl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-stl-rw-binary: nrmse stl $(TESTS_OUT)/tetrahedron.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/stl --binary --input $(TESTS_OUT)/tetrahedron.ra tet.stl	;\
	$(TOOLDIR)/stl --input tet.stl tet.ra        				;\
	$(TOOLDIR)/nrmse -t 0 $(TESTS_OUT)/tetrahedron.ra tet.ra 		;\
	rm *.ra *.stl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-stl-read-ascii: vec join nrmse stl $(TESTS_OUT)/ascii_enc.stl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/stl --input $(TESTS_OUT)/ascii_enc.stl tri.ra 	;\
	$(TOOLDIR)/vec -- 1. 0. 0. v0.ra				;\
	$(TOOLDIR)/vec -- 0. 0. 0. v1.ra				;\
	$(TOOLDIR)/vec -- 0. 1. 0. v2.ra				;\
	$(TOOLDIR)/vec -- 0. 0. -1. n.ra				;\
	$(TOOLDIR)/join 1 v0.ra v1.ra v2.ra n.ra tri_ref.ra		;\
	$(TOOLDIR)/nrmse -t 0. tri_ref.ra tri.ra 			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-stl-stat-rot: vec join nrmse stl $(TESTS_OUT)/tetrahedron.ra $(TESTS_OUT)/hexahedron.ra $(TESTS_OUT)/icosahedron.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/stl --stat							;\
	$(TOOLDIR)/stl --model TET -s 0.5 tet_sc.ra				;\
	$(TOOLDIR)/stl --transform "2:0:0:0:0:0:0" --input tet_sc.ra tett.ra	;\
	$(TOOLDIR)/nrmse -t 0. tett.ra $(TESTS_OUT)/tetrahedron.ra			;\
	$(TOOLDIR)/stl --transform "1:10:0:0:0:0:0" --model ICO ico.ra 		;\
	$(TOOLDIR)/stl -m "-10:5:-5" --input ico.ra ico_1.ra			;\
	$(TOOLDIR)/stl --transform "0.5:0:-5:5:0:0:0" --input ico_1.ra ico_2.ra ;\
	$(TOOLDIR)/stl -s 2 --input ico_2.ra ico_3.ra				;\
	$(TOOLDIR)/stl --transform "1:0:0:0:15:0:0" --input ico_3.ra ico_4.ra	;\
	$(TOOLDIR)/stl --transform "1:0:0:0:-15:0:0" --input ico_4.ra ico_5.ra	;\
	$(TOOLDIR)/stl --transform "1:0:0:0:345:0:0" --input ico_4.ra ico_6.ra	;\
	$(TOOLDIR)/stl --transform "1:0:0:0:0:360:0" --input ico_3.ra ico_7.ra	;\
	$(TOOLDIR)/stl --transform "1:0:0:0:0:0:-360" --input ico_3.ra ico_8.ra	;\
	$(TOOLDIR)/nrmse -t 1.e-7 ico_5.ra $(TESTS_OUT)/icosahedron.ra		;\
	$(TOOLDIR)/nrmse -t 1.e-7 ico_6.ra $(TESTS_OUT)/icosahedron.ra		;\
	$(TOOLDIR)/nrmse -t 1.e-15 ico_7.ra $(TESTS_OUT)/icosahedron.ra		;\
	$(TOOLDIR)/nrmse -t 1.e-15 ico_8.ra $(TESTS_OUT)/icosahedron.ra		;\
	$(TOOLDIR)/stl --transform "1:0:0:0:360:0:0" --model HEX hexz.ra	;\
	$(TOOLDIR)/stl --transform "1:0:0:0:0:-360:0" --model HEX hexx.ra	;\
	$(TOOLDIR)/stl --transform "1:0:0:0:0:0:180" --model HEX hexy.ra	;\
	$(TOOLDIR)/stl --transform "1:0:0:0:0:0:180" --input hexy.ra hexyy.ra	;\
	$(TOOLDIR)/nrmse -t 1.e-15 hexz.ra $(TESTS_OUT)/hexahedron.ra		;\
	$(TOOLDIR)/nrmse -t 1.e-15 hexx.ra $(TESTS_OUT)/hexahedron.ra		;\
	$(TOOLDIR)/nrmse -t 1.e-16 hexyy.ra $(TESTS_OUT)/hexahedron.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-stl-rw tests/test-stl-read-ascii tests/test-stl-rw-binary

