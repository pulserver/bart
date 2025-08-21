


tests/test-pole: ones phasepole fmac nrmse conj
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phasepole -s --center 0.25:0.25:0. -x64:64:1 pole.ra				;\
	$(TOOLDIR)/conj pole.ra pole.ra								;\
	$(TOOLDIR)/phasepole -s --center 0.15:0.25:0. -x64:64:1 pole2.ra				;\
	$(TOOLDIR)/fmac pole.ra pole2.ra pole.ra						;\
	$(TOOLDIR)/phasepole pole.ra det_pole.ra							;\
	$(TOOLDIR)/nrmse -t 0.05 pole.ra det_pole.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pole-nlinv: ones phasepole nlinv fmac nrmse conj zeros join $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phasepole -s --center -0.25:0.05:0. -x128:128:1 o.ra				;\
	$(TOOLDIR)/zeros 4 128 128 1 8 z.ra							;\
	$(TOOLDIR)/join 3 o.ra z.ra i.ra							;\
	$(TOOLDIR)/nlinv --sens-os=1.5 --phase-pole=8 -i10 -Ii.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra img.ra col.ra	;\
	$(TOOLDIR)/phasepole col.ra cpole.ra 							;\
	$(TOOLDIR)/ones 2 128 128 o.ra								;\
	$(TOOLDIR)/nrmse -t 0. cpole.ra o.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pole-nlinv-3D: vec reshape phasepole zeros join phantom resize nlinv
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/vec -- -10 -10 -10 10 10 10 t.ra						;\
	$(TOOLDIR)/reshape 3 3 2 t.ra r.ra							;\
	$(TOOLDIR)/phasepole -s -x 48:48:48 r.ra o.ra						;\
	$(TOOLDIR)/zeros 4 48 48 48 4 z.ra							;\
	$(TOOLDIR)/join 3 o.ra z.ra i.ra							;\
	$(TOOLDIR)/phantom -s4 -k -3 -x32 k.ra							;\
	$(TOOLDIR)/resize -c 0 48 1 48 2 48 k.ra k.ra						;\
	$(TOOLDIR)/nlinv -N -S --cgiter=4 -a32 --sens-os=1.5 -w-1000 -i12 --phase-pole=8 -M0.001 -Ii.ra k.ra img1.ra col1.ra ;\
	$(TOOLDIR)/nlinv -N -S --cgiter=4 -a32 --sens-os=1.5 -w-1000 -i12                -M0.001 -Ii.ra k.ra img2.ra col2.ra ;\
	$(TOOLDIR)/phasepole -e col1.ra r1.ra							;\
	$(TOOLDIR)/phasepole -e col2.ra r2.ra							;\
	if [    -f r1.ra ]; then exit 1 ; fi							;\
	if [ !  -f r2.ra ]; then exit 1 ; fi							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-pole
TESTS += tests/test-pole-nlinv

TESTS_SLOW += tests/test-pole-nlinv-3D

