
tests/test-unwrap: unwrap zexp carg index scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/index 0 400 i.ra								;\
	$(TOOLDIR)/scale 0.01 i.ra i.ra								;\
	$(TOOLDIR)/scale 1.i i.ra o.ra								;\
	$(TOOLDIR)/zexp o.ra o.ra								;\
	$(TOOLDIR)/carg o.ra o.ra								;\
	$(TOOLDIR)/unwrap 1 o.ra u.ra								;\
	$(TOOLDIR)/nrmse -t 1.e-7 i.ra u.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-unwrap-lap: unwrap zexp carg index scale flip join saxpy ones nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/index 0 401 i.ra								;\
	$(TOOLDIR)/scale 0.01 i.ra i.ra								;\
	$(TOOLDIR)/flip 1 i.ra i2.ra								;\
	$(TOOLDIR)/join 0 i.ra i2.ra i.ra							;\
	$(TOOLDIR)/ones 1 802 o.ra								;\
	$(TOOLDIR)/saxpy -- -2 o.ra i.ra i.ra							;\
	$(TOOLDIR)/scale 1.i i.ra o.ra								;\
	$(TOOLDIR)/zexp o.ra o.ra								;\
	$(TOOLDIR)/carg o.ra o.ra								;\
	$(TOOLDIR)/unwrap -l 1 o.ra u.ra							;\
	$(TOOLDIR)/nrmse -t 5.e-4 i.ra u.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $

TESTS += tests/test-unwrap
TESTS += tests/test-unwrap-lap

