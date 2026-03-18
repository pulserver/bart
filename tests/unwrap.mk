
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

TESTS += tests/test-unwrap

